package com.imagecleaner.app.ui

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.PorterDuff
import android.net.Uri
import android.util.Log
import androidx.compose.ui.geometry.Offset
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.imagecleaner.app.core.ModelManifest
import com.imagecleaner.app.core.NativeEnhancer
import com.imagecleaner.app.core.NativeEnhancer.MODE_AUTO
import com.imagecleaner.app.core.NativeEnhancer.MODE_COLORIZE
import com.imagecleaner.app.core.NativeEnhancer.MODE_FACES
import com.imagecleaner.app.core.NativeEnhancer.MODE_INPAINT
import com.imagecleaner.app.core.NativeEnhancer.MODE_UPSCALE
import com.imagecleaner.app.data.ModelManager
import com.imagecleaner.app.data.PhotoIO
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainViewModel(app: Application) : AndroidViewModel(app) {
    companion object {
        private const val TAG = "PixelUp"
        // Undo snapshots are kept compressed: a 2000 px result is ~1 MB as
        // JPEG, versus tens of MB as a bitmap.
        private const val MAX_HISTORY = 8
        private const val HISTORY_QUALITY = 95
    }

    data class Progress(val stage: Int, val percent: Float, val message: String)

    data class UiState(
        val modelState: ModelManager.State = ModelManager.State.Checking,
        /** Current image: the picked photo, then each processed result. */
        val image: Bitmap? = null,
        /** The image the last edit was applied to, only while comparing. */
        val compareImage: Bitmap? = null,
        val comparing: Boolean = false,
        /** A previous version exists and can be restored. */
        val canUndo: Boolean = false,
        /** Edits made since the last save / photo pick. */
        val dirty: Boolean = false,
        val mode: Int = MODE_AUTO,
        val scale: Int = 2,
        val fidelity: Float = 0.7f,
        val colorStrength: Float = 1.15f,  // chroma multiplier for colorization
        val brush: Float = 0.06f,          // brush radius, fraction of image width
        val maskVersion: Int = 0,          // bumped on every stroke change
        val hasMask: Boolean = false,
        val canUndoStroke: Boolean = false,
        val running: Boolean = false,
        val progress: Progress = Progress(0, 0f, ""),
        val notice: String? = null
    )

    private val _ui = MutableStateFlow(UiState())
    val ui: StateFlow<UiState> = _ui.asStateFlow()

    private val manifest: ModelManifest = ModelManifest.parse(
        app.assets.open("models/models.json").bufferedReader().use { it.readText() }
    )

    private val modelManager = ModelManager(app)
    private var handle: Long = 0L
    private var creating = false

    /** Compressed snapshots of the images before each edit, oldest first. */
    private val history = ArrayDeque<ByteArray>()

    /** Paint layer for object removal; alpha = painted area, image-sized. */
    private var maskBitmap: Bitmap? = null
    private val maskPaint = android.graphics.Paint().apply {
        color = android.graphics.Color.WHITE
        style = android.graphics.Paint.Style.FILL
        isAntiAlias = true
    }

    // Each stroke is kept as its points so it can be replayed when undone.
    private data class Stroke(val points: List<Offset>, val radius: Float)
    private val strokes = mutableListOf<Stroke>()
    private var pendingPoints: MutableList<Offset>? = null
    private var pendingRadius = 1f

    init {
        viewModelScope.launch {
            modelManager.state.collect { state ->
                _ui.value = _ui.value.copy(modelState = state)
                if (state is ModelManager.State.Ready && handle == 0L && !creating) {
                    // The flag closes the window between the check and the
                    // assignment below: Ready can be re-emitted while the
                    // native pipeline is still loading its models, which used
                    // to create (and leak) a second handle.
                    creating = true
                    try {
                        withContext(Dispatchers.IO) {
                            handle = NativeEnhancer.create(
                                modelDir = modelManager.modelDir.absolutePath,
                                manifestJson = manifest.raw,
                                threads = Runtime.getRuntime()
                                    .availableProcessors().coerceIn(2, 8),
                                useVulkan = true
                            )
                        }
                        creating = false
                    } catch (e: Exception) {
                        creating = false
                        _ui.value = _ui.value.copy(
                            modelState = ModelManager.State.Error(
                                "推理引擎初始化失败：${e.message}"
                            )
                        )
                    }
                }
            }
        }
        retryModels()
    }

    fun retryModels() {
        viewModelScope.launch(Dispatchers.IO) {
            modelManager.ensureModels(manifest)
        }
    }

    fun setMode(mode: Int) {
        _ui.value = _ui.value.copy(mode = mode, comparing = false, compareImage = null)
        // Models load lazily; the inpainting one is ~100 MB and costs several
        // seconds. Start it when the user picks the mode, not on the first
        // tap, so painting time hides the load.
        val h = handle
        if (h != 0L && mode != MODE_AUTO) {
            viewModelScope.launch(Dispatchers.IO) {
                try {
                    NativeEnhancer.preload(h, mode)
                } catch (e: Exception) {
                    Log.w(TAG, "model preload failed", e)
                }
            }
        }
    }

    fun setScale(scale: Int) {
        _ui.value = _ui.value.copy(scale = scale)
    }

    fun setFidelity(value: Float) {
        _ui.value = _ui.value.copy(fidelity = value)
    }

    fun setColorStrength(value: Float) {
        _ui.value = _ui.value.copy(colorStrength = value.coerceIn(0.6f, 2f))
    }

    fun setBrush(value: Float) {
        _ui.value = _ui.value.copy(brush = value.coerceIn(0.01f, 0.25f))
    }

    /** The paint layer at source resolution, or null before a photo is loaded. */
    fun maskBitmap(): Bitmap? = maskBitmap

    // ---- painting -------------------------------------------------------

    fun beginStroke() {
        pendingPoints = mutableListOf()
        pendingRadius = 1f
    }

    fun paintMask(cx: Float, cy: Float, radius: Float) {
        val mask = maskBitmap ?: return
        val r = radius.coerceAtLeast(1f)
        Canvas(mask).drawCircle(cx, cy, r, maskPaint)
        val points = pendingPoints
        if (points != null) {
            if (points.isEmpty()) pendingRadius = r
            points.add(Offset(cx, cy))
        }
        _ui.value = _ui.value.copy(
            maskVersion = _ui.value.maskVersion + 1,
            hasMask = true
        )
    }

    fun endStroke() {
        val points = pendingPoints ?: return
        pendingPoints = null
        if (points.isEmpty()) return
        strokes.add(Stroke(points, pendingRadius))
        _ui.value = _ui.value.copy(hasMask = true, canUndoStroke = true)
    }

    /** Removes the last stroke; the remaining ones are replayed onto a clean layer. */
    fun undoStroke() {
        if (strokes.isEmpty()) return
        strokes.removeAt(strokes.size - 1)
        redrawMask()
        _ui.value = _ui.value.copy(
            maskVersion = _ui.value.maskVersion + 1,
            hasMask = strokes.isNotEmpty(),
            canUndoStroke = strokes.isNotEmpty()
        )
    }

    fun clearMask() {
        strokes.clear()
        pendingPoints = null
        redrawMask()
        _ui.value = _ui.value.copy(
            maskVersion = _ui.value.maskVersion + 1,
            hasMask = false,
            canUndoStroke = false
        )
    }

    private fun redrawMask() {
        val mask = maskBitmap ?: return
        val canvas = Canvas(mask)
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
        for (stroke in strokes) {
            for (p in stroke.points) {
                canvas.drawCircle(p.x, p.y, stroke.radius, maskPaint)
            }
        }
    }

    private fun resetMask(image: Bitmap?) {
        maskBitmap = image?.let {
            Bitmap.createBitmap(it.width, it.height, Bitmap.Config.ARGB_8888)
        }
        strokes.clear()
        pendingPoints = null
        _ui.value = _ui.value.copy(
            maskVersion = 0,
            hasMask = false,
            canUndoStroke = false
        )
    }

    // ---- history ---------------------------------------------------------

    private fun pushHistory(bitmap: Bitmap) {
        val out = ByteArrayOutputStream()
        if (!bitmap.compress(Bitmap.CompressFormat.JPEG, HISTORY_QUALITY, out)) {
            Log.w(TAG, "history snapshot failed")
            return
        }
        history.addLast(out.toByteArray())
        while (history.size > MAX_HISTORY) history.removeFirst()
    }

    private fun decodeSnapshot(bytes: ByteArray): Bitmap? =
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
        })

    private fun applyEdit(result: Bitmap) {
        _ui.value = _ui.value.copy(
            image = result,
            compareImage = null,
            comparing = false,
            canUndo = history.isNotEmpty(),
            dirty = true,
            running = false
        )
        resetMask(result)
    }

    /** Back to the image before the last edit. */
    fun undoResult() {
        if (_ui.value.running) return
        val snapshot = history.removeLastOrNull() ?: return
        viewModelScope.launch {
            // Keep other edits out until the snapshot is on screen.
            _ui.value = _ui.value.copy(running = true)
            val restored = withContext(Dispatchers.Default) { decodeSnapshot(snapshot) }
            if (restored == null) {
                _ui.value = _ui.value.copy(running = false, notice = "无法还原上一步")
                return@launch
            }
            _ui.value = _ui.value.copy(
                image = restored,
                compareImage = null,
                comparing = false,
                canUndo = history.isNotEmpty(),
                dirty = history.isNotEmpty(),
                running = false
            )
            resetMask(restored)
        }
    }

    /** Toggles between the current image and the one the last edit started from. */
    fun toggleCompare() {
        val state = _ui.value
        if (state.comparing) {
            _ui.value = state.copy(comparing = false, compareImage = null)
            return
        }
        val snapshot = history.lastOrNull() ?: return
        viewModelScope.launch {
            val previous = withContext(Dispatchers.Default) { decodeSnapshot(snapshot) }
                ?: return@launch
            _ui.value = _ui.value.copy(comparing = true, compareImage = previous)
        }
    }

    // ---- editing ---------------------------------------------------------

    fun consumeNotice() {
        _ui.value = _ui.value.copy(notice = null)
    }

    fun pickImage(uri: Uri) {
        viewModelScope.launch {
            try {
                val bitmap = withContext(Dispatchers.IO) {
                    PhotoIO.decode(
                        getApplication(), uri,
                        maxLongEdgeFor(_ui.value.mode, _ui.value.scale)
                    )
                }
                history.clear()
                _ui.value = _ui.value.copy(
                    image = bitmap,
                    compareImage = null,
                    comparing = false,
                    canUndo = false,
                    dirty = false
                )
                resetMask(bitmap)
            } catch (e: Exception) {
                _ui.value = _ui.value.copy(notice = "无法读取图片：${e.message}")
            }
        }
    }

    fun enhance() {
        val state = _ui.value
        val image = state.image ?: return
        if (state.running || handle == 0L) return
        if (state.mode == MODE_INPAINT) {
            inpaint(image)
            return
        }

        viewModelScope.launch {
            _ui.value = _ui.value.copy(
                running = true,
                comparing = false,
                compareImage = null,
                progress = Progress(0, 0f, "准备中")
            )
            try {
                val (input, downscaled) = withContext(Dispatchers.Default) {
                    prepareInput(image, state.mode, state.scale)
                }
                val result = withContext(Dispatchers.Default) {
                    runNative(input, state.mode, state.fidelity, state.colorStrength, state.scale)
                }
                if (input !== image) input.recycle()
                withContext(Dispatchers.Default) { pushHistory(image) }
                applyEdit(result)
                _ui.value = _ui.value.copy(
                    progress = Progress(4, 1f, "完成"),
                    notice = if (downscaled)
                        "为保证流畅，较大的图片已先缩小再处理" else null
                )
            } catch (e: Exception) {
                Log.e(TAG, "enhancement failed", e)
                _ui.value = _ui.value.copy(
                    running = false,
                    notice = "处理失败：${e.message}"
                )
            }
        }
    }

    /** LaMa object removal over the painted mask. */
    private fun inpaint(image: Bitmap) {
        val mask = maskBitmap ?: return
        if (!_ui.value.hasMask) {
            _ui.value = _ui.value.copy(notice = "请先涂抹要消除的区域")
            return
        }
        viewModelScope.launch {
            _ui.value = _ui.value.copy(
                running = true, comparing = false, compareImage = null,
                progress = Progress(6, 0f, "消除瑕疵")
            )
            try {
                val result = withContext(Dispatchers.Default) {
                    val w = image.width
                    val h = image.height
                    val src = ByteBuffer.allocateDirect(w * h * 4)
                        .order(ByteOrder.nativeOrder())
                    image.copyPixelsToBuffer(src)

                    // Alpha channel of the paint layer becomes the mask bytes.
                    val argb = IntArray(w * h)
                    mask.getPixels(argb, 0, w, 0, 0, w, h)
                    val maskBytes = ByteBuffer.allocateDirect(w * h)
                    val chunk = ByteArray(w * h)
                    for (i in argb.indices) {
                        chunk[i] = if (Color.alpha(argb[i]) >= 128) 255.toByte() else 0
                    }
                    maskBytes.put(chunk)
                    maskBytes.rewind()

                    val dst = ByteBuffer.allocateDirect(w * h * 4)
                        .order(ByteOrder.nativeOrder())
                    NativeEnhancer.inpaint(
                        handle, src, w, h, maskBytes, dst
                    ) { _, percent, message ->
                        _ui.value = _ui.value.copy(
                            progress = Progress(6, percent, message)
                        )
                    }
                    dst.rewind()
                    Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also {
                        it.copyPixelsFromBuffer(dst)
                    }
                }
                withContext(Dispatchers.Default) { pushHistory(image) }
                applyEdit(result)
                _ui.value = _ui.value.copy(progress = Progress(6, 1f, "完成"))
            } catch (e: Exception) {
                Log.e(TAG, "inpaint failed", e)
                _ui.value = _ui.value.copy(
                    running = false,
                    notice = "消除失败：${e.message}"
                )
            }
        }
    }

    /** Fits the bitmap to the memory budget for the requested scale. */
    private fun prepareInput(source: Bitmap, mode: Int, scale: Int): Pair<Bitmap, Boolean> {
        val cap = maxLongEdgeFor(mode, scale)
        val longEdge = maxOf(source.width, source.height)
        if (longEdge <= cap) return source to false
        val ratio = cap.toFloat() / longEdge
        val w = (source.width * ratio).toInt().coerceAtLeast(1)
        val h = (source.height * ratio).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(source, w, h, true) to true
    }

    private fun runNative(
        bitmap: Bitmap, mode: Int, fidelity: Float, colorStrength: Float, scale: Int
    ): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val src = ByteBuffer.allocateDirect(w * h * 4).order(ByteOrder.nativeOrder())
        bitmap.copyPixelsToBuffer(src)

        val outScale = if (mode == MODE_FACES || mode == MODE_COLORIZE) 1 else scale
        val outW = w * outScale
        val outH = h * outScale
        val dst = ByteBuffer.allocateDirect(outW * outH * 4).order(ByteOrder.nativeOrder())

        val (realW, realH) = NativeEnhancer.enhance(
            handle, src, w, h, dst, mode, fidelity, colorStrength, scale,
        ) { stage, percent, message ->
            _ui.value = _ui.value.copy(progress = Progress(stage, percent, message))
        }
        check(realW == outW && realH == outH) {
            "native returned ${realW}x${realH}, expected ${outW}x${outH}"
        }
        dst.rewind()
        val out = Bitmap.createBitmap(realW, realH, Bitmap.Config.ARGB_8888)
        out.copyPixelsFromBuffer(dst)
        return out
    }

    // ---- output ----------------------------------------------------------

    /** Saves the current image; [onSaved] runs on the main thread after success. */
    fun saveResult(onSaved: (() -> Unit)? = null) {
        val image = _ui.value.image ?: return
        viewModelScope.launch {
            try {
                val name = withContext(Dispatchers.IO) {
                    PhotoIO.saveToGallery(getApplication(), image)
                }
                _ui.value = _ui.value.copy(
                    notice = "已保存到相册：Pictures/PixelUp/$name",
                    dirty = false
                )
                onSaved?.invoke()
            } catch (e: Exception) {
                _ui.value = _ui.value.copy(notice = "保存失败：${e.message}")
            }
        }
    }

    fun shareResult() {
        val image = _ui.value.image ?: return
        viewModelScope.launch {
            val intent = withContext(Dispatchers.IO) {
                PhotoIO.share(getApplication(), image)
            }
            getApplication<android.app.Application>().startActivity(
                intent.addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
            )
        }
    }

    override fun onCleared() {
        if (handle != 0L) NativeEnhancer.nativeDestroy(handle)
    }

    private fun maxLongEdgeFor(mode: Int, scale: Int): Int = when {
        // Colorization never upscales: it only needs the L channel at full
        // resolution plus a 256x256 network pass, so a moderate cap is enough.
        mode == NativeEnhancer.MODE_COLORIZE -> 2048
        mode == NativeEnhancer.MODE_FACES -> maxLongEdgeFor(1)
        else -> maxLongEdgeFor(scale)
    }

    private fun maxLongEdgeFor(scale: Int): Int = when (scale) {
        4 -> 960   // 4x float accumulation stays within a ~300 MB native budget
        2 -> 1536
        else -> 2048
    }
}
