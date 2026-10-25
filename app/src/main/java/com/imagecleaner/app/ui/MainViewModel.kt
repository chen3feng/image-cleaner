package com.imagecleaner.app.ui

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.imagecleaner.app.core.ModelManifest
import com.imagecleaner.app.core.NativeEnhancer
import com.imagecleaner.app.core.NativeEnhancer.MODE_AUTO
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
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainViewModel(app: Application) : AndroidViewModel(app) {
    companion object { private const val TAG = "PixelUp" }

    data class Progress(val stage: Int, val percent: Float, val message: String)

    data class UiState(
        val modelState: ModelManager.State = ModelManager.State.Checking,
        val source: Bitmap? = null,
        val result: Bitmap? = null,
        val mode: Int = MODE_AUTO,
        val scale: Int = 2,
        val fidelity: Float = 0.7f,
        val brush: Float = 0.06f,          // brush radius, fraction of image width
        val maskVersion: Int = 0,          // bumped on every stroke
        val hasMask: Boolean = false,
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

    /** Paint layer for object removal; alpha = painted area, image-sized. */
    private var maskBitmap: Bitmap? = null
    private val maskPaint = android.graphics.Paint().apply {
        color = android.graphics.Color.WHITE
        style = android.graphics.Paint.Style.FILL
        isAntiAlias = true
    }
    private val maskCanvasPaint = maskPaint

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
        _ui.value = _ui.value.copy(mode = mode)
    }

    fun setScale(scale: Int) {
        _ui.value = _ui.value.copy(scale = scale)
    }

    fun setFidelity(value: Float) {
        _ui.value = _ui.value.copy(fidelity = value)
    }

    fun setBrush(value: Float) {
        _ui.value = _ui.value.copy(brush = value.coerceIn(0.01f, 0.25f))
    }

    /** The paint layer at source resolution, or null before a photo is loaded. */
    fun maskBitmap(): Bitmap? = maskBitmap

    private fun resetMask(source: Bitmap) {
        maskBitmap?.recycle()
        maskBitmap = Bitmap.createBitmap(
            source.width, source.height, Bitmap.Config.ARGB_8888
        )
        _ui.value = _ui.value.copy(maskVersion = 0, hasMask = false)
    }

    fun paintMask(cx: Float, cy: Float) {
        val mask = maskBitmap ?: return
        val radius = (_ui.value.brush * mask.width).coerceAtLeast(1f)
        android.graphics.Canvas(mask).drawCircle(cx, cy, radius, maskCanvasPaint)
        _ui.value = _ui.value.copy(
            maskVersion = _ui.value.maskVersion + 1,
            hasMask = true
        )
    }

    fun clearMask() {
        val source = _ui.value.source ?: return
        resetMask(source)
    }

    /** Back to painting: drop the last result and the strokes. */
    fun restartPaint() {
        val source = _ui.value.source ?: return
        _ui.value.result?.recycle()
        resetMask(source)
        _ui.value = _ui.value.copy(result = null)
    }

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
                _ui.value.source?.recycle()
                _ui.value.result?.recycle()
                resetMask(bitmap)
                _ui.value = _ui.value.copy(source = bitmap, result = null)
            } catch (e: Exception) {
                _ui.value = _ui.value.copy(notice = "无法读取图片：${e.message}")
            }
        }
    }

    fun enhance() {
        val state = _ui.value
        val source = state.source ?: return
        if (state.running || handle == 0L) return
        if (state.mode == MODE_INPAINT) {
            inpaint(source)
            return
        }

        viewModelScope.launch {
            state.result?.recycle()
            _ui.value = state.copy(running = true, result = null,
                progress = Progress(0, 0f, "准备中"))
            try {
                val (input, downscaled) = withContext(Dispatchers.Default) {
                    prepareInput(source, state.mode, state.scale)
                }
                val result = withContext(Dispatchers.Default) {
                    runNative(input, state.mode, state.fidelity, state.scale)
                }
                if (input != source) input.recycle()
                _ui.value = _ui.value.copy(
                    running = false,
                    result = result,
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
    private fun inpaint(source: Bitmap) {
        val mask = maskBitmap ?: return
        if (!_ui.value.hasMask) {
            _ui.value = _ui.value.copy(notice = "请先涂抹要消除的区域")
            return
        }
        viewModelScope.launch {
            _ui.value = _ui.value.copy(
                running = true, result = null, progress = Progress(6, 0f, "消除瑕疵")
            )
            try {
                val result = withContext(Dispatchers.Default) {
                    val w = source.width
                    val h = source.height
                    val src = ByteBuffer.allocateDirect(w * h * 4)
                        .order(ByteOrder.nativeOrder())
                    source.copyPixelsToBuffer(src)

                    // Alpha channel of the paint layer becomes the mask bytes.
                    val argb = IntArray(w * h)
                    mask.getPixels(argb, 0, w, 0, 0, w, h)
                    val maskBytes = ByteBuffer.allocateDirect(w * h)
                    val chunk = ByteArray(w * h)
                    for (i in argb.indices) {
                        chunk[i] = if (android.graphics.Color.alpha(argb[i]) >= 128) 255.toByte() else 0
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
                _ui.value = _ui.value.copy(
                    running = false, result = result, progress = Progress(6, 1f, "完成")
                )
            } catch (e: Exception) {
                Log.e(TAG, "inpaint failed", e)
                _ui.value = _ui.value.copy(
                    running = false, notice = "消除失败：${e.message}"
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

    private fun runNative(bitmap: Bitmap, mode: Int, fidelity: Float, scale: Int): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val src = ByteBuffer.allocateDirect(w * h * 4).order(ByteOrder.nativeOrder())
        bitmap.copyPixelsToBuffer(src)

        val outScale = if (mode == MODE_FACES || mode == NativeEnhancer.MODE_COLORIZE) 1 else scale
        val outW = w * outScale
        val outH = h * outScale
        val dst = ByteBuffer.allocateDirect(outW * outH * 4).order(ByteOrder.nativeOrder())

        val (realW, realH) = NativeEnhancer.enhance(
            handle, src, w, h, dst, mode, fidelity, scale,
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

    fun saveResult() {
        val result = _ui.value.result ?: return
        viewModelScope.launch {
            try {
                val name = withContext(Dispatchers.IO) {
                    PhotoIO.saveToGallery(getApplication(), result)
                }
                _ui.value = _ui.value.copy(notice = "已保存到相册：Pictures/PixelUp/$name")
            } catch (e: Exception) {
                _ui.value = _ui.value.copy(notice = "保存失败：${e.message}")
            }
        }
    }

    fun shareResult() {
        val result = _ui.value.result ?: return
        viewModelScope.launch {
            val intent = withContext(Dispatchers.IO) {
                PhotoIO.share(getApplication(), result)
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
