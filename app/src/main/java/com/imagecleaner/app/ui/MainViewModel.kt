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

    fun consumeNotice() {
        _ui.value = _ui.value.copy(notice = null)
    }

    fun pickImage(uri: Uri) {
        viewModelScope.launch {
            try {
                val bitmap = withContext(Dispatchers.IO) {
                    PhotoIO.decode(getApplication(), uri, maxLongEdgeFor(_ui.value.scale))
                }
                _ui.value.source?.recycle()
                _ui.value.result?.recycle()
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

    /** Fits the bitmap to the memory budget for the requested scale. */
    private fun prepareInput(source: Bitmap, mode: Int, scale: Int): Pair<Bitmap, Boolean> {
        val cap = maxLongEdgeFor(if (mode == MODE_FACES) 1 else scale)
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

        val outScale = if (mode == MODE_FACES) 1 else scale
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

    private fun maxLongEdgeFor(scale: Int): Int = when (scale) {
        4 -> 960   // 4x float accumulation stays within a ~300 MB native budget
        2 -> 1536
        else -> 2048
    }
}
