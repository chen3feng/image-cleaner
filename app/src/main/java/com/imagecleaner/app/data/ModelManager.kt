package com.imagecleaner.app.data

import android.content.Context
import android.util.Log
import androidx.annotation.WorkerThread
import com.imagecleaner.app.core.ModelFile
import com.imagecleaner.app.core.ModelManifest
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

/**
 * Verifies the on-disk model directory and lazily downloads missing files from
 * the URL declared in models.json. Downloads resume via temp parts are not
 * attempted; models are immutable artifacts keyed by SHA-256.
 */
class ModelManager(context: Context) {

    sealed interface State {
        data object Checking : State
        data class Downloading(val index: Int, val total: Int, val name: String, val percent: Float) : State
        data object Ready : State
        data class Error(val message: String) : State
    }

    val modelDir: File = File(context.filesDir, "models").apply { mkdirs() }

    private val _state = MutableStateFlow<State>(State.Checking)
    val state: StateFlow<State> = _state.asStateFlow()

    private val http = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    @WorkerThread
    fun ensureModels(manifest: ModelManifest) {
        _state.value = State.Checking
        try {
            manifest.files.forEachIndexed { index, file ->
                val target = File(modelDir, file.path)
                if (isValid(target, file)) return@forEachIndexed
                download(manifest, file, index, manifest.files.size)
            }
            _state.value = State.Ready
        } catch (e: Exception) {
            Log.e("PixelUp", "model ensure failed", e)
            _state.value = State.Error(e.message ?: "model download failed")
        }
    }

    private fun isValid(target: File, file: ModelFile): Boolean {
        if (!target.isFile || target.length() == 0L) return false
        if (file.size > 0 && target.length() != file.size) return false
        if (file.hashKnown && sha256(target) != file.sha256.lowercase()) return false
        return true
    }

    private fun download(manifest: ModelManifest, file: ModelFile, index: Int, total: Int) {
        val url = manifest.baseUrl.trimEnd('/') + "/" + file.path
        val part = File(modelDir, file.path + ".part")
        val target = File(modelDir, file.path)
        target.parentFile?.mkdirs()

        _state.value = State.Downloading(index, total, file.path, 0f)
        val request = Request.Builder().url(url).build()
        http.newCall(request).execute().use { response ->
            if (!response.isSuccessful) error("HTTP ${response.code} for ${file.path}")
            val body = response.body ?: error("empty body for ${file.path}")
            val expected = body.contentLength().takeIf { it > 0 } ?: file.size.toLong()
            var seen = 0L
            part.outputStream().use { out ->
                body.byteStream().use { input ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        val read = input.read(buffer)
                        if (read == -1) break
                        out.write(buffer, 0, read)
                        seen += read
                        if (expected > 0) {
                            _state.value = State.Downloading(
                                index, total, file.path, seen.toFloat() / expected
                            )
                        }
                    }
                }
            }
        }

        if (file.hashKnown && sha256(part) != file.sha256.lowercase()) {
            part.delete()
            error("checksum mismatch for ${file.path}")
        }
        if (!part.renameTo(target)) {
            part.copyTo(target, overwrite = true)
            part.delete()
        }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(256 * 1024)
            while (true) {
                val read = input.read(buffer)
                if (read == -1) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
