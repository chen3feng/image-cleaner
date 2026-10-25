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
import java.io.IOException
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
        // Short connect timeout: an unreachable GitHub endpoint should fail
        // fast so the next mirror gets a turn (a photo-sized bundle is many
        // files, and each dead-source attempt used to cost 15s).
        .connectTimeout(8, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .callTimeout(10, TimeUnit.MINUTES)
        .retryOnConnectionFailure(true)
        .build()

    /** Index of the base URL that last worked; tried first for later files. */
    private var preferredBase = 0

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

    /**
     * Downloads one file, trying the primary base URL and then every mirror.
     * Mirrors matter in practice: GitHub's release CDN (objects.githubusercontent.com)
     * is often unreachable from mainland China even when github.com itself pings.
     */
    private fun download(manifest: ModelManifest, file: ModelFile, index: Int, total: Int) {
        val part = File(modelDir, file.path + ".part")
        val target = File(modelDir, file.path)
        target.parentFile?.mkdirs()

        val bases = buildList {
            add(manifest.baseUrl)
            addAll(manifest.mirrorBaseUrls)
        }.map { it.trimEnd('/') }

        // Start from whichever source worked last, then wrap around.
        val start = preferredBase.coerceIn(0, bases.lastIndex)
        val order = (bases.indices).map { (start + it) % bases.size }

        var lastError: Exception? = null
        for ((attempt, i) in order.withIndex()) {
            try {
                _state.value = State.Downloading(index, total, file.path, 0f)
                fetch(bases[i] + "/" + file.path, part, file, index, total)
                if (file.hashKnown && sha256(part) != file.sha256.lowercase()) {
                    part.delete()
                    throw IOException("checksum mismatch for ${file.path}")
                }
                if (!part.renameTo(target)) {
                    part.copyTo(target, overwrite = true)
                    part.delete()
                }
                preferredBase = i
                return
            } catch (e: Exception) {
                part.delete()
                lastError = e
                Log.w("PixelUp", "download attempt ${attempt + 1}/${bases.size} failed for " +
                    "${file.path}: ${e.message}")
            }
        }
        throw IOException(
            "all ${bases.size} sources failed for ${file.path}: ${lastError?.message}",
            lastError
        )
    }

    private fun fetch(url: String, part: File, file: ModelFile, index: Int, total: Int) {
        http.newCall(Request.Builder().url(url).build()).execute().use { response ->
            if (!response.isSuccessful) throw IOException("HTTP ${response.code} for ${file.path}")
            val body = response.body ?: throw IOException("empty body for ${file.path}")
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
