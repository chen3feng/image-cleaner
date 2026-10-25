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
        outer@ for ((attempt, i) in order.withIndex()) {
            // Keep the partial file across sources: every mirror serves the
            // same artifact, so the next attempt (or the user's retry) resumes
            // instead of re-downloading ~100 MB on a flaky mobile link.
            try {
                _state.value = State.Downloading(index, total, file.path, 0f)
                fetch(bases[i] + "/" + file.path, part, file, index, total)
            } catch (e: Exception) {
                lastError = e
                Log.w("PixelUp", "download attempt ${attempt + 1}/${bases.size} failed for " +
                    "${file.path}: ${e.message}")
                continue@outer
            }
            // The transfer finished; a checksum failure is not worth retrying
            // against another mirror with the same bytes already on disk.
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
        }
        throw IOException(
            "all ${bases.size} sources failed for ${file.path}: ${lastError?.message}",
            lastError
        )
    }

    private fun fetch(url: String, part: File, file: ModelFile, index: Int, total: Int) {
        // Resume from a previous partial transfer when the server honours
        // Range; a 200 response means it does not, so restart from zero.
        val already = part.length().takeIf { it > 0 } ?: 0L
        if (file.size in 1..already) part.delete()
        val resumeFrom = part.length()
        val request = Request.Builder().url(url).apply {
            if (resumeFrom > 0) header("Range", "bytes=$resumeFrom-")
        }.build()

        http.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw IOException("HTTP ${response.code} for ${file.path}")
            val body = response.body ?: throw IOException("empty body for ${file.path}")
            val appending = response.code == 206 && resumeFrom > 0
            if (!appending && resumeFrom > 0) part.delete()
            val totalBytes = if (body.contentLength() > 0) {
                body.contentLength() + (if (appending) resumeFrom else 0)
            } else {
                file.size.toLong().takeIf { it > 0 } ?: -1
            }
            var seen = if (appending) resumeFrom else 0L
            java.io.FileOutputStream(part, appending).use { out ->
                body.byteStream().use { input ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        val read = input.read(buffer)
                        if (read == -1) break
                        out.write(buffer, 0, read)
                        seen += read
                        if (totalBytes > 0) {
                            _state.value = State.Downloading(
                                index, total, file.path,
                                (seen.toFloat() / totalBytes).coerceIn(0f, 1f)
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
