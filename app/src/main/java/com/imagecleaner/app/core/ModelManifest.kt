package com.imagecleaner.app.core

import com.imagecleaner.app.BuildConfig
import org.json.JSONObject

data class ModelFile(
    val id: String,
    val path: String,
    val size: Long,
    val sha256: String
) {
    val hashKnown: Boolean get() = sha256.length == 64 && sha256.all { it.isDigit() || it in 'a'..'f' }
}

data class ModelManifest(
    val version: Int,
    val baseUrl: String,
    val files: List<ModelFile>,
    val raw: String,
    // Alternative prefixes tried in order when the primary base times out
    // (GitHub's release CDN is frequently unreachable from mainland China).
    val mirrorBaseUrls: List<String> = emptyList()
) {
    companion object {
        fun parse(json: String): ModelManifest {
            val root = JSONObject(json)
            val filesJson = root.getJSONArray("files")
            val files = buildList {
                for (i in 0 until filesJson.length()) {
                    val f = filesJson.getJSONObject(i)
                    add(
                        ModelFile(
                            id = f.getString("id"),
                            path = f.getString("path"),
                            size = f.optLong("size", 0),
                            sha256 = f.optString("sha256", "")
                        )
                    )
                }
            }
            val urlFromManifest = root.getString("base_url")
            val override = BuildConfig.MODELS_BASE_URL_OVERRIDE.takeIf { it.isNotBlank() }
            val mirrorsJson = root.optJSONArray("mirror_base_urls")
            val mirrors = buildList {
                if (mirrorsJson != null) {
                    for (i in 0 until mirrorsJson.length()) add(mirrorsJson.getString(i))
                }
            }
            return ModelManifest(
                version = root.optInt("version", 1),
                // An explicit build-time override replaces both the primary and
                // the mirrors: it exists for self-hosting the bundles.
                baseUrl = override ?: urlFromManifest,
                files = files,
                raw = json,
                mirrorBaseUrls = if (override != null) emptyList() else mirrors
            )
        }
    }
}
