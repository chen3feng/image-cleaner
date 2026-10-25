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
    val raw: String
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
            val url = BuildConfig.MODELS_BASE_URL_OVERRIDE
                .takeIf { it.isNotBlank() } ?: urlFromManifest
            return ModelManifest(
                version = root.optInt("version", 1),
                baseUrl = url,
                files = files,
                raw = json
            )
        }
    }
}
