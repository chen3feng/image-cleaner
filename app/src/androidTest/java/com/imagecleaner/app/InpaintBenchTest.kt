package com.imagecleaner.app

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Bundle
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.imagecleaner.app.core.ModelManifest
import com.imagecleaner.app.core.NativeEnhancer
import com.imagecleaner.app.data.ModelManager
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Times LaMa on the real device through the same JNI path the app uses.
 *
 *   adb shell am instrument -w -e class com.imagecleaner.app.InpaintBenchTest \
 *       -e vulkan true com.imagecleaner.app.test/androidx.test.runner.AndroidJUnitRunner
 */
@RunWith(AndroidJUnit4::class)
class InpaintBenchTest {

    @Test
    fun benchmark() {
        val args = InstrumentationRegistry.getArguments()
        val useVulkan = args.getString("vulkan", "true")!!.toBoolean()
        val runs = args.getString("runs", "3")!!.toInt()
        val w = args.getString("w", "1440")!!.toInt()
        val h = args.getString("h", "1080")!!.toInt()

        val app = InstrumentationRegistry.getInstrumentation().targetContext
        val manifest = ModelManifest.parse(
            app.assets.open("models/models.json").bufferedReader().use { it.readText() }
        )
        val modelDir = File(app.filesDir, "models")
        val missing = manifest.files.filter { !File(modelDir, it.path).exists() }
        assertTrue("missing models: ${missing.map { it.path }}", missing.isEmpty())

        val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 8)
        val createStart = System.currentTimeMillis()
        val handle = NativeEnhancer.create(modelDir.absolutePath, manifest.raw, threads, useVulkan)
        val createMs = System.currentTimeMillis() - createStart
        try {
            val image = args.getString("photo")?.let { path ->
                android.graphics.BitmapFactory.decodeFile(path)
                    ?.let { Bitmap.createScaledBitmap(it, w, h, true) }
            } ?: syntheticPhoto(w, h)
            val mx = args.getString("mx", "520")!!.toFloat()
            val my = args.getString("my", "320")!!.toFloat()
            val mw = args.getString("mw", "50")!!.toFloat()
            val mh = args.getString("mh", "30")!!.toFloat()
            val mask = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            Canvas(mask).drawRect(mx, my, mx + mw, my + mh, Paint().apply {
                color = Color.WHITE
            })

            val src = ByteBuffer.allocateDirect(w * h * 4).order(ByteOrder.nativeOrder())
            image.copyPixelsToBuffer(src)
            val argb = IntArray(w * h)
            mask.getPixels(argb, 0, w, 0, 0, w, h)
            val maskBytes = ByteBuffer.allocateDirect(w * h)
            val chunk = ByteArray(w * h) { if (Color.alpha(argb[it]) >= 128) 255.toByte() else 0 }
            maskBytes.put(chunk)
            maskBytes.rewind()
            val dst = ByteBuffer.allocateDirect(w * h * 4).order(ByteOrder.nativeOrder())

            for (run in 0 until runs) {
                dst.rewind()
                val t0 = System.currentTimeMillis()
                NativeEnhancer.inpaint(handle, src, w, h, maskBytes, dst)
                val ms = System.currentTimeMillis() - t0
                android.util.Log.i(
                    "InpaintBench",
                    "run=$run vulkan=$useVulkan image=${w}x$h threads=$threads " +
                        "create=${createMs}ms inpaint=${ms}ms"
                )
                src.rewind()
            }

            if (args.getString("save", "false") == "true") {
                dst.rewind()
                val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                out.copyPixelsFromBuffer(dst)
                val dir = app.getExternalFilesDir(null)
                java.io.File(dir, "bench_out.png").outputStream().use {
                    out.compress(Bitmap.CompressFormat.PNG, 100, it)
                }
                java.io.File(dir, "bench_in.png").outputStream().use {
                    image.compress(Bitmap.CompressFormat.PNG, 100, it)
                }
                android.util.Log.i("InpaintBench", "saved to $dir")
            }
        } finally {
            NativeEnhancer.nativeDestroy(handle)
        }
    }

    /** A photo-like image: gradients plus fine noise so the net cannot skip work. */
    private fun syntheticPhoto(w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val px = IntArray(w * h)
        var seed = 12345
        for (y in 0 until h) {
            for (x in 0 until w) {
                seed = seed * 1103515245 + 12345
                val n = (seed ushr 16 and 31) - 16
                val r = (x * 255 / w + n).coerceIn(0, 255)
                val g = (y * 255 / h + n).coerceIn(0, 255)
                val b = ((x + y) * 255 / (w + h) + n).coerceIn(0, 255)
                px[y * w + x] = Color.rgb(r, g, b)
            }
        }
        bmp.setPixels(px, 0, w, 0, 0, w, h)
        return bmp
    }
}
