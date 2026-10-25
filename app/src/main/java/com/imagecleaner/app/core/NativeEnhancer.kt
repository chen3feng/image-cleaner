package com.imagecleaner.app.core

import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

/**
 * JNI bridge to libpixelup_core.so. A single long-lived handle is kept open by
 * the ViewModel; models load lazily on first use of each pipeline stage.
 */
object NativeEnhancer {

    const val MODE_AUTO = 0
    const val MODE_FACES = 1
    const val MODE_UPSCALE = 2

    init {
        System.loadLibrary("pixelup_core")
    }

    fun interface ProgressListener {
        fun onProgress(stage: Int, percent: Float, message: String)
    }

    private val callbacks = ConcurrentHashMap<Long, ProgressListener>()
    private val nextCallbackId = AtomicLong(1)

    @JvmStatic external fun nativeCreate(
        modelDir: String, manifestJson: String, threads: Int, useVulkan: Boolean
    ): Long

    @JvmStatic external fun nativeDestroy(handle: Long)
    @JvmStatic external fun nativeLastError(): String

    @JvmStatic external fun nativeEnhance(
        handle: Long,
        src: ByteBuffer, srcW: Int, srcH: Int,
        dst: ByteBuffer, dstCapacity: Int,
        mode: Int, fidelity: Float, targetScale: Int,
        callbackId: Long
    ): IntArray?

    /** Called from native code on the worker thread. */
    @JvmStatic
    fun dispatchProgress(id: Long, stage: Int, percent: Float, message: String) {
        callbacks[id]?.onProgress(stage, percent, message)
    }

    fun create(modelDir: String, manifestJson: String, threads: Int, useVulkan: Boolean): Long {
        val handle = nativeCreate(modelDir, manifestJson, threads, useVulkan)
        if (handle == 0L) throw IOException(nativeLastError())
        return handle
    }

    /** Returns output (width, height). Buffer contents are tightly packed RGBA. */
    fun enhance(
        handle: Long,
        src: ByteBuffer, srcW: Int, srcH: Int,
        dst: ByteBuffer,
        mode: Int, fidelity: Float, targetScale: Int,
        listener: ProgressListener? = null
    ): Pair<Int, Int> {
        val id = listener?.let {
            val newId = nextCallbackId.getAndIncrement()
            callbacks[newId] = it
            newId
        } ?: 0L
        try {
            val dims = nativeEnhance(
                handle, src, srcW, srcH, dst, dst.capacity(),
                mode, fidelity, targetScale, id
            ) ?: throw IOException(nativeLastError())
            return dims[0] to dims[1]
        } finally {
            callbacks.remove(id)
        }
    }
}
