package com.imagecleaner.app.ui

import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import kotlin.math.roundToInt

/**
 * Before/after comparison: the left part of the divider shows the original,
 * the right part shows the result (Fit content scaling, letterboxed).
 */
@Composable
fun CompareView(
    before: ImageBitmap,
    after: ImageBitmap?,
    modifier: Modifier = Modifier
) {
    var fraction by remember { mutableFloatStateOf(0.5f) }
    val base = after ?: before

    Box(modifier) {
        androidx.compose.foundation.Canvas(
            Modifier
                .fillMaxSize()
                .pointerInput(base) {
                    detectTapGestures { offset ->
                        fraction = (offset.x / size.width).coerceIn(0.02f, 0.98f)
                    }
                }
                .pointerInput(base) {
                    detectHorizontalDragGestures { _, dragAmount ->
                        fraction = (fraction + dragAmount / size.width).coerceIn(0f, 1f)
                    }
                }
        ) {
            val scale = minOf(size.width / base.width, size.height / base.height)
            val dw = base.width * scale
            val dh = base.height * scale
            val ox = ((size.width - dw) / 2).roundToInt()
            val oy = ((size.height - dh) / 2).roundToInt()
            val dstSize = IntSize(dw.roundToInt(), dh.roundToInt())

            // Result fills the whole frame (or, when absent, the original).
            drawImage(after ?: before, dstOffset = IntOffset(ox, oy), dstSize = dstSize)

            if (after != null) {
                val dividerX = ox + fraction * dw
                clipRect(right = dividerX) {
                    drawImage(before, dstOffset = IntOffset(ox, oy), dstSize = dstSize)
                }
                drawLine(
                    color = Color.White,
                    start = Offset(dividerX, oy.toFloat()),
                    end = Offset(dividerX, oy + dh),
                    strokeWidth = 3f
                )
                drawCircle(
                    color = Color.White,
                    radius = 16f,
                    center = Offset(dividerX, size.height / 2),
                    alpha = 0.9f
                )
            }
        }
    }
}
