package com.imagecleaner.app.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import kotlin.math.roundToInt

/**
 * Shows the photo letterboxed, with the painted mask tinted on top, and turns
 * drags into paint strokes in image coordinates.
 *
 * The mask lives at the source image's resolution, so strokes stay accurate
 * regardless of how the preview is scaled on screen.
 */
@Composable
fun PaintView(
    image: ImageBitmap,
    mask: ImageBitmap?,
    maskVersion: Int,
    brushRadiusImage: Float,
    onPaint: (cx: Float, cy: Float) -> Unit,
    modifier: Modifier = Modifier
) {
    var boxSize by remember { mutableStateOf(Size.Zero) }

    // Layout of the image inside the box (letterboxed, Fit).
    fun imageRect(): Pair<Offset, Size> {
        if (boxSize.width <= 0f || boxSize.height <= 0f) return Offset.Zero to Size.Zero
        val scale = minOf(boxSize.width / image.width, boxSize.height / image.height)
        val w = image.width * scale
        val h = image.height * scale
        return Offset((boxSize.width - w) / 2f, (boxSize.height - h) / 2f) to Size(w, h)
    }

    Box(
        modifier
            .onSizeChanged {
                boxSize = Size(it.width.toFloat(), it.height.toFloat())
            }
            .pointerInput(brushRadiusImage) {
                awaitPointerEventScope {
                    var last: Offset? = null
                    while (true) {
                        val event = awaitPointerEvent()
                        val pos = event.changes.firstOrNull()?.position ?: continue
                        val (origin, size) = imageRect()
                        if (size.width <= 0f) continue
                        val inside = pos.x >= origin.x && pos.x < origin.x + size.width &&
                            pos.y >= origin.y && pos.y < origin.y + size.height
                        when (event.type) {
                            PointerEventType.Press, PointerEventType.Move -> {
                                if (!inside) { last = null; continue }
                                val ix = (pos.x - origin.x) / size.width * image.width
                                val iy = (pos.y - origin.y) / size.height * image.height
                                // Interpolate between events so fast drags stay continuous.
                                val prev = last
                                if (prev != null && event.type == PointerEventType.Move) {
                                    val steps = (Offset(ix, iy) - prev).getDistance() /
                                        (brushRadiusImage * 0.5f).coerceAtLeast(1f)
                                    val n = steps.toInt().coerceIn(1, 32)
                                    for (i in 1..n) {
                                        val t = i.toFloat() / n
                                        onPaint(prev.x + (ix - prev.x) * t,
                                                prev.y + (iy - prev.y) * t)
                                    }
                                } else {
                                    onPaint(ix, iy)
                                }
                                last = Offset(ix, iy)
                                event.changes.forEach { it.consume() }
                            }
                            PointerEventType.Release -> last = null
                        }
                    }
                }
            }
    ) {
        androidx.compose.foundation.Canvas(Modifier.fillMaxSize()) {
            val (origin, size) = imageRect()
            if (size.width <= 0f) return@Canvas
            drawImage(
                image,
                dstOffset = IntOffset(origin.x.roundToInt(), origin.y.roundToInt()),
                dstSize = IntSize(size.width.roundToInt(), size.height.roundToInt()),
                filterQuality = FilterQuality.Medium
            )
            if (mask != null && maskVersion >= 0) {
                // Tint the painted area so the user sees what will be removed.
                drawImage(
                    mask,
                    dstOffset = IntOffset(origin.x.roundToInt(), origin.y.roundToInt()),
                    dstSize = IntSize(size.width.roundToInt(), size.height.roundToInt()),
                    alpha = 0.45f,
                    colorFilter = ColorFilter.tint(Color(0xFFFF4D6D)),
                    filterQuality = FilterQuality.Medium
                )
            }
        }
    }
}
