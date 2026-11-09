package com.imagecleaner.app.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.calculateCentroid
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import kotlin.math.min
import kotlin.math.roundToInt

private const val MAX_ZOOM = 6f

/**
 * Shows the photo (letterboxed, zoomable), with the painted mask tinted on top,
 * and turns drags into paint strokes in image coordinates.
 *
 * Gestures: one finger paints, two fingers pan and zoom. The mask lives at the
 * source image's resolution, so strokes stay accurate at any zoom level.
 */
@Composable
fun PaintView(
    image: ImageBitmap,
    mask: ImageBitmap?,
    maskVersion: Int,
    brushRadiusImage: Float,
    onStrokeStart: () -> Unit,
    onPaint: (cx: Float, cy: Float, radiusImage: Float) -> Unit,
    onStrokeEnd: () -> Unit,
    modifier: Modifier = Modifier
) {
    var boxSize by remember { mutableStateOf(Size.Zero) }
    var zoom by remember { mutableStateOf(1f) }
    var pan by remember { mutableStateOf(Offset.Zero) }

    fun baseRect(): Pair<Offset, Size> {
        if (boxSize.width <= 0f || boxSize.height <= 0f) return Offset.Zero to Size.Zero
        val scale = min(boxSize.width / image.width, boxSize.height / image.height)
        val w = image.width * scale
        val h = image.height * scale
        return Offset((boxSize.width - w) / 2f, (boxSize.height - h) / 2f) to Size(w, h)
    }

    fun clampPan(p: Offset, z: Float): Offset {
        val (_, base) = baseRect()
        val overX = ((base.width * z - boxSize.width) / 2f).coerceAtLeast(0f)
        val overY = ((base.height * z - boxSize.height) / 2f).coerceAtLeast(0f)
        return Offset(
            p.x.coerceIn(-overX, overX),
            p.y.coerceIn(-overY, overY)
        )
    }

    // Where the (zoomed) image is drawn inside the box.
    fun displayRect(): Pair<Offset, Size> {
        val (baseOrigin, base) = baseRect()
        if (base.width <= 0f) return Offset.Zero to Size.Zero
        val w = base.width * zoom
        val h = base.height * zoom
        val cx = baseOrigin.x + base.width / 2f
        val cy = baseOrigin.y + base.height / 2f
        return Offset(cx - w / 2f + pan.x, cy - h / 2f + pan.y) to Size(w, h)
    }

    // Pan/zoom around the gesture centroid so the pixel under the fingers stays put.
    fun transform(zoomChange: Float, panChange: Offset, centroid: Offset) {
        val oldZoom = zoom
        val newZoom = (oldZoom * zoomChange).coerceIn(1f, MAX_ZOOM)
        val (_, base) = baseRect()
        val ratio = newZoom / oldZoom
        val (origin, _) = displayRect()
        val moved = origin + panChange
        val u = centroid - moved
        val newOrigin = centroid - u * ratio
        val center = Offset(boxSize.width / 2f, boxSize.height / 2f)
        val newW = base.width * newZoom
        val newH = base.height * newZoom
        val nextPan = Offset(
            newOrigin.x - center.x + newW / 2f,
            newOrigin.y - center.y + newH / 2f
        )
        zoom = newZoom
        pan = if (newZoom <= 1.001f) Offset.Zero else clampPan(nextPan, newZoom)
    }

    Box(
        modifier
            .onSizeChanged {
                boxSize = Size(it.width.toFloat(), it.height.toFloat())
                pan = clampPan(pan, zoom)
            }
            .pointerInput(brushRadiusImage, image.width, image.height) {
                awaitPointerEventScope {
                    var last: Offset? = null      // last painted point, image coords
                    var pending: Offset? = null   // finger-down point, not painted yet
                    var ignore = false            // down landed on a child (e.g. reset)
                    var strokeStarted = false
                    // After a two-finger gesture, ignore the remaining finger
                    // until every pointer is lifted: otherwise lifting one
                    // finger mid-pinch would start painting a stray stroke.
                    var transforming = false
                    val slop = viewConfiguration.touchSlop
                    // The brush is specified in image pixels at 1x; zooming in
                    // shrinks it in image space so it keeps the same size on
                    // screen and gives finer control.
                    fun brushRadius(): Float = brushRadiusImage / zoom

                    fun beginStroke() {
                        if (!strokeStarted) {
                            strokeStarted = true
                            onStrokeStart()
                        }
                    }

                    fun endStroke() {
                        if (strokeStarted) {
                            strokeStarted = false
                            onStrokeEnd()
                        }
                    }

                    fun paintAt(screen: Offset) {
                        val (origin, size) = displayRect()
                        if (size.width <= 0f) return
                        val inside = screen.x >= origin.x &&
                            screen.x < origin.x + size.width &&
                            screen.y >= origin.y &&
                            screen.y < origin.y + size.height
                        if (!inside) {
                            last = null
                            return
                        }
                        beginStroke()
                        val radius = brushRadius()
                        val ix = (screen.x - origin.x) / size.width * image.width
                        val iy = (screen.y - origin.y) / size.height * image.height
                        // Interpolate between events so fast drags stay continuous.
                        val prev = last
                        if (prev != null) {
                            val steps = (Offset(ix, iy) - prev).getDistance() /
                                (radius * 0.5f).coerceAtLeast(1f)
                            val n = steps.toInt().coerceIn(1, 64)
                            for (i in 1..n) {
                                val t = i.toFloat() / n
                                onPaint(prev.x + (ix - prev.x) * t,
                                        prev.y + (iy - prev.y) * t, radius)
                            }
                        } else {
                            onPaint(ix, iy, radius)
                        }
                        last = Offset(ix, iy)
                    }

                    while (true) {
                        val event = awaitPointerEvent()
                        val pressed = event.changes.filter { it.pressed }

                        if (pressed.size >= 2) {
                            // Two fingers mean pan/zoom. Drop the pending dot
                            // the first finger would otherwise have left.
                            endStroke()
                            pending = null
                            last = null
                            transforming = true
                            val centroid = event.calculateCentroid(useCurrent = true)
                            if (centroid != Offset.Unspecified) {
                                transform(
                                    event.calculateZoom(),
                                    event.calculatePan(),
                                    centroid
                                )
                            }
                            event.changes.forEach { it.consume() }
                            continue
                        }

                        if (pressed.size == 1) {
                            val change = pressed[0]
                            if (transforming) {
                                change.consume()
                                continue
                            }
                            if (event.type == PointerEventType.Press) {
                                // Hold the down point instead of painting it:
                                // if a second finger lands, this is a pan and
                                // nothing should be drawn.
                                ignore = change.isConsumed
                                pending = if (ignore) null else change.position
                                last = null
                                continue
                            }
                            if (ignore) continue

                            val pos = change.position
                            val down = pending
                            if (down != null) {
                                if ((pos - down).getDistance() < slop) continue
                                // Past the slop this is a stroke, so include
                                // the point where the finger landed.
                                paintAt(down)
                                pending = null
                            }
                            paintAt(pos)
                            change.consume()
                            continue
                        }

                        // All pointers up: a tap with no movement still paints
                        // a dot, but only if no second finger was involved.
                        val down = pending
                        if (down != null && !transforming) paintAt(down)
                        endStroke()
                        pending = null
                        last = null
                        ignore = false
                        transforming = false
                    }
                }
            }
    ) {
        Canvas(Modifier.fillMaxSize()) {
            val (origin, size) = displayRect()
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

        Surface(
            modifier = Modifier.align(Alignment.TopEnd).padding(8.dp),
            color = MaterialTheme.colorScheme.surface.copy(alpha = 0.78f),
            shape = MaterialTheme.shapes.small
        ) {
            Text(
                "双指缩放 · 单指涂抹",
                style = MaterialTheme.typography.labelSmall,
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        if (zoom > 1.01f) {
            Surface(
                modifier = Modifier.align(Alignment.BottomEnd).padding(8.dp),
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f),
                shape = MaterialTheme.shapes.small
            ) {
                TextButton(onClick = {
                    zoom = 1f
                    pan = Offset.Zero
                }) { Text("重置视图") }
            }
        }
    }
}
