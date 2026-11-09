package com.imagecleaner.app.ui

import android.graphics.Bitmap
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onRoot
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.test.pinch
import androidx.compose.ui.test.swipe
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import kotlin.math.abs

/**
 * On-device checks for the erase-mode gestures. The pinch has to be injected
 * through Compose (adb's `input` only knows one pointer, and `sendevent` needs
 * root), so this runs as an instrumented test.
 */
class PaintViewZoomTest {

    @get:Rule
    val rule = createComposeRule()

    private val imageWidth = 600
    private val imageHeight = 800

    private fun blankImage(): ImageBitmap =
        Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888)
            .asImageBitmap()

    private fun showPaintView(
        painted: MutableList<Offset>,
        radii: MutableList<Float>? = null,
        strokeCount: () -> Unit = {}
    ) {
        rule.setContent {
            PaintView(
                image = blankImage(),
                mask = null,
                maskVersion = 0,
                brushRadiusImage = 30f,
                onStrokeStart = strokeCount,
                onPaint = { x, y, r ->
                    painted += Offset(x, y)
                    radii?.add(r)
                },
                onStrokeEnd = {},
                modifier = Modifier.fillMaxSize()
            )
        }
        rule.waitForIdle()
    }

    @Test
    fun pinchZoomsAndResetRestoresFit() {
        showPaintView(mutableListOf())

        rule.onNodeWithText("重置视图").assertDoesNotExist()

        rule.onRoot().performTouchInput {
            pinch(
                start0 = center - Offset(width * 0.10f, 0f),
                end0 = center - Offset(width * 0.28f, 0f),
                start1 = center + Offset(width * 0.10f, 0f),
                end1 = center + Offset(width * 0.28f, 0f),
            )
        }
        rule.waitForIdle()

        // The reset affordance only renders while zoom > 1.01, so this is a
        // direct read of the zoom state.
        rule.onNodeWithText("重置视图").assertExists()

        rule.onNodeWithText("重置视图").performClick()
        rule.waitForIdle()
        rule.onNodeWithText("重置视图").assertDoesNotExist()
    }

    @Test
    fun paintMapsBackThroughZoom() {
        val painted = mutableListOf<Offset>()
        val radii = mutableListOf<Float>()
        showPaintView(painted, radii)

        rule.onRoot().performTouchInput {
            pinch(
                start0 = center - Offset(width * 0.10f, 0f),
                end0 = center - Offset(width * 0.28f, 0f),
                start1 = center + Offset(width * 0.10f, 0f),
                end1 = center + Offset(width * 0.28f, 0f),
            )
        }
        rule.waitForIdle()
        rule.onNodeWithText("重置视图").assertExists()

        painted.clear()
        // A short drag across the (zoomed) centre must land near the image
        // centre, in image pixels, not screen pixels.
        rule.onRoot().performTouchInput {
            swipe(start = center, end = center + Offset(width * 0.06f, 0f))
        }
        rule.waitForIdle()

        assertFalse("no stroke reached onPaint", painted.isEmpty())
        val target = Offset(imageWidth / 2f, imageHeight / 2f)
        for (p in painted) {
            assertTrue("x out of image: $p", p.x in 0f..imageWidth.toFloat())
            assertTrue("y out of image: $p", p.y in 0f..imageHeight.toFloat())
        }
        val last = painted.last()
        assertTrue(
            "stroke did not map to the image centre: $last (want ~$target)",
            abs(last.x - target.x) < imageWidth * 0.15f &&
                abs(last.y - target.y) < imageHeight * 0.15f
        )
        // The brush is 30 px at 1x; after zooming in it must shrink in image
        // space so it keeps the same size on screen.
        assertTrue("brush did not shrink with zoom: ${radii.first()}",
            radii.isNotEmpty() && radii.first() < 30f * 0.6f)
    }

    @Test
    fun tappingResetDoesNotPaint() {
        val painted = mutableListOf<Offset>()
        showPaintView(painted)
        rule.onRoot().performTouchInput {
            pinch(
                start0 = center - Offset(width * 0.10f, 0f),
                end0 = center - Offset(width * 0.28f, 0f),
                start1 = center + Offset(width * 0.10f, 0f),
                end1 = center + Offset(width * 0.28f, 0f),
            )
        }
        rule.waitForIdle()
        rule.onNodeWithText("重置视图").assertExists()

        painted.clear()
        rule.onNodeWithText("重置视图").performTouchInput {
            down(center)
            up()
        }
        rule.waitForIdle()

        // The button consumed the tap, so the paint layer must stay untouched
        // and the view must be back to fit.
        assertTrue("tapping reset painted ${painted.size} points", painted.isEmpty())
        rule.onNodeWithText("重置视图").assertDoesNotExist()
    }

    @Test
    fun tapPaintsSingleDot() {
        val painted = mutableListOf<Offset>()
        showPaintView(painted)

        rule.onRoot().performTouchInput {
            down(center)
            up()
        }
        rule.waitForIdle()

        // A tap has no move events; the dot must still land when the finger
        // lifts, at the image centre.
        assertTrue("tap painted ${painted.size} points", painted.isNotEmpty())
        val target = Offset(imageWidth / 2f, imageHeight / 2f)
        for (p in painted) {
            assertTrue("tap did not map to the image centre: $p",
                (p - target).getDistance() < 2f)
        }
    }

    @Test
    fun twoFingerPanDoesNotPaint() {
        val painted = mutableListOf<Offset>()
        showPaintView(painted)

        rule.onRoot().performTouchInput {
            // The first finger lands and jitters before the second arrives —
            // exactly what happens when a user starts a pan.
            down(0, center)
            moveTo(0, center + Offset(10f, 0f))
            down(1, center + Offset(width * 0.2f, 0f))
            moveTo(0, center - Offset(width * 0.1f, 0f))
            moveTo(1, center + Offset(width * 0.3f, 0f))
            up(0)
            up(1)
        }
        rule.waitForIdle()

        assertTrue("two-finger pan painted ${painted.size} points: ${painted.take(3)}",
            painted.isEmpty())
    }
}
