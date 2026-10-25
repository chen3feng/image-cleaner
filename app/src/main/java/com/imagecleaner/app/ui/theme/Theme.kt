package com.imagecleaner.app.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val DarkColors = darkColorScheme(
    primary = Color(0xFF9DB0FF),
    onPrimary = Color(0xFF0A1230),
    secondary = Color(0xFF7CE0D3),
    background = Color(0xFF0B0B0F),
    surface = Color(0xFF141419),
    surfaceVariant = Color(0xFF1E1E26),
    onSurface = Color(0xFFEDEDF2),
    onSurfaceVariant = Color(0xFFB4B4C0)
)

private val LightColors = lightColorScheme(
    primary = Color(0xFF3D55C8),
    secondary = Color(0xFF2A8A7E)
)

@Composable
fun PixelUpTheme(content: @Composable () -> Unit) {
    // The UI is a dark photo workspace by design; light theme follows the
    // system but stays secondary.
    val colors = if (isSystemInDarkTheme()) DarkColors else LightColors
    MaterialTheme(colorScheme = colors, content = content)
}
