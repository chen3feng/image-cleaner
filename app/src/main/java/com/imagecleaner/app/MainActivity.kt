package com.imagecleaner.app

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.LaunchedEffect
import androidx.lifecycle.viewmodel.compose.viewModel
import com.imagecleaner.app.ui.MainScreen
import com.imagecleaner.app.ui.MainViewModel
import com.imagecleaner.app.ui.theme.PixelUpTheme

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContent {
            PixelUpTheme {
                val viewModel: MainViewModel = viewModel()
                LaunchedEffect(intent) {
                    intent.sharedImageUri()?.let(viewModel::pickImage)
                }
                MainScreen(viewModel)
            }
        }
    }

    private fun Intent.sharedImageUri(): Uri? {
        if (action != Intent.ACTION_SEND || type?.startsWith("image/") != true) return null
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
        } else {
            @Suppress("DEPRECATION")
            getParcelableExtra(Intent.EXTRA_STREAM) as? Uri
        }
    }
}
