package com.imagecleaner.app

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.flow.MutableStateFlow
import com.imagecleaner.app.ui.MainScreen
import com.imagecleaner.app.ui.MainViewModel
import com.imagecleaner.app.ui.theme.PixelUpTheme

class MainActivity : ComponentActivity() {

    // Bumped whenever a new SEND intent arrives while we are already on top, so
    // the Compose effect re-runs without recreating the activity (a recreated
    // activity means a new ViewModel and a second native pipeline, which used
    // to load ~180 MB of models all over again).
    private val sharedUri = MutableStateFlow<Pair<Uri, Int>?>(null)
    private var shareCount = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        publishSharedImage(intent)
        setContent {
            PixelUpTheme {
                val viewModel: MainViewModel = viewModel()
                val shared by sharedUri.collectAsState()
                LaunchedEffect(shared) {
                    shared?.let { (uri, _) -> viewModel.pickImage(uri) }
                }
                MainScreen(viewModel)
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        publishSharedImage(intent)
    }

    private fun publishSharedImage(intent: Intent?) {
        intent?.sharedImageUri()?.let { uri ->
            sharedUri.value = uri to ++shareCount
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
