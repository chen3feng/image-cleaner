package com.imagecleaner.app.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.Compare
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Palette
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.Undo
import androidx.compose.material.icons.filled.Upgrade
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.imagecleaner.app.core.NativeEnhancer.MODE_AUTO
import com.imagecleaner.app.core.NativeEnhancer.MODE_COLORIZE
import com.imagecleaner.app.core.NativeEnhancer.MODE_FACES
import com.imagecleaner.app.core.NativeEnhancer.MODE_INPAINT
import com.imagecleaner.app.core.NativeEnhancer.MODE_UPSCALE
import com.imagecleaner.app.data.ModelManager

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(viewModel: MainViewModel = viewModel()) {
    val state by viewModel.ui.collectAsStateWithLifecycle()
    val snackbarHost = remember { SnackbarHostState() }
    var askToSave by remember { mutableStateOf(false) }

    val pickMedia = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri -> if (uri != null) viewModel.pickImage(uri) }

    val launchPicker = {
        pickMedia.launch(
            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
        )
        Unit
    }

    // Changing photos throws the edit history away, so ask first if the
    // current result has not been saved yet.
    val pick = {
        if (state.dirty) askToSave = true else launchPicker()
    }

    LaunchedEffect(state.notice) {
        state.notice?.let {
            snackbarHost.showSnackbar(it)
            viewModel.consumeNotice()
        }
    }

    if (askToSave) {
        AlertDialog(
            onDismissRequest = { askToSave = false },
            title = { Text("有未保存的修改") },
            text = { Text("当前结果还没有保存到相册，换图前要保存吗？") },
            confirmButton = {
                TextButton(onClick = {
                    askToSave = false
                    viewModel.saveResult { launchPicker() }
                }) { Text("保存并换图") }
            },
            dismissButton = {
                Row {
                    TextButton(onClick = { askToSave = false }) { Text("取消") }
                    TextButton(onClick = {
                        askToSave = false
                        launchPicker()
                    }) { Text("不保存") }
                }
            }
        )
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbarHost) },
        topBar = {
            TopAppBar(
                title = { Text("PixelUp") },
                actions = {
                    IconButton(
                        onClick = { viewModel.toggleCompare() },
                        enabled = state.canUndo && !state.running
                    ) {
                        Icon(
                            Icons.Filled.Compare,
                            contentDescription = if (state.comparing) "查看结果" else "对比"
                        )
                    }
                    IconButton(
                        onClick = { viewModel.undoResult() },
                        enabled = state.canUndo && !state.running
                    ) {
                        Icon(Icons.Filled.Undo, contentDescription = "撤销")
                    }
                    IconButton(onClick = pick) {
                        Icon(Icons.Filled.Image, contentDescription = "选择图片")
                    }
                    IconButton(
                        onClick = { viewModel.saveResult() },
                        enabled = state.image != null && !state.running
                    ) {
                        Icon(Icons.Filled.Save, contentDescription = "保存到相册")
                    }
                    IconButton(
                        onClick = { viewModel.shareResult() },
                        enabled = state.image != null && !state.running
                    ) {
                        Icon(Icons.Filled.Share, contentDescription = "分享")
                    }
                }
            )
        }
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            if (state.modelState !is ModelManager.State.Ready) {
                ModelSetupCard(state.modelState, onRetry = viewModel::retryModels)
            }

            Box(Modifier.weight(1f).fillMaxWidth()) {
                val image = state.image
                val shown = if (state.comparing) state.compareImage ?: image else image
                when {
                    image == null -> EmptyState(pick)
                    // Erase mode paints directly on the working image; the
                    // mask is tinted on top.
                    state.mode == MODE_INPAINT && !state.comparing -> PaintView(
                        image = image.asImageBitmap(),
                        mask = viewModel.maskBitmap()?.asImageBitmap(),
                        maskVersion = state.maskVersion,
                        brushRadiusImage = state.brush * image.width,
                        onStrokeStart = viewModel::beginStroke,
                        onPaint = viewModel::paintMask,
                        onStrokeEnd = viewModel::endStroke
                    )
                    shown != null -> ResultView(shown.asImageBitmap())
                }
                if (state.comparing) {
                    Surface(
                        modifier = Modifier.align(Alignment.TopCenter).padding(8.dp),
                        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f),
                        shape = MaterialTheme.shapes.small
                    ) {
                        Text(
                            "对比：编辑前",
                            style = MaterialTheme.typography.labelSmall,
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                        )
                    }
                }
                if (state.running) {
                    // The erase pipeline reads the mask on a worker thread;
                    // swallow touches so it cannot be edited concurrently.
                    Box(
                        Modifier.matchParentSize().pointerInput(Unit) {
                            awaitPointerEventScope {
                                while (true) {
                                    awaitPointerEvent().changes.forEach { it.consume() }
                                }
                            }
                        }
                    )
                }
                if (state.running) {
                    RunningOverlay(
                        state.progress,
                        Modifier.align(Alignment.BottomCenter)
                    )
                }
            }

            ControlPanel(state, viewModel, pick)
        }
    }
}

@Composable
private fun ResultView(image: ImageBitmap) {
    Image(
        bitmap = image,
        contentDescription = "处理结果",
        contentScale = ContentScale.Fit,
        modifier = Modifier.fillMaxSize()
    )
}

@Composable
private fun EmptyState(onPick: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            Icons.Filled.AutoAwesome,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(56.dp)
        )
        Spacer(Modifier.height(16.dp))
        Text("本地 AI 图像增强", style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.height(8.dp))
        Text(
            "人脸修复 · 超分辨率，全部在手机端运行，图片不会上传",
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(Modifier.height(24.dp))
        Button(onClick = onPick) { Text("选择图片") }
    }
}

@Composable
private fun RunningOverlay(progress: MainViewModel.Progress, modifier: Modifier = Modifier) {
    Card(
        modifier.fillMaxWidth().padding(16.dp),
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(progress.message.ifEmpty { "处理中" },
                style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(8.dp))
            LinearProgressIndicator(
                progress = { progress.percent.coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth()
            )
        }
    }
}

@Composable
private fun ModelSetupCard(state: ModelManager.State, onRetry: () -> Unit) {
    Card(Modifier.fillMaxWidth().padding(16.dp)) {
        Column(Modifier.padding(20.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            when (state) {
                is ModelManager.State.Error -> {
                    Text("模型下载失败：${state.message}",
                        color = MaterialTheme.colorScheme.error,
                        textAlign = TextAlign.Center)
                    Spacer(Modifier.height(12.dp))
                    OutlinedButton(onClick = onRetry) {
                        Icon(Icons.Filled.Refresh, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("重试")
                    }
                }
                is ModelManager.State.Downloading -> {
                    Text("首次使用需要下载本地模型 (${state.index + 1}/${state.total})")
                    Spacer(Modifier.height(4.dp))
                    Text(state.name, style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Spacer(Modifier.height(12.dp))
                    LinearProgressIndicator(
                        progress = { state.percent.coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth()
                    )
                }
                else -> {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp)
                        Spacer(Modifier.width(12.dp))
                        Text("正在检查本地模型…")
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ControlPanel(
    state: MainViewModel.UiState,
    viewModel: MainViewModel,
    onPick: () -> Unit
) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text("模式", style = MaterialTheme.typography.labelLarge)
            Spacer(Modifier.height(8.dp))
            ModeSelector(state.mode, enabled = !state.running, viewModel::setMode)

            Spacer(Modifier.height(16.dp))

            if (state.mode == MODE_INPAINT) {
                Text(
                    if (state.hasMask) "已选区域可直接消除，也可以继续涂抹后一起消除"
                    else "在图上涂抹要去掉的东西，再点「消除选区」",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(Modifier.height(8.dp))
                Text("画笔大小", style = MaterialTheme.typography.labelLarge)
                Slider(
                    value = state.brush,
                    onValueChange = viewModel::setBrush,
                    enabled = !state.running,
                    valueRange = 0.01f..0.25f
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(
                        onClick = viewModel::undoStroke,
                        enabled = state.canUndoStroke && !state.running
                    ) { Text("撤销一笔") }
                    OutlinedButton(
                        onClick = viewModel::clearMask,
                        enabled = state.hasMask && !state.running
                    ) { Text("清除选区") }
                }
            } else if (state.mode == MODE_COLORIZE) {
                Text(
                    "适合黑白老照片：只补颜色，不改分辨率",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    "色彩强度：${(state.colorStrength * 100).toInt()}%",
                    style = MaterialTheme.typography.labelLarge
                )
                Slider(
                    value = state.colorStrength,
                    onValueChange = viewModel::setColorStrength,
                    enabled = !state.running,
                    valueRange = 0.6f..2f
                )
            } else {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("倍率", style = MaterialTheme.typography.labelLarge)
                    Spacer(Modifier.width(16.dp))
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        listOf(2, 4).forEach { scale ->
                            FilterChip(
                                selected = state.scale == scale,
                                enabled = state.mode != MODE_FACES && !state.running,
                                onClick = { viewModel.setScale(scale) },
                                label = { Text("${scale}x") }
                            )
                        }
                    }
                }

                Spacer(Modifier.height(12.dp))

                Text(
                    "人脸保真度：${(state.fidelity * 100).toInt()}%（越低修复越强）",
                    style = MaterialTheme.typography.labelLarge
                )
                Slider(
                    value = state.fidelity,
                    onValueChange = viewModel::setFidelity,
                    enabled = state.mode != MODE_UPSCALE && !state.running,
                    valueRange = 0f..1f
                )
            }

            Spacer(Modifier.height(8.dp))

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedButton(onClick = onPick, enabled = !state.running) {
                    Text("换一张")
                }
                Button(
                    onClick = viewModel::enhance,
                    enabled = state.image != null && !state.running &&
                        state.modelState is ModelManager.State.Ready &&
                        (state.mode != MODE_INPAINT || state.hasMask),
                    modifier = Modifier.weight(1f)
                ) {
                    if (state.running) {
                        CircularProgressIndicator(
                            Modifier.size(18.dp), strokeWidth = 2.dp,
                            color = MaterialTheme.colorScheme.onPrimary
                        )
                        Spacer(Modifier.width(8.dp))
                        Text("处理中…")
                    } else {
                        Icon(Icons.Filled.AutoAwesome, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text(if (state.mode == MODE_INPAINT) "消除选区" else "开始增强")
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ModeSelector(
    selected: Int,
    enabled: Boolean,
    onSelect: (Int) -> Unit
) {
    data class Mode(val id: Int, val label: String, val icon: ImageVector)
    val modes = listOf(
        Mode(MODE_AUTO, "智能增强", Icons.Filled.AutoAwesome),
        Mode(MODE_FACES, "人脸修复", Icons.Filled.Face),
        Mode(MODE_UPSCALE, "超分辨率", Icons.Filled.Upgrade),
        Mode(MODE_COLORIZE, "上色", Icons.Filled.Palette),
        Mode(MODE_INPAINT, "消除", Icons.Filled.AutoFixHigh)
    )
    // Five modes no longer fit in one segmented row on a phone; chips wrap.
    FlowRow(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        modes.forEach { mode ->
            FilterChip(
                selected = selected == mode.id,
                enabled = enabled,
                onClick = { onSelect(mode.id) },
                leadingIcon = {
                    Icon(mode.icon, contentDescription = null, Modifier.size(18.dp))
                },
                label = { Text(mode.label, style = MaterialTheme.typography.labelLarge) }
            )
        }
    }
}
