param(
    [Parameter(Mandatory = $true)]
    [string]$NotebookPath
)

$ErrorActionPreference = 'Stop'

function Set-CellSourceLines([object]$cell, [string[]]$lines) {
    $out = @()
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($i -lt $lines.Count - 1) { $out += ($lines[$i] + "`n") } else { $out += $lines[$i] }
    }
    $cell.source = $out
}

$nb = Get-Content -Raw -Encoding UTF8 $NotebookPath | ConvertFrom-Json

foreach ($c in $nb.cells) {
    if ($c.cell_type -ne 'markdown') { continue }

    $text = ($c.source -join '')

    if ($text.StartsWith('# TPLCNet 教程（v3）')) {
        Set-CellSourceLines $c @(
            '# TPLCNet 教程',
            '',
            '**环境准备说明：** 本笔记本讲解 TPLCNet 的核心模块，并给出真实温室数据上的可视化示例。',
            '- 只翻译/整理文字说明，代码逻辑保持不变',
            '- 如需运行示例，请先按第 1 节准备依赖与数据路径'
        )
        continue
    }

    if ($text.StartsWith('## 0. 环境与依赖')) {
        Set-CellSourceLines $c @(
            '### 1. 环境与依赖',
            '',
            '说明：',
            '- 建议 Python 3.9+',
            '- 建议使用你现有的 `TPLC_Net` conda 环境',
            '- 下面的安装命令仅在缺依赖时使用'
        )
        continue
    }

    if ($text.StartsWith('## 1. 包导入')) {
        Set-CellSourceLines $c @(
            '### 2. 包导入',
            '',
            '我们将导入：',
            '- 模型：`TPLCNet`',
            '- 模块层：`MultiScaleGenerator`、`DepthwiseSeparableConv2d`、`extract_topk_periods`、`reshape_1d_to_2d/reshape_2d_to_1d`',
            '- 训练器：`Trainer`',
            '- 数据管线：`prepare_greenhouse_datasets/make_loaders`'
        )
        continue
    }

    if ($text.StartsWith('## 2. TPLCNet 总体结构（先建立直觉）')) {
        Set-CellSourceLines $c @(
            '### 3. TPLCNet 总体结构（先建立直觉）',
            '',
            '#### 3.1 输入/输出张量约定',
            '- 输入 $x$：形状 `[B, T, C_{in}]`',
            '- 输出 $\\hat{y}$：形状 `[B, pred_len, C_{out}]`',
            '',
            '#### 3.2 一句话概括',
            'TPLCNet = **多尺度下采样** +（每个尺度上）**FFT 识别主周期** → 1D→2D → **2D 深度可分离卷积** → **多周期加权融合** → **多尺度预测融合**。'
        )
        continue
    }

    if ($text.StartsWith('## 3. 模块 A：MultiScaleGenerator')) {
        if ($c.source.Count -ge 2) {
            $second = ($c.source[1] -replace "`r?`n$", '')
            Set-CellSourceLines $c @('### 4. 模块 A：MultiScaleGenerator（多尺度生成器）', $second)
        } else {
            Set-CellSourceLines $c @('### 4. 模块 A：MultiScaleGenerator（多尺度生成器）')
        }
        continue
    }

    if ($text.StartsWith('### 3.1 平均池化下采样示意图（真实温室数据）')) {
        Set-CellSourceLines $c @(
            '#### 4.1 平均池化下采样示意图（真实温室数据）',
            '下面用真实温室序列，展示 `avg_pool1d(kernel=2, stride=2)` 如何把长度从 `T` 变成 `T/2`、`T/4`…'
        )
        continue
    }

    if ($text.StartsWith('## 4. 模块 B：extract_topk_periods')) {
        if ($c.source.Count -ge 2) {
            $second = ($c.source[1] -replace "`r?`n$", '')
            Set-CellSourceLines $c @('### 5. 模块 B：extract_topk_periods（FFT 周期识别）', $second)
        } else {
            Set-CellSourceLines $c @('### 5. 模块 B：extract_topk_periods（FFT 周期识别）')
        }
        continue
    }

    if ($text.StartsWith('## 5. 模块 C：reshape_1d_to_2d')) {
        if ($c.source.Count -ge 2) {
            $second = ($c.source[1] -replace "`r?`n$", '')
            Set-CellSourceLines $c @('### 6. 模块 C：reshape_1d_to_2d / reshape_2d_to_1d（1D↔2D 重塑）', $second)
        } else {
            Set-CellSourceLines $c @('### 6. 模块 C：reshape_1d_to_2d / reshape_2d_to_1d（1D↔2D 重塑）')
        }
        continue
    }

    if ($text.StartsWith('## 6. 模块 D：DepthwiseSeparableConv2d')) {
        if ($c.source.Count -ge 2) {
            $second = ($c.source[1] -replace "`r?`n$", '')
            Set-CellSourceLines $c @('### 7. 模块 D：DepthwiseSeparableConv2d（2D 深度可分离卷积）', $second)
        } else {
            Set-CellSourceLines $c @('### 7. 模块 D：DepthwiseSeparableConv2d（2D 深度可分离卷积）')
        }
        continue
    }

    if ($text.StartsWith('## 7. 模块 E：TPLCNet（总模型）')) {
        if ($c.source.Count -ge 2) {
            $second = ($c.source[1] -replace "`r?`n$", '')
            Set-CellSourceLines $c @('### 8. 模块 E：TPLCNet（总模型）', $second)
        } else {
            Set-CellSourceLines $c @('### 8. 模块 E：TPLCNet（总模型）')
        }
        continue
    }

    if ($text.StartsWith('## 8. 真实温室数据：加载与可视化')) {
        Set-CellSourceLines $c @('### 9. 真实温室数据：加载与可视化')
        continue
    }

    if ($text.StartsWith('### 8.0 原始序列展示（反标准化，更直观看周期性）')) {
        Set-CellSourceLines $c @(
            '#### 9.1 原始序列展示（反标准化，更直观看周期性）',
            '先把真实温室序列画出来，并用 FFT 找到的 top-1 周期 p 做辅助标注。'
        )
        continue
    }

    if ($text.Trim() -eq '### 8.1 FFT 频谱图 + top-K 周期标注') {
        Set-CellSourceLines $c @('#### 9.2 FFT 频谱图 + top-K 周期标注')
        continue
    }

    if ($text.Trim() -eq '### 8.2 1D→2D 热力图（单通道 + 通道均值）') {
        Set-CellSourceLines $c @('#### 9.3 1D→2D 热力图（单通道 + 通道均值）')
        continue
    }

    if ($text.Trim() -eq '### 8.3 多尺度输出曲线对比（各尺度 vs 融合 vs 真实）') {
        Set-CellSourceLines $c @('#### 9.4 多尺度输出曲线对比（各尺度 vs 融合 vs 真实）')
        continue
    }
}

$json = $nb | ConvertTo-Json -Depth 100
Set-Content -Path $NotebookPath -Value $json -Encoding UTF8

$remaining = (Select-String -Path $NotebookPath -Pattern '"## ' -AllMatches).Count
Write-Output "remaining \"##\" matches: $remaining"

