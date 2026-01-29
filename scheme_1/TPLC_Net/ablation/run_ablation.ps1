# TPLC 模型消融实验运行脚本 (PowerShell)
#
# 使用方式:
#   .\run_ablation.ps1                    # 运行所有配置
#   .\run_ablation.ps1 -Config baseline   # 运行基线
#   .\run_ablation.ps1 -Config +revin     # 运行 +RevIN
#   .\run_ablation.ps1 -Config full       # 运行完整模型

param(
    [string]$Config = "all",
    [int]$Epochs = 0,
    [int]$Seed = 0
)

Write-Host "=============================================="
Write-Host "TPLC 模型消融实验"
Write-Host "=============================================="

# 切换到脚本所在目录
Set-Location $PSScriptRoot

# 检查 Python 环境
Write-Host ""
Write-Host "Python 环境:"
python --version

# 构建命令参数
$args_list = @("run_ablation.py", "--config", $Config)

if ($Epochs -gt 0) {
    $args_list += @("--epochs", $Epochs)
}

if ($Seed -gt 0) {
    $args_list += @("--seed", $Seed)
}

# 运行实验
Write-Host ""
Write-Host "运行配置: $Config"
Write-Host "命令: python $($args_list -join ' ')"
Write-Host ""

python @args_list

Write-Host ""
Write-Host "=============================================="
Write-Host "实验完成！"
Write-Host "结果保存在 results/ 目录"
Write-Host "=============================================="
