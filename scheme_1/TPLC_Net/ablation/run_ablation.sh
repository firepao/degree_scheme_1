#!/bin/bash
# TPLC 模型消融实验运行脚本
# 
# 使用方式:
#   Windows PowerShell: .\run_ablation.ps1
#   Linux/Mac: bash run_ablation.sh
#
# 单独运行某个配置:
#   python run_ablation.py --config baseline
#   python run_ablation.py --config +revin
#   python run_ablation.py --config full

set -e

echo "=============================================="
echo "TPLC 模型消融实验"
echo "=============================================="

cd "$(dirname "$0")"

# 检查 Python 环境
python --version

# 运行所有消融实验
echo ""
echo "开始运行所有消融配置..."
echo ""

python run_ablation.py --config all

echo ""
echo "=============================================="
echo "消融实验完成！"
echo "结果保存在 results/ 目录"
echo "=============================================="
