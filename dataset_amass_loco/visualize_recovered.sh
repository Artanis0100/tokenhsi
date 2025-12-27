#!/bin/bash
# 可视化recovered_smpl24_params.npy的脚本

set -e

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tokenhsi

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
PARAMS_FILE="${1:-recovered_smpl24_params.npy}"
OUTPUT="${2:-recovered_smpl24_params.mp4}"
MAX_FRAMES="${3:-}"  # 空字符串表示渲染所有帧
DEVICE="${4:-cpu}"

# 检查参数文件是否存在
if [ ! -f "$PARAMS_FILE" ]; then
    echo "错误: 找不到参数文件: $PARAMS_FILE"
    echo "请提供正确的路径，例如:"
    echo "  bash visualize_recovered.sh recovered_smpl24_params.npy"
    exit 1
fi

echo "=========================================="
echo "可视化SMPL参数文件"
echo "=========================================="
echo "输入文件: $PARAMS_FILE"
echo "输出文件: $OUTPUT"
echo "最大帧数: $MAX_FRAMES"
echo "设备: $DEVICE"
echo "=========================================="
echo ""

# 运行可视化脚本
if [ -z "$MAX_FRAMES" ]; then
    # 不指定max_frames，渲染所有帧
    python visualize_smpl24.py \
        --params "$PARAMS_FILE" \
        --output "$OUTPUT" \
        --device "$DEVICE"
else
    python visualize_smpl24.py \
        --params "$PARAMS_FILE" \
        --output "$OUTPUT" \
        --max_frames "$MAX_FRAMES" \
        --device "$DEVICE"
fi

echo ""
echo "=========================================="
echo "完成！视频文件: $OUTPUT"
echo "=========================================="

