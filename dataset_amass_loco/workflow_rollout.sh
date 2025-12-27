#!/bin/bash
# 完整工作流：从rollout_data.npy重定向到SMPL并可视化

set -e

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tokenhsi

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
ROLLOUT_FILE="${1:-motions/ACCAD/ACCAD+__+Female1Walking_c3d+__+B3_-_walk1_stageii/phys_humanoid_v3/rollout_data/rollout_repeat_0_env_1.npy}"
OUTPUT_PARAMS="${2:-recovered_smpl24_params_from_rollout.npy}"
OUTPUT_VIDEO="${3:-recovered_smpl24_params_from_rollout.mp4}"
DEVICE="${4:-cpu}"

# 检查rollout文件是否存在
if [ ! -f "$ROLLOUT_FILE" ]; then
    echo "错误: 找不到rollout文件: $ROLLOUT_FILE"
    echo "请提供正确的路径"
    exit 1
fi

echo "=========================================="
echo "完整工作流：Rollout数据重定向和可视化"
echo "=========================================="
echo "输入文件: $ROLLOUT_FILE"
echo "输出参数: $OUTPUT_PARAMS"
echo "输出视频: $OUTPUT_VIDEO"
echo "设备: $DEVICE"
echo "=========================================="
echo ""

# 步骤1: 重定向
echo "步骤1: 重定向rollout数据到SMPL..."
python retarget_from_rollout.py \
    --input "$ROLLOUT_FILE" \
    --output "$OUTPUT_PARAMS" \
    --device "$DEVICE" \
    --num_iters 200 \
    --lr 1e-2

if [ $? -ne 0 ]; then
    echo "错误: 重定向失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "步骤2: 可视化SMPL参数..."
echo "=========================================="

# 步骤2: 可视化
python visualize_smpl24.py \
    --params "$OUTPUT_PARAMS" \
    --output "$OUTPUT_VIDEO" \
    --device "$DEVICE"

if [ $? -ne 0 ]; then
    echo "错误: 可视化失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="
echo "输出参数文件: $OUTPUT_PARAMS"
echo "输出视频文件: $OUTPUT_VIDEO"
echo "=========================================="

