#!/bin/bash
# 使用新的Skeleton-Aware网络进行运动重定向
# 使用方法: bash retarget_with_new_network.sh [ref_motion_path] [output_path]

set -e

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tokenhsi

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
REF_MOTION="${1:-motions/ACCAD/ACCAD+__+Female1Walking_c3d+__+B3_-_walk1_stageii/phys_humanoid_v3/ref_motion.npy}"
OUTPUT="${2:-recovered_smpl24_params.npy}"
DEVICE="${3:-cpu}"  # 或 cuda

# 检查ref_motion文件是否存在
if [ ! -f "$REF_MOTION" ]; then
    echo "错误: 找不到ref_motion文件: $REF_MOTION"
    echo "请提供正确的路径，例如:"
    echo "  bash retarget_with_new_network.sh motions/ACCAD/.../phys_humanoid_v3/ref_motion.npy"
    exit 1
fi

echo "=========================================="
echo "使用新的Skeleton-Aware网络进行运动重定向"
echo "=========================================="
echo "输入文件: $REF_MOTION"
echo "输出文件: $OUTPUT"
echo "设备: $DEVICE"
echo "=========================================="
echo ""

# 运行重定向脚本
python retarget_with_new_network.py \
    --input "$REF_MOTION" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --num_iters 200 \
    --lr 1e-2

echo ""
echo "=========================================="
echo "完成！输出文件: $OUTPUT"
echo "=========================================="

