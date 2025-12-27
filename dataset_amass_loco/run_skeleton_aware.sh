#!/bin/bash
# 使用Skeleton-Aware方法恢复SMPL数据的脚本
# 使用方法: bash run_skeleton_aware.sh [ref_motion_path] [output_path]

set -e

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tokenhsi

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
REF_MOTION="${1:-motions/ACCAD/ACCAD+__+Female1Walking_c3d+__+B3_-_walk1_stageii/phys_humanoid_v3/ref_motion.npy}"
OUTPUT="${2:-recovered_smpl24_params_skeleton_aware.npy}"
DEVICE="${3:-cpu}"  # 或 cuda

# 检查ref_motion文件是否存在
if [ ! -f "$REF_MOTION" ]; then
    echo "错误: 找不到ref_motion文件: $REF_MOTION"
    echo "请提供正确的路径，例如:"
    echo "  bash run_skeleton_aware.sh motions/ACCAD/.../phys_humanoid_v3/ref_motion.npy"
    exit 1
fi

echo "=========================================="
echo "使用Skeleton-Aware方法恢复SMPL数据"
echo "=========================================="
echo "输入文件: $REF_MOTION"
echo "输出文件: $OUTPUT"
echo "设备: $DEVICE"
echo "=========================================="
echo ""

# 运行恢复脚本
python recover_smpl24_from_ref_motion.py \
    --ref_motion "$REF_MOTION" \
    --output "$OUTPUT" \
    --method skeleton_aware \
    --iters 200 \
    --lr 1e-2 \
    --device "$DEVICE"

echo ""
echo "=========================================="
echo "完成！输出文件: $OUTPUT"
echo "=========================================="

