#!/usr/bin/env python
"""
测试脚本：使用Skeleton-Aware方法恢复SMPL数据
"""
import sys
import os
sys.path.append("./")
import os.path as osp
import numpy as np

# 导入恢复函数
from recover_smpl24_from_ref_motion import fit_smpl_to_motion_skeleton_aware
from lpanlib.poselib.skeleton.skeleton3d import SkeletonMotion

def test_skeleton_aware_method():
    """测试Skeleton-Aware方法"""
    # 使用一个示例ref_motion文件
    script_dir = osp.dirname(osp.abspath(__file__))
    ref_motion_path = osp.join(
        script_dir,
        "motions/ACCAD/ACCAD+__+Female1Walking_c3d+__+B3_-_walk1_stageii/phys_humanoid_v3/ref_motion.npy"
    )
    
    # 如果文件不存在，尝试找其他文件
    if not osp.exists(ref_motion_path):
        # 查找任意一个ref_motion.npy文件
        import glob
        ref_motions = glob.glob(osp.join(script_dir, "motions/**/phys_humanoid_v3/ref_motion.npy"), recursive=True)
        if ref_motions:
            ref_motion_path = ref_motions[0]
            print(f"使用找到的文件: {ref_motion_path}")
        else:
            print("错误: 找不到ref_motion.npy文件")
            return
    
    print(f"加载ref_motion: {ref_motion_path}")
    motion = SkeletonMotion.from_file(ref_motion_path)
    
    print(f"运动数据: {motion.global_translation.shape} 帧, {motion.global_translation.shape[1]} 关节")
    print(f"FPS: {motion.fps}")
    
    # 使用Skeleton-Aware方法恢复
    print("\n开始使用Skeleton-Aware方法恢复SMPL数据...")
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    print(f"使用设备: {device}")
    
    params = fit_smpl_to_motion_skeleton_aware(
        motion,
        num_iters=200,
        lr=1e-2,
        device=device,
        use_network_init=True,
        network_ckpt_path=None,  # 不使用预训练权重
    )
    
    # 验证输出格式
    print("\n验证输出格式:")
    print(f"  poses shape: {params['poses'].shape}")
    print(f"  trans shape: {params['trans'].shape}")
    print(f"  fps: {params['fps']}")
    
    # 检查输出结构
    assert "poses" in params, "缺少 'poses' 键"
    assert "trans" in params, "缺少 'trans' 键"
    assert "fps" in params, "缺少 'fps' 键"
    assert params["poses"].shape[1] == 72, f"poses应该是72维，实际是{params['poses'].shape[1]}"
    assert params["trans"].shape[1] == 3, f"trans应该是3维，实际是{params['trans'].shape[1]}"
    
    print("\n✓ 输出格式验证通过！")
    
    # 保存结果
    output_path = osp.join(script_dir, "recovered_smpl24_params_skeleton_aware.npy")
    np.save(output_path, params)
    print(f"\n保存结果到: {output_path}")
    
    # 加载并验证保存的文件
    loaded_params = np.load(output_path, allow_pickle=True).item()
    print(f"\n验证保存的文件:")
    print(f"  poses shape: {loaded_params['poses'].shape}")
    print(f"  trans shape: {loaded_params['trans'].shape}")
    print(f"  fps: {loaded_params['fps']}")
    
    print("\n✓ 测试完成！")

if __name__ == "__main__":
    test_skeleton_aware_method()

