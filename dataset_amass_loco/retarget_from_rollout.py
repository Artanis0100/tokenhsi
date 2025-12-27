#!/usr/bin/env python
"""
从rollout_data.npy文件进行运动重定向
支持rollout_data格式（包含root_translation和global_rotation）
"""
import sys
import os
import os.path as osp
import numpy as np
import torch

# 添加路径
script_dir = osp.dirname(osp.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
sys.path.append(osp.dirname(osp.dirname(osp.dirname(script_dir))))

from lpanlib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree
from skeleton_aware_network import create_default_predictor
from recover_smpl24_from_ref_motion import (
    get_body_model,
    joints_to_use,
)


def load_rollout_data(rollout_path):
    """
    加载rollout_data.npy文件并转换为SkeletonMotion
    
    Args:
        rollout_path: rollout_data.npy文件路径
        
    Returns:
        SkeletonMotion对象
    """
    print(f"加载rollout数据: {rollout_path}")
    data = np.load(rollout_path, allow_pickle=True).item()
    
    root_translation = data['root_translation']  # (T, 3)
    global_rotation = data['global_rotation']  # (T, 15, 4) - quaternions in xyzw format
    fps = data.get('fps', 30.0)
    
    T = root_translation.shape[0]
    J = global_rotation.shape[1]
    
    print(f"  帧数: {T}")
    print(f"  关节数: {J}")
    print(f"  FPS: {fps}")
    
    # 转换为torch tensor
    root_trans = torch.from_numpy(root_translation).float()
    global_rot = torch.from_numpy(global_rotation).float()
    
    # 加载phys_humanoid_v3骨架
    script_dir = osp.dirname(osp.abspath(__file__))
    phys_humanoid_v3_xml_path = osp.join(
        script_dir, 
        "../assets/mjcf/phys_humanoid_v3.xml"
    )
    
    if not osp.exists(phys_humanoid_v3_xml_path):
        # 尝试从ref_motion.npy获取骨架
        ref_motion_path = osp.join(
            osp.dirname(osp.dirname(rollout_path)),
            "ref_motion.npy"
        )
        if osp.exists(ref_motion_path):
            print(f"  使用ref_motion.npy的骨架结构: {ref_motion_path}")
            ref_motion = SkeletonMotion.from_file(ref_motion_path)
            skeleton = ref_motion.skeleton
        else:
            raise FileNotFoundError(
                f"找不到phys_humanoid_v3.xml: {phys_humanoid_v3_xml_path}\n"
                f"也找不到ref_motion.npy: {ref_motion_path}"
            )
    else:
        print(f"  加载phys_humanoid_v3骨架: {phys_humanoid_v3_xml_path}")
        from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree
        skeleton = SkeletonTree.from_mjcf(phys_humanoid_v3_xml_path)
    
    # 创建SkeletonState
    # global_rotation是全局旋转，我们使用is_local=False
    skeleton_state = SkeletonState.from_rotation_and_root_translation(
        skeleton,
        global_rot,  # (T, J, 4) - 全局旋转
        root_trans,  # (T, 3) - 根节点平移
        is_local=False  # 明确指定这是全局旋转
    )
    
    # 转换为local表示（SkeletonMotion需要）
    skeleton_state = skeleton_state.local_repr()
    
    # 创建SkeletonMotion
    motion = SkeletonMotion.from_skeleton_state(skeleton_state, fps=fps)
    
    return motion


def retarget_from_rollout(
    rollout_path,
    output_path,
    device="cpu",
    num_iters=200,
    lr=1e-2,
    use_network_init=True,
):
    """
    从rollout_data.npy进行运动重定向
    
    Args:
        rollout_path: rollout_data.npy文件路径
        output_path: 输出文件路径
        device: 计算设备
        num_iters: 优化迭代次数
        lr: 学习率
        use_network_init: 是否使用网络初始化
    """
    print("=" * 60)
    print("从Rollout数据重定向到SMPL")
    print("=" * 60)
    
    # 检查输入文件
    if not osp.exists(rollout_path):
        raise FileNotFoundError(f"找不到输入文件: {rollout_path}")
    
    print(f"\n输入文件: {rollout_path}")
    print(f"输出文件: {output_path}")
    print(f"使用设备: {device}")
    
    # 加载rollout数据并转换为SkeletonMotion
    print("\n加载rollout数据...")
    motion = load_rollout_data(rollout_path)
    
    print(f"  帧数: {motion.global_translation.shape[0]}")
    print(f"  关节数: {motion.global_translation.shape[1]}")
    print(f"  FPS: {motion.fps}")
    
    # 获取15关节位置
    target_pos = motion.global_translation  # (T, 15, 3)
    T, J_phys, _ = target_pos.shape
    
    smpl_joint_ids = joints_to_use["from_smpl_original_to_phys_humanoid_v3"]
    assert len(smpl_joint_ids) == J_phys, f"关节数量不匹配: {len(smpl_joint_ids)} vs {J_phys}"
    
    target_pos = target_pos.to(device) if isinstance(target_pos, torch.Tensor) else torch.from_numpy(target_pos).to(device)
    
    # 构建SMPL模型
    print("\n构建SMPL模型...")
    bm = get_body_model("SMPL", "NEUTRAL", batch_size=T, debug=False)
    bm = bm.to(device)
    
    # 步骤1: 使用新的skeleton-aware网络预测24个SMPL关节位置
    predicted_smpl_pos = None
    use_network_prediction = False
    
    if use_network_init:
        print("\n使用新的Skeleton-Aware网络预测24关节SMPL位置...")
        try:
            # 创建新的网络模型
            network_model = create_default_predictor(
                pretrained_path=None,
                device=device,
                hidden_channels=128,
                kernel_size=15
            )
            network_model.eval()
            
            # 预测24关节位置
            with torch.no_grad():
                predicted_smpl_pos = network_model(target_pos)  # (T, 24, 3)
            
            print(f"  网络预测完成，形状: {predicted_smpl_pos.shape}")
            
            # 提取对应的15个关节用于验证
            predicted_15_joints = predicted_smpl_pos[:, smpl_joint_ids, :]  # (T, 15, 3)
            init_error = (predicted_15_joints - target_pos).pow(2).mean().item()
            print(f"  网络预测的15关节误差: {init_error:.6f}")
            
            if init_error < 1.0:
                use_network_prediction = True
                print("  ✓ 网络预测质量良好，将用于初始化优化")
            else:
                print(f"  ⚠ 网络预测误差较大 ({init_error:.6f})，将仅使用优化方法")
                use_network_prediction = False
        except Exception as e:
            print(f"  ⚠ 网络预测失败，回退到标准优化方法: {e}")
            import traceback
            traceback.print_exc()
            use_network_prediction = False
    
    # 步骤2: 优化SMPL参数
    print("\n开始优化SMPL参数...")
    
    # 初始化SMPL参数
    if use_network_prediction:
        print("  使用网络预测的位置进行初始化...")
        transl_init = motion.root_translation.to(device) if hasattr(motion, 'root_translation') else torch.zeros(T, 3, device=device)
        
        global_orient_init = torch.zeros(T, 3, device=device, requires_grad=True)
        body_pose_init = torch.zeros(T, 69, device=device, requires_grad=True)
        transl_init_param = transl_init.clone().detach().requires_grad_(True)
        
        # 快速预优化
        print("  预优化以匹配网络预测...")
        init_optimizer = torch.optim.Adam([global_orient_init, body_pose_init, transl_init_param], lr=lr * 2)
        for init_iter in range(min(50, num_iters // 4)):
            init_optimizer.zero_grad()
            out_init = bm(global_orient=global_orient_init, body_pose=body_pose_init, transl=transl_init_param)
            pred_joints = out_init.joints  # (T, 24, 3)
            loss_init = (pred_joints - predicted_smpl_pos).pow(2).mean()
            loss_init.backward()
            init_optimizer.step()
            
            if (init_iter + 1) % 10 == 0:
                print(f"    预优化迭代 {init_iter + 1}/{min(50, num_iters // 4)}, 损失: {loss_init.item():.6f}")
    else:
        print("  使用标准初始化...")
        transl_init = motion.root_translation.to(device) if hasattr(motion, 'root_translation') else torch.zeros(T, 3, device=device)
        global_orient_init = torch.zeros(T, 3, device=device, requires_grad=True)
        body_pose_init = torch.zeros(T, 69, device=device, requires_grad=True)
        transl_init_param = transl_init.clone().detach().requires_grad_(True)
    
    # 主优化循环
    print(f"  主优化循环 ({num_iters} 次迭代)...")
    optimizer = torch.optim.Adam([global_orient_init, body_pose_init, transl_init_param], lr=lr)
    
    for iter_idx in range(num_iters):
        optimizer.zero_grad()
        out = bm(global_orient=global_orient_init, body_pose=body_pose_init, transl=transl_init_param)
        pred_joints = out.joints  # (T, 24, 3)
        
        # 匹配15个共享关节
        pred_shared = pred_joints[:, smpl_joint_ids, :]  # (T, 15, 3)
        loss = (pred_shared - target_pos).pow(2).mean()
        
        loss.backward()
        optimizer.step()
        
        if (iter_idx + 1) % 50 == 0 or iter_idx == 0:
            print(f"    迭代 {iter_idx + 1}/{num_iters}, 损失: {loss.item():.6f}")
    
    # 提取最终结果
    print("\n提取最终结果...")
    with torch.no_grad():
        final_global_orient = global_orient_init.cpu().numpy()
        final_body_pose = body_pose_init.cpu().numpy()
        final_transl = transl_init_param.cpu().numpy()
    
    # 组合为72维姿态向量
    poses = np.concatenate([final_global_orient, final_body_pose], axis=1)  # (T, 72)
    
    # 构建输出字典
    out_dict = {
        "poses": poses.astype(np.float32),
        "trans": final_transl.astype(np.float32),
        "fps": float(motion.fps),
    }
    
    # 保存结果
    print(f"\n保存结果到: {output_path}")
    os.makedirs(osp.dirname(output_path) if osp.dirname(output_path) else '.', exist_ok=True)
    np.save(output_path, out_dict)
    
    print("\n" + "=" * 60)
    print("重定向完成！")
    print("=" * 60)
    print(f"\n输出信息:")
    print(f"  poses shape: {out_dict['poses'].shape}")
    print(f"  trans shape: {out_dict['trans'].shape}")
    print(f"  fps: {out_dict['fps']}")
    
    return out_dict


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="从rollout_data.npy进行运动重定向"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入rollout_data.npy文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="recovered_smpl24_params_from_rollout.npy",
        help="输出文件路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备 (cpu/cuda)"
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=200,
        help="优化迭代次数"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="学习率"
    )
    parser.add_argument(
        "--no_network_init",
        action="store_true",
        help="不使用网络初始化（仅使用优化）"
    )
    
    args = parser.parse_args()
    
    # 获取脚本目录
    script_dir = osp.dirname(osp.abspath(__file__))
    
    # 处理路径
    if not osp.isabs(args.input):
        input_path = osp.join(script_dir, args.input)
    else:
        input_path = args.input
    
    if not osp.isabs(args.output):
        output_path = osp.join(script_dir, args.output)
    else:
        output_path = args.output
    
    # 执行重定向
    retarget_from_rollout(
        rollout_path=input_path,
        output_path=output_path,
        device=args.device,
        num_iters=args.num_iters,
        lr=args.lr,
        use_network_init=not args.no_network_init,
    )


if __name__ == "__main__":
    main()

