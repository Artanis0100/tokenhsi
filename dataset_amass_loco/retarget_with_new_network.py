#!/usr/bin/env python
"""
使用新的Skeleton-Aware网络进行运动重定向
从ref_motion.npy重定向到recovered_smpl24_params.npy
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

from lpanlib.poselib.skeleton.skeleton3d import SkeletonMotion
from skeleton_aware_network import create_default_predictor
from recover_smpl24_from_ref_motion import (
    get_body_model,
    joints_to_use,
)


def retarget_motion_with_new_network(
    ref_motion_path,
    output_path,
    device="cpu",
    num_iters=200,
    lr=1e-2,
    use_network_init=True,
):
    """
    使用新的Skeleton-Aware网络进行运动重定向
    
    Args:
        ref_motion_path: ref_motion.npy文件路径
        output_path: 输出文件路径
        device: 计算设备
        num_iters: 优化迭代次数
        lr: 学习率
        use_network_init: 是否使用网络初始化
    """
    print("=" * 60)
    print("使用新的Skeleton-Aware网络进行运动重定向")
    print("=" * 60)
    
    # 检查输入文件
    if not osp.exists(ref_motion_path):
        raise FileNotFoundError(f"找不到输入文件: {ref_motion_path}")
    
    print(f"\n输入文件: {ref_motion_path}")
    print(f"输出文件: {output_path}")
    print(f"使用设备: {device}")
    
    # 加载运动数据
    print("\n加载运动数据...")
    motion = SkeletonMotion.from_file(ref_motion_path)
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
                pretrained_path=None,  # 不使用预训练权重
                device=device,
                hidden_channels=128,
                kernel_size=15
            )
            network_model.eval()
            
            # 预测24关节位置
            with torch.no_grad():
                # 输入: (T, 15, 3) phys_humanoid_v3关节位置
                # 网络期望输入: (B, T, J, C) 或 (T, J, C)
                predicted_smpl_pos = network_model(target_pos)  # (T, 24, 3)
            
            print(f"  网络预测完成，形状: {predicted_smpl_pos.shape}")
            
            # 提取对应的15个关节用于验证
            predicted_15_joints = predicted_smpl_pos[:, smpl_joint_ids, :]  # (T, 15, 3)
            init_error = (predicted_15_joints - target_pos).pow(2).mean().item()
            print(f"  网络预测的15关节误差: {init_error:.6f}")
            
            # 如果误差太大，说明网络未训练或不适用
            if init_error < 1.0:  # 阈值可以根据需要调整
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
    
    # 步骤2: 获取SMPL T-pose的默认骨骼长度
    print("\n获取SMPL T-pose骨骼长度...")
    with torch.no_grad():
        # 使用betas=0获取SMPL默认T-pose
        smpl_tpose = bm(betas=torch.zeros(1, 10, device=device))
        smpl_tpose_joints = smpl_tpose.joints[0]  # (24, 3)
    
    # 定义SMPL骨骼连接关系（对应phys_humanoid_v3的15个关节）
    # SMPL关节索引: [0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]
    # 对应phys: [pelvis, torso, head, right_upper_arm, right_lower_arm, right_hand,
    #            left_upper_arm, left_lower_arm, left_hand, right_thigh, right_shin, right_foot,
    #            left_thigh, left_shin, left_foot]
    smpl_bone_connections = [
        (6, 0),   # torso -> pelvis (SMPL joint 6 -> 0)
        (12, 6),  # head -> torso (SMPL joint 12 -> 6)
        (17, 6),  # right_upper_arm -> torso (SMPL joint 17 -> 6)
        (19, 17), # right_lower_arm -> right_upper_arm (SMPL joint 19 -> 17)
        (21, 19), # right_hand -> right_lower_arm (SMPL joint 21 -> 19)
        (16, 6),  # left_upper_arm -> torso (SMPL joint 16 -> 6)
        (18, 16), # left_lower_arm -> left_upper_arm (SMPL joint 18 -> 16)
        (20, 18), # left_hand -> left_lower_arm (SMPL joint 20 -> 18)
        (2, 0),   # right_thigh -> pelvis (SMPL joint 2 -> 0)
        (5, 2),   # right_shin -> right_thigh (SMPL joint 5 -> 2)
        (8, 5),   # right_foot -> right_shin (SMPL joint 8 -> 5)
        (1, 0),   # left_thigh -> pelvis (SMPL joint 1 -> 0)
        (4, 1),   # left_shin -> left_thigh (SMPL joint 4 -> 1)
        (7, 4),   # left_foot -> left_shin (SMPL joint 7 -> 4)
    ]
    
    # 计算SMPL T-pose的骨骼长度
    smpl_tpose_bone_lengths = {}
    for child_smpl, parent_smpl in smpl_bone_connections:
        bone_vec = smpl_tpose_joints[child_smpl] - smpl_tpose_joints[parent_smpl]
        bone_length = torch.norm(bone_vec).item()
        smpl_tpose_bone_lengths[(child_smpl, parent_smpl)] = bone_length
    
    print(f"  计算了 {len(smpl_tpose_bone_lengths)} 根SMPL T-pose骨骼的长度")
    for (child, parent), length in list(smpl_tpose_bone_lengths.items())[:3]:
        print(f"    SMPL骨骼 ({child}->{parent}): {length:.4f}")
    
    # 步骤3: 计算目标骨骼长度（从phys_humanoid_v3）
    print("\n计算phys_humanoid_v3骨骼长度...")
    # phys_humanoid_v3的骨骼连接关系（基于关节顺序）
    # [pelvis(0), torso(1), head(2), right_upper_arm(3), right_lower_arm(4), right_hand(5),
    #  left_upper_arm(6), left_lower_arm(7), left_hand(8), right_thigh(9), right_shin(10), right_foot(11),
    #  left_thigh(12), left_shin(13), left_foot(14)]
    # 定义骨骼连接: (child_idx, parent_idx)
    phys_bone_connections = [
        (1, 0),   # torso -> pelvis
        (2, 1),   # head -> torso
        (3, 1),   # right_upper_arm -> torso
        (4, 3),   # right_lower_arm -> right_upper_arm
        (5, 4),   # right_hand -> right_lower_arm
        (6, 1),   # left_upper_arm -> torso
        (7, 6),   # left_lower_arm -> left_upper_arm
        (8, 7),   # left_hand -> left_lower_arm
        (9, 0),   # right_thigh -> pelvis
        (10, 9),  # right_shin -> right_thigh
        (11, 10), # right_foot -> right_shin
        (12, 0),  # left_thigh -> pelvis
        (13, 12), # left_shin -> left_thigh
        (14, 13), # left_foot -> left_shin
    ]
    
    # 计算phys_humanoid_v3的平均骨骼长度（使用多帧平均，更稳定）
    print("  使用多帧平均计算骨骼长度...")
    target_bone_lengths = {}
    num_frames_for_avg = min(10, T)  # 使用前10帧或全部帧的平均值
    
    for child_idx, parent_idx in phys_bone_connections:
        bone_lengths = []
        for t in range(num_frames_for_avg):
            bone_vec = target_pos[t, child_idx] - target_pos[t, parent_idx]
            bone_length = torch.norm(bone_vec).item()
            bone_lengths.append(bone_length)
        avg_length = np.mean(bone_lengths)
        target_bone_lengths[(child_idx, parent_idx)] = avg_length
    
    print(f"  计算了 {len(target_bone_lengths)} 根phys_humanoid_v3骨骼的平均长度")
    for (child, parent), length in list(target_bone_lengths.items())[:3]:
        print(f"    phys骨骼 ({child}->{parent}): {length:.4f}")
    
    # 步骤4: 创建phys到SMPL的骨骼映射并计算比例因子
    print("\n计算骨骼长度比例...")
    phys_to_smpl_bone_map = {}
    bone_scale_factors = {}
    
    for i, (child_phys, parent_phys) in enumerate(phys_bone_connections):
        # 找到对应的SMPL关节索引
        child_smpl = smpl_joint_ids[child_phys]
        parent_smpl = smpl_joint_ids[parent_phys]
        phys_to_smpl_bone_map[(child_phys, parent_phys)] = (child_smpl, parent_smpl)
        
        # 计算比例因子：target_length / smpl_tpose_length
        target_length = target_bone_lengths[(child_phys, parent_phys)]
        smpl_tpose_length = smpl_tpose_bone_lengths[(child_smpl, parent_smpl)]
        scale_factor = target_length / smpl_tpose_length if smpl_tpose_length > 1e-6 else 1.0
        bone_scale_factors[(child_phys, parent_phys)] = scale_factor
    
    # 打印比例因子信息
    print(f"  骨骼长度比例因子:")
    for (child_phys, parent_phys), scale in list(bone_scale_factors.items())[:5]:
        target_len = target_bone_lengths[(child_phys, parent_phys)]
        child_smpl, parent_smpl = phys_to_smpl_bone_map[(child_phys, parent_phys)]
        smpl_len = smpl_tpose_bone_lengths[(child_smpl, parent_smpl)]
        print(f"    phys({child_phys}->{parent_phys}): {target_len:.4f} / SMPL({child_smpl}->{parent_smpl}): {smpl_len:.4f} = {scale:.4f}")
    
    # 计算平均比例因子（用于整体缩放参考）
    avg_scale = np.mean(list(bone_scale_factors.values()))
    print(f"  平均比例因子: {avg_scale:.4f}")
    
    # 步骤5: 优化SMPL参数以匹配目标关节位置和骨骼长度比例
    print("\n开始优化SMPL参数（考虑骨骼长度比例）...")
    
    # 初始化betas参数（用于调整SMPL体型和骨骼长度）
    betas_init = torch.zeros(1, 10, device=device, requires_grad=True)
    
    # 初始化全局方向和平移
    transl_init = motion.root_translation.to(device) if hasattr(motion, 'root_translation') else torch.zeros(T, 3, device=device)
    
    # 初始化SMPL参数
    if use_network_prediction:
        print("  使用网络预测的位置进行初始化...")
        # 初始化姿态（使用零初始化，然后通过优化匹配）
        global_orient_init = torch.zeros(T, 3, device=device, requires_grad=True)
        body_pose_init = torch.zeros(T, 69, device=device, requires_grad=True)
        transl_init_param = transl_init.clone().detach().requires_grad_(True)
        
        # 快速预优化以接近网络预测的位置
        print("  预优化以匹配网络预测...")
        init_optimizer = torch.optim.Adam([global_orient_init, body_pose_init, transl_init_param, betas_init], lr=lr * 2)
        for init_iter in range(min(50, num_iters // 4)):
            init_optimizer.zero_grad()
            out_init = bm(global_orient=global_orient_init, body_pose=body_pose_init, 
                         transl=transl_init_param, betas=betas_init.expand(T, -1))
            pred_joints = out_init.joints  # (T, 24, 3)
            
            # 匹配所有24个关节（如果使用网络预测）
            loss_init = (pred_joints - predicted_smpl_pos).pow(2).mean()
            loss_init.backward()
            init_optimizer.step()
            
            if (init_iter + 1) % 10 == 0:
                print(f"    预优化迭代 {init_iter + 1}/{min(50, num_iters // 4)}, 损失: {loss_init.item():.6f}")
    else:
        # 标准初始化
        print("  使用标准初始化...")
        global_orient_init = torch.zeros(T, 3, device=device, requires_grad=True)
        body_pose_init = torch.zeros(T, 69, device=device, requires_grad=True)
        transl_init_param = transl_init.clone().detach().requires_grad_(True)
    
    # 主优化循环
    print(f"  主优化循环 ({num_iters} 次迭代)...")
    optimizer = torch.optim.Adam([global_orient_init, body_pose_init, transl_init_param, betas_init], lr=lr)
    
    for iter_idx in range(num_iters):
        optimizer.zero_grad()
        out = bm(global_orient=global_orient_init, body_pose=body_pose_init, 
                transl=transl_init_param, betas=betas_init.expand(T, -1))
        pred_joints = out.joints  # (T, 24, 3)
        
        # 损失1: 匹配15个共享关节位置
        pred_shared = pred_joints[:, smpl_joint_ids, :]  # (T, 15, 3)
        loss_position = (pred_shared - target_pos).pow(2).mean()
        
        # 损失2: 匹配骨骼长度比例（确保相对比例正确）
        loss_bone_ratio = torch.tensor(0.0, device=device)
        loss_bone_absolute = torch.tensor(0.0, device=device)
        
        for (child_phys, parent_phys), target_length in target_bone_lengths.items():
            child_smpl, parent_smpl = phys_to_smpl_bone_map[(child_phys, parent_phys)]
            # 计算SMPL当前骨骼长度（使用第一帧作为参考）
            smpl_bone_vec = pred_joints[0, child_smpl] - pred_joints[0, parent_smpl]
            smpl_bone_length = torch.norm(smpl_bone_vec)
            
            # 获取SMPL T-pose的骨骼长度
            smpl_tpose_length = smpl_tpose_bone_lengths[(child_smpl, parent_smpl)]
            
            # 计算当前SMPL骨骼相对于T-pose的比例
            smpl_ratio = smpl_bone_length / (smpl_tpose_length + 1e-6)
            
            # 目标比例：target_length / smpl_tpose_length
            target_ratio = bone_scale_factors[(child_phys, parent_phys)]
            
            # 损失1: 比例匹配（确保相对比例正确）
            ratio_diff = (smpl_ratio - target_ratio).pow(2)
            loss_bone_ratio += ratio_diff
            
            # 损失2: 绝对长度匹配（作为辅助约束）
            length_diff = (smpl_bone_length - target_length).pow(2)
            loss_bone_absolute += length_diff
        
        loss_bone_ratio = loss_bone_ratio / len(target_bone_lengths)
        loss_bone_absolute = loss_bone_absolute / len(target_bone_lengths)
        
        # 总损失：位置匹配 + 骨骼比例匹配（主要） + 绝对长度匹配（辅助）
        loss = loss_position + 1.0 * loss_bone_ratio + 0.3 * loss_bone_absolute
        
        # 正则化betas（防止过度变形）
        loss_reg_betas = 1e-3 * betas_init.pow(2).mean()
        loss = loss + loss_reg_betas
        
        loss.backward()
        optimizer.step()
        
        if (iter_idx + 1) % 50 == 0 or iter_idx == 0:
            print(f"    迭代 {iter_idx + 1}/{num_iters}, 位置损失: {loss_position.item():.6f}, "
                  f"骨骼比例损失: {loss_bone_ratio.item():.6f}, 骨骼绝对损失: {loss_bone_absolute.item():.6f}, "
                  f"总损失: {loss.item():.6f}")
    
    # 提取最终结果
    print("\n提取最终结果...")
    with torch.no_grad():
        final_global_orient = global_orient_init.cpu().numpy()
        final_body_pose = body_pose_init.cpu().numpy()
        final_transl = transl_init_param.cpu().numpy()
        final_betas = betas_init.cpu().numpy()  # (1, 10)
    
    print(f"  优化的betas值: {final_betas[0]}")
    print(f"  betas范围: [{final_betas.min():.4f}, {final_betas.max():.4f}]")
    
    # 组合为72维姿态向量 (3 global_orient + 69 body_pose)
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
        description="使用新的Skeleton-Aware网络进行运动重定向"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="motions/ACCAD/ACCAD+__+Female1Walking_c3d+__+B3_-_walk1_stageii/phys_humanoid_v3/ref_motion.npy",
        help="输入ref_motion.npy文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="recovered_smpl24_params.npy",
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
    retarget_motion_with_new_network(
        ref_motion_path=input_path,
        output_path=output_path,
        device=args.device,
        num_iters=args.num_iters,
        lr=args.lr,
        use_network_init=not args.no_network_init,
    )


if __name__ == "__main__":
    main()

