import sys
sys.path.append("./")
import os
import os.path as osp
import argparse
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import numpy as np
import torch
import math
from scipy.spatial.transform import Rotation as R

# Suppress FBX import warnings (we don't use FBX functionality)
# The skeleton3d module imports FBX at top level, causing warnings
_fbx_warning_buffer = StringIO()
with redirect_stderr(_fbx_warning_buffer), redirect_stdout(_fbx_warning_buffer):
    from lpanlib.poselib.skeleton.skeleton3d import SkeletonMotion
from body_models.model_loader import get_body_model

# 导入skeleton-aware网络模块
try:
    from skeleton_aware_network import SMPLJointPredictor, create_default_predictor
    SKELETON_AWARE_AVAILABLE = True
except ImportError:
    # 如果导入失败（例如相对导入问题），尝试绝对导入
    import sys
    script_dir = osp.dirname(osp.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    try:
        from skeleton_aware_network import SMPLJointPredictor, create_default_predictor
        SKELETON_AWARE_AVAILABLE = True
    except ImportError:
        SKELETON_AWARE_AVAILABLE = False
        print("警告: 无法导入skeleton_aware_network模块，skeleton_aware方法将不可用")


# Same mapping as in generate_motion.py
# This maps phys_humanoid_v3 joint indices to SMPL joint indices
# phys_humanoid_v3 order: [pelvis, torso, head, right_upper_arm, right_lower_arm, right_hand,
#                           left_upper_arm, left_lower_arm, left_hand, right_thigh, right_shin, right_foot,
#                           left_thigh, left_shin, left_foot]
# IMPORTANT: This must match generate_motion.py exactly!
# Mapping: [0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]
#   pelvis->0, torso->6, head->12(Neck), 
#   right_upper_arm->17, right_lower_arm->19, right_hand->21,
#   left_upper_arm->16, left_lower_arm->18, left_hand->20,
#   right_thigh->2, right_shin->5, right_foot->8,
#   left_thigh->1, left_shin->4, left_foot->7
joints_to_use = {
    "from_smpl_original_to_phys_humanoid_v3": np.array(
        [0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]  # Must match generate_motion.py!
    ),
}

# Alternative mapping for direct pose mapping (if poses are available instead of joint positions)
# This would map from AMP/phys_humanoid_v3 pose indices to SMPL pose indices
# Note: This is different from the joint position mapping above
# If you have 15×3 axis-angle poses, you can use this to directly map to 24×3 SMPL poses
amp_to_smpl_pose_mapping = {
    "amp_pose_indices": [0, 1, 4, 7, 10, 2, 5, 8, 11, 16, 18, 20, 17, 19, 21],  # Example mapping
    "smpl_pose_indices": [0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7],  # Corresponding SMPL joints
}


def fit_smpl_to_motion(
    motion: SkeletonMotion,
    num_iters: int = 200,
    lr: float = 1e-2,
    device: str = "cpu",
):
    """
    Fit a 24-joint SMPL trajectory approximately to a given SkeletonMotion
    (e.g. loaded from ref_motion.npy), using only the 15 joints that are
    shared between SMPL and phys_humanoid_v3.

    APPROACH COMPARISON:
    - This function uses OPTIMIZATION-BASED IK: takes joint positions as input,
      optimizes SMPL parameters (global_orient, body_pose, transl) to match them.
      More accurate but slower.
    
    - Alternative approach (fit_smpl_from_poses_direct): takes axis-angle poses as input,
      directly maps 15 poses to 24 SMPL poses with zero-fill, then optionally refines.
      Faster but requires poses as input (not available in ref_motion.npy which only
      has joint positions).

    Returns a dict with:
        - poses: (T, 72) angle-axis (global_orient + body_pose)
        - trans: (T, 3) root translations
        - fps: scalar

    This is an approximate inverse and will not reproduce the original
    AMASS params exactly.
    """
    motion = motion.to(device=device) if hasattr(motion, "to") else motion

    # Observed joint positions from the ref motion
    # shape: (T, J_phys, 3)
    # The ref_motion has joints in phys_humanoid_v3 order:
    # [pelvis, torso, head, right_upper_arm, right_lower_arm, right_hand,
    #  left_upper_arm, left_lower_arm, left_hand, right_thigh, right_shin, right_foot,
    #  left_thigh, left_shin, left_foot]
    target_pos = motion.global_translation  # phys_humanoid_v3 joints in order [0, 1, 2, ..., 14]
    T, J_phys, _ = target_pos.shape

    # Mapping: which SMPL joints correspond to each phys_humanoid_v3 joint
    # joints_to_use[i] gives the SMPL joint index that corresponds to phys_humanoid_v3 joint i
    smpl_joint_ids = joints_to_use["from_smpl_original_to_phys_humanoid_v3"]
    assert (
        len(smpl_joint_ids) == J_phys
    ), f"Expected phys_humanoid_v3 to have same joint count as mapping length: {len(smpl_joint_ids)} vs {J_phys}"

    target_pos = target_pos.to(device)

    # Build SMPL model
    bm = get_body_model("SMPL", "NEUTRAL", batch_size=T, debug=False)
    bm = bm.to(device)

    # Initialize parameters
    # Better initialization: estimate initial pose from joint positions
    # First, get a rough estimate by running a few iterations with higher learning rate
    global_orient_init = torch.zeros(T, 3, device=device, requires_grad=True)
    body_pose_init = torch.zeros(T, 69, device=device, requires_grad=True)
    transl_init = motion.root_translation.to(device)
    transl_init_param = transl_init.clone().detach().requires_grad_(True)
    
    # Quick initialization pass (few iterations to get rough estimate)
    init_optimizer = torch.optim.Adam([global_orient_init, body_pose_init, transl_init_param], lr=lr * 5)
    for init_iter in range(min(20, num_iters // 10)):  # Use 10% of iterations for init
        init_optimizer.zero_grad()
        out_init = bm(global_orient=global_orient_init, body_pose=body_pose_init, transl=transl_init_param)
        smpl_joints_init = out_init.joints[:, smpl_joint_ids, :]
        loss_init = (smpl_joints_init - target_pos).pow(2).mean()
        loss_init.backward()
        init_optimizer.step()
    
    # Use initialized values as starting point
    global_orient = global_orient_init.detach().clone().requires_grad_(True)
    body_pose = body_pose_init.detach().clone().requires_grad_(True)
    transl = transl_init_param.detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([global_orient, body_pose, transl], lr=lr)

    for iter_idx in range(num_iters):
        optimizer.zero_grad()

        out = bm(global_orient=global_orient, body_pose=body_pose, transl=transl)
        # Extract SMPL joints in the order that corresponds to phys_humanoid_v3 joints
        # IMPORTANT: The ref_motion stores joints in phys_humanoid_v3 skeleton tree order:
        # [pelvis, torso, head, right_upper_arm, right_lower_arm, right_hand,
        #  left_upper_arm, left_lower_arm, left_hand, right_thigh, right_shin, right_foot,
        #  left_thigh, left_shin, left_foot]
        # smpl_joint_ids[i] gives the SMPL joint index that corresponds to phys_humanoid_v3 joint i
        smpl_joints = out.joints[:, smpl_joint_ids, :]  # (T, J_phys, 3)
        # Now smpl_joints[:, i, :] corresponds to target_pos[:, i, :] for phys_humanoid_v3 joint i

        # Data term: match joint positions
        # Each element i: smpl_joints[:, i, :] should match target_pos[:, i, :]
        loss_data = (smpl_joints - target_pos).pow(2).mean()
        
        # Temporal smoothness: encourage smooth pose changes
        if T > 1:
            loss_smooth_orient = (global_orient[1:] - global_orient[:-1]).pow(2).mean()
            loss_smooth_pose = (body_pose[1:] - body_pose[:-1]).pow(2).mean()
            loss_smooth = 0.1 * (loss_smooth_orient + loss_smooth_pose)
        else:
            loss_smooth = torch.tensor(0.0, device=device)
        
        # Light regularization: prevent extreme poses
        loss_reg = 1e-5 * (body_pose.pow(2).mean() + global_orient.pow(2).mean())

        loss = loss_data + loss_smooth + loss_reg
        loss.backward()
        optimizer.step()
        
        # Reduce learning rate for fine-tuning in later iterations
        if iter_idx == num_iters // 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.1

    # Fix pelvis orientation: apply rotations to fix forward/backward issues
    # Only use Y-axis rotation to avoid hunchback (X-axis rotation causes hunchback)
    # 180-degree rotation around Y axis to flip forward/backward (pelvis faces forward)
    flip_rotation_y = np.array([0.0, math.pi, 0.0])  # 180 deg around Y axis only
    flip_rot_y_matrix = R.from_rotvec(flip_rotation_y).as_matrix()
    flip_rot_matrix = flip_rot_y_matrix  # Only Y rotation, no X rotation to avoid hunchback
    
    # Apply rotation to each frame's global_orient
    global_orient_fixed = global_orient.detach().cpu().numpy()
    transl_np = transl.detach().cpu().numpy()
    
    for t in range(T):
        # Convert current global_orient to rotation matrix
        current_rot = R.from_rotvec(global_orient_fixed[t]).as_matrix()
        # Compose with flip rotation: flip_rot @ current_rot
        new_rot = flip_rot_matrix @ current_rot
        global_orient_fixed[t] = R.from_matrix(new_rot).as_rotvec()
    
    # Align body facing direction with walking direction
    # Calculate walking direction from root translation
    if T > 1:
        # Compute velocity direction (walking direction)
        transl_vel = transl_np[1:] - transl_np[:-1]  # (T-1, 3)
        # Use average velocity direction for smoother alignment
        avg_vel = np.mean(transl_vel, axis=0)  # (3,)
        avg_vel_norm = np.linalg.norm(avg_vel)
        
        if avg_vel_norm > 1e-3:  # If there's significant movement
            # Normalize velocity to get direction
            walking_dir = avg_vel / avg_vel_norm  # (3,)
            
            # For each frame, align body forward direction with walking direction
            # SMPL forward direction is typically +X or +Y in local frame
            # We'll use the Y-axis as forward (common in SMPL)
            smpl_forward_local = np.array([0.0, 1.0, 0.0])  # Y-forward in SMPL local frame
            
            for t in range(T):
                # Get current body orientation
                body_rot = R.from_rotvec(global_orient_fixed[t]).as_matrix()
                # Transform local forward to world space
                body_forward_world = body_rot @ smpl_forward_local
                
                # Check if body is facing opposite to walking direction
                dot_product = np.dot(body_forward_world[:2], walking_dir[:2])  # Use XY plane only
                
                # If facing opposite direction (dot < 0), rotate 180 degrees around Z
                if dot_product < 0:
                    # Add 180-degree rotation around Z axis
                    z_rotation = R.from_rotvec(np.array([0.0, 0.0, math.pi])).as_matrix()
                    body_rot_aligned = body_rot @ z_rotation
                    global_orient_fixed[t] = R.from_matrix(body_rot_aligned).as_rotvec()
    
    global_orient_fixed = torch.from_numpy(global_orient_fixed).to(device)
    poses = torch.cat([global_orient_fixed, body_pose], dim=-1)  # (T, 72)

    out_dict = {
        "poses": poses.detach().cpu().numpy().astype(np.float32),
        "trans": transl.detach().cpu().numpy().astype(np.float32),
        "fps": float(motion.fps),
    }
    return out_dict


def fit_smpl_to_motion_skeleton_aware(
    motion: SkeletonMotion,
    num_iters: int = 200,
    lr: float = 1e-2,
    device: str = "cpu",
    use_network_init: bool = True,
    network_ckpt_path: str = None,
):
    """
    使用Skeleton-Aware网络方法从15关节phys_humanoid_v3运动恢复24关节SMPL参数。
    该方法结合了深度学习和优化的优点：
    1. 使用skeleton-aware网络预测24个SMPL关节位置（快速初始化）
    2. 使用优化方法精化预测结果（精确匹配）
    
    参考论文: Skeleton-Aware Networks for Deep Motion Retargeting (Aberman et al., 2020)
    
    Args:
        motion: SkeletonMotion对象，包含phys_humanoid_v3的15关节运动
        num_iters: 优化迭代次数
        lr: 学习率
        device: 设备
        use_network_init: 是否使用网络初始化（如果False，则仅使用优化方法）
        network_ckpt_path: 预训练网络权重路径（可选）
    
    Returns:
        dict with poses (T, 72), trans (T, 3), fps
    """
    motion = motion.to(device=device) if hasattr(motion, "to") else motion
    
    # 获取15关节位置
    target_pos = motion.global_translation  # (T, 15, 3)
    T, J_phys, _ = target_pos.shape
    
    smpl_joint_ids = joints_to_use["from_smpl_original_to_phys_humanoid_v3"]
    assert len(smpl_joint_ids) == J_phys, f"Joint count mismatch: {len(smpl_joint_ids)} vs {J_phys}"
    
    target_pos = target_pos.to(device)
    
    # 构建SMPL模型
    bm = get_body_model("SMPL", "NEUTRAL", batch_size=T, debug=False)
    bm = bm.to(device)
    
    # 步骤1: 使用skeleton-aware网络预测所有24个SMPL关节位置
    use_network_prediction = False
    predicted_smpl_pos = None
    
    if use_network_init:
        print("使用Skeleton-Aware网络预测24关节SMPL位置...")
        try:
            # 创建网络模型
            network_model = create_default_predictor(
                pretrained_path=network_ckpt_path,
                device=device
            )
            network_model.eval()
            
            # 预测24关节位置
            with torch.no_grad():
                # 输入: (T, 15, 3) phys_humanoid_v3关节位置
                predicted_smpl_pos = network_model(target_pos)  # (T, 24, 3)
            
            print(f"网络预测完成，形状: {predicted_smpl_pos.shape}")
            
            # 提取对应的15个关节用于初始化验证
            predicted_15_joints = predicted_smpl_pos[:, smpl_joint_ids, :]  # (T, 15, 3)
            init_error = (predicted_15_joints - target_pos).pow(2).mean().item()
            print(f"网络预测的15关节误差: {init_error:.6f}")
            
            # 如果误差太大，说明网络未训练或不适用，不使用预测结果
            if init_error < 1.0:  # 阈值可以根据需要调整
                use_network_prediction = True
                print("网络预测质量良好，将用于初始化优化")
            else:
                print(f"网络预测误差较大 ({init_error:.6f})，将仅使用优化方法")
                use_network_prediction = False
        except Exception as e:
            print(f"网络预测失败，回退到标准优化方法: {e}")
            use_network_prediction = False
    
    # 步骤2: 优化SMPL参数以匹配目标关节位置
    # 初始化SMPL参数
    if use_network_prediction:
        # 使用网络预测的位置进行更智能的初始化
        print("基于网络预测初始化SMPL参数...")
        
        # 初始化全局方向和平移
        transl_init = motion.root_translation.to(device)
        
        # 初始化姿态（使用零初始化，然后通过优化匹配）
        global_orient_init = torch.zeros(T, 3, device=device, requires_grad=True)
        body_pose_init = torch.zeros(T, 69, device=device, requires_grad=True)
        transl_init_param = transl_init.clone().detach().requires_grad_(True)
        
        # 快速预优化以接近网络预测的位置
        print("预优化以匹配网络预测...")
        init_optimizer = torch.optim.Adam([global_orient_init, body_pose_init, transl_init_param], lr=lr * 2)
        for init_iter in range(min(50, num_iters // 4)):
            init_optimizer.zero_grad()
            out_init = bm(global_orient=global_orient_init, body_pose=body_pose_init, transl=transl_init_param)
            
            # 匹配15个对应关节
            smpl_joints_init = out_init.joints[:, smpl_joint_ids, :]
            loss_init = (smpl_joints_init - target_pos).pow(2).mean()
            
            # 可选: 也匹配所有24个关节（如果网络预测可用）
            if use_network_prediction:
                loss_init += 0.5 * (out_init.joints[:, :24, :] - predicted_smpl_pos).pow(2).mean()
            
            loss_init.backward()
            init_optimizer.step()
        
        global_orient = global_orient_init.detach().clone().requires_grad_(True)
        body_pose = body_pose_init.detach().clone().requires_grad_(True)
        transl = transl_init_param.detach().clone().requires_grad_(True)
    else:
        # 标准初始化
        global_orient = torch.zeros(T, 3, device=device, requires_grad=True)
        body_pose = torch.zeros(T, 69, device=device, requires_grad=True)
        transl = motion.root_translation.to(device).clone().detach().requires_grad_(True)
    
    # 主优化循环
    print(f"开始主优化循环 ({num_iters} 次迭代)...")
    optimizer = torch.optim.Adam([global_orient, body_pose, transl], lr=lr)
    
    for iter_idx in range(num_iters):
        optimizer.zero_grad()
        
        out = bm(global_orient=global_orient, body_pose=body_pose, transl=transl)
        smpl_joints = out.joints[:, smpl_joint_ids, :]  # (T, 15, 3)
        
        # 数据项: 匹配15个对应关节位置
        loss_data = (smpl_joints - target_pos).pow(2).mean()
        
        # 可选: 如果使用网络预测，也匹配所有24个关节
        if use_network_prediction and iter_idx < num_iters // 2:
            loss_data += 0.3 * (out.joints[:, :24, :] - predicted_smpl_pos.detach()).pow(2).mean()
        
        # 时间平滑性
        if T > 1:
            loss_smooth_orient = (global_orient[1:] - global_orient[:-1]).pow(2).mean()
            loss_smooth_pose = (body_pose[1:] - body_pose[:-1]).pow(2).mean()
            loss_smooth = 0.1 * (loss_smooth_orient + loss_smooth_pose)
        else:
            loss_smooth = torch.tensor(0.0, device=device)
        
        # 正则化
        loss_reg = 1e-5 * (body_pose.pow(2).mean() + global_orient.pow(2).mean())
        
        loss = loss_data + loss_smooth + loss_reg
        loss.backward()
        optimizer.step()
        
        # 学习率衰减
        if iter_idx == num_iters // 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.1
        
        if (iter_idx + 1) % 50 == 0:
            current_error = loss_data.item()
            print(f"迭代 {iter_idx + 1}/{num_iters}, 损失: {current_error:.6f}")
    
    # 修复骨盆方向（尝试不同的旋转组合来避免驼背）
    # 注意：X轴旋转可能导致驼背，先只尝试Y轴旋转
    global_orient_fixed = global_orient.detach().cpu().numpy()
    transl_np = transl.detach().cpu().numpy()
    
    # 尝试：只使用Y轴180度旋转（前后翻转），移除X轴旋转来避免驼背
    flip_rotation_y = np.array([0.0, math.pi, 0.0])  # 180 deg around Y axis only
    flip_rot_y_matrix = R.from_rotvec(flip_rotation_y).as_matrix()
    
    for t in range(T):
        current_rot = R.from_rotvec(global_orient_fixed[t]).as_matrix()
        # 只应用Y轴旋转，避免X轴旋转导致的驼背
        new_rot = flip_rot_y_matrix @ current_rot
        global_orient_fixed[t] = R.from_matrix(new_rot).as_rotvec()
    
    # 对齐身体朝向与行走方向
    if T > 1:
        transl_vel = transl_np[1:] - transl_np[:-1]
        avg_vel = np.mean(transl_vel, axis=0)
        avg_vel_norm = np.linalg.norm(avg_vel)
        
        if avg_vel_norm > 1e-3:
            walking_dir = avg_vel / avg_vel_norm
            smpl_forward_local = np.array([0.0, 1.0, 0.0])
            
            for t in range(T):
                body_rot = R.from_rotvec(global_orient_fixed[t]).as_matrix()
                body_forward_world = body_rot @ smpl_forward_local
                dot_product = np.dot(body_forward_world[:2], walking_dir[:2])
                
                if dot_product < 0:
                    z_rotation = R.from_rotvec(np.array([0.0, 0.0, math.pi])).as_matrix()
                    body_rot_aligned = body_rot @ z_rotation
                    global_orient_fixed[t] = R.from_matrix(body_rot_aligned).as_rotvec()
    
    global_orient_fixed = torch.from_numpy(global_orient_fixed).to(device)
    poses = torch.cat([global_orient_fixed, body_pose], dim=-1)  # (T, 72)
    
    out_dict = {
        "poses": poses.detach().cpu().numpy().astype(np.float32),
        "trans": transl.detach().cpu().numpy().astype(np.float32),
        "fps": float(motion.fps),
    }
    return out_dict


def fit_smpl_from_poses_direct(
    amp_poses: torch.Tensor,
    num_iters: int = 10,
    device: str = "cpu",
):
    """
    Alternative approach: Direct pose mapping from AMP/phys_humanoid_v3 poses to SMPL poses.
    This is faster but requires axis-angle poses as input (not joint positions).
    
    Args:
        amp_poses: (T, 15, 3) axis-angle poses from AMP/phys_humanoid_v3
        num_iters: Number of refinement iterations (optional IK refinement)
        device: Device to use
    
    Returns:
        dict with poses (T, 72), trans (T, 3), fps
    """
    T = amp_poses.shape[0]
    amp_poses = amp_poses.to(device)
    
    # Build SMPL model
    bm = get_body_model("SMPL", "NEUTRAL", batch_size=T, debug=False)
    bm = bm.to(device)
    
    # Step 1: Direct mapping - map 15 poses to 24 SMPL poses
    smpl_pose = torch.zeros(T, 24, 3, device=device)
    
    # Map known joints (using the joint mapping, but for poses)
    # Note: This assumes the pose order matches the joint order
    smpl_joint_ids = joints_to_use["from_smpl_original_to_phys_humanoid_v3"]
    for i, smpl_joint_idx in enumerate(smpl_joint_ids):
        smpl_pose[:, smpl_joint_idx, :] = amp_poses[:, i, :]
    
    # Step 2: Missing joints (spine*3, neck, head, collar*2) remain zero
    # This is fine for initialization - they'll be refined if needed
    
    # Step 3: Optional IK refinement using effector positions
    # Extract effector positions (ankles and wrists) from current pose
    effector_smpl_ids = [7, 8, 20, 21]  # Left ankle, right ankle, left wrist, right wrist
    
    # Convert to body_pose and global_orient format
    global_orient = smpl_pose[:, 0, :].clone().requires_grad_(True)
    body_pose = smpl_pose[:, 1:, :].reshape(T, 69).clone().requires_grad_(True)
    transl = torch.zeros(T, 3, device=device, requires_grad=True)
    
    # Refinement iterations
    if num_iters > 0:
        optimizer = torch.optim.Adam([global_orient, body_pose, transl], lr=1e-2)
        
        # Get target effector positions from current pose
        with torch.no_grad():
            out_init = bm(global_orient=global_orient, body_pose=body_pose, transl=transl)
            target_effector_pos = out_init.joints[:, effector_smpl_ids, :].detach()
        
        for iter_idx in range(num_iters):
            optimizer.zero_grad()
            out = bm(global_orient=global_orient, body_pose=body_pose, transl=transl)
            effector_pos = out.joints[:, effector_smpl_ids, :]
            loss = (effector_pos - target_effector_pos).pow(2).mean()
            loss.backward()
            optimizer.step()
    
    poses = torch.cat([global_orient, body_pose], dim=-1)  # (T, 72)
    
    return {
        "poses": poses.detach().cpu().numpy().astype(np.float32),
        "trans": transl.detach().cpu().numpy().astype(np.float32),
        "fps": 30.0,  # Default, should be provided as parameter
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Approximate recovery of 24-joint SMPL params from a ref_motion.npy "
            "(SkeletonMotion of phys_humanoid_v3)."
        )
    )
    parser.add_argument(
        "--ref_motion",
        type=str,
        required=True,
        help="Path to ref_motion.npy produced by generate_motion.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output .npy path for recovered SMPL-like params. "
            "If not given, writes 'recovered_smpl24_params.npy' "
            "next to this script."
        ),
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Number of optimization iterations.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use: 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="optimization",
        choices=["optimization", "skeleton_aware", "hybrid"],
        help="恢复方法: 'optimization' (纯优化), 'skeleton_aware' (网络+优化), 'hybrid' (网络初始化+优化)",
    )
    parser.add_argument(
        "--network_ckpt",
        type=str,
        default=None,
        help="预训练skeleton-aware网络权重路径（可选）",
    )

    args = parser.parse_args()

    ref_motion_path = args.ref_motion
    if args.output is None:
        script_dir = osp.dirname(osp.abspath(__file__))
        out_path = osp.join(script_dir, "recovered_smpl24_params.npy")
    else:
        out_path = args.output

    if not osp.isfile(ref_motion_path):
        raise FileNotFoundError(f"ref_motion file not found: {ref_motion_path}")

    motion = SkeletonMotion.from_file(ref_motion_path)

    # 根据选择的方法进行恢复
    if args.method == "optimization":
        print("使用纯优化方法...")
        params = fit_smpl_to_motion(
            motion,
            num_iters=args.iters,
            lr=args.lr,
            device=args.device,
        )
    elif args.method in ["skeleton_aware", "hybrid"]:
        if not SKELETON_AWARE_AVAILABLE:
            raise ImportError(
                "skeleton_aware方法需要skeleton_aware_network模块，但导入失败。"
                "请确保skeleton_aware_network.py文件在同一目录下。"
            )
        print(f"使用Skeleton-Aware网络方法 ({args.method})...")
        params = fit_smpl_to_motion_skeleton_aware(
            motion,
            num_iters=args.iters,
            lr=args.lr,
            device=args.device,
            use_network_init=True,
            network_ckpt_path=args.network_ckpt,
        )
    else:
        raise ValueError(f"未知的方法: {args.method}")

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    np.save(out_path, params)
    print(f"Saved approximate 24-joint SMPL params to: {out_path}")


if __name__ == "__main__":
    main()


