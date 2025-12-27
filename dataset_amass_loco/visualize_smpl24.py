#!/usr/bin/env python
"""
可视化SMPL参数文件，生成视频
只渲染SMPL mesh，不添加额外元素
"""
import sys
import os
import os.path as osp
import argparse
import numpy as np
import torch

# 添加路径
script_dir = osp.dirname(osp.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
sys.path.append(osp.dirname(osp.dirname(osp.dirname(script_dir))))

from body_models.model_loader import get_body_model


def visualize_smpl_params(
    params_path,
    output_path,
    max_frames=None,
    device="cpu",
    fps=30,
):
    """
    可视化SMPL参数文件 - 只渲染mesh
    
    Args:
        params_path: SMPL参数文件路径 (.npy)
        output_path: 输出视频路径 (.mp4)
        max_frames: 最大渲染帧数（None表示全部）
        device: 计算设备
        fps: 输出视频帧率
    """
    print("=" * 60)
    print("可视化SMPL参数")
    print("=" * 60)
    print(f"输入文件: {params_path}")
    print(f"输出文件: {output_path}")
    print(f"设备: {device}")
    print("=" * 60)
    
    # 加载参数
    print("\n加载参数文件...")
    params = np.load(params_path, allow_pickle=True).item()
    
    poses = params["poses"]  # (T, 72)
    trans = params["trans"]  # (T, 3)
    file_fps = params.get("fps", fps)
    
    T = poses.shape[0]
    # 如果用户没有指定max_frames，使用所有帧
    if max_frames is not None and max_frames > 0:
        T = min(T, max_frames)
        poses = poses[:T]
        trans = trans[:T]
    
    print(f"  帧数: {T}")
    print(f"  原始FPS: {file_fps}")
    print(f"  输出FPS: {fps}")
    
    # 转换为torch tensor
    poses_tensor = torch.from_numpy(poses).float().to(device)
    trans_tensor = torch.from_numpy(trans).float().to(device)
    
    # 分离global_orient和body_pose
    global_orient = poses_tensor[:, :3]  # (T, 3)
    body_pose = poses_tensor[:, 3:72]  # (T, 69)
    
    # 构建SMPL模型
    print("\n构建SMPL模型...")
    bm = get_body_model("SMPL", "NEUTRAL", batch_size=T, debug=False)
    bm = bm.to(device)
    
    # 生成mesh
    print("\n生成SMPL mesh...")
    with torch.no_grad():
        output = bm(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=trans_tensor,
        )
        vertices = output.vertices.cpu().numpy()  # (T, 6890, 3)
        # faces可能是tensor或numpy数组
        if hasattr(bm.faces, 'cpu'):
            faces = bm.faces.cpu().numpy()  # (13776, 3)
        else:
            faces = np.array(bm.faces)  # (13776, 3)
    
    print(f"  顶点数: {vertices.shape[1]}")
    print(f"  面数: {faces.shape[0]}")
    
    # 使用matplotlib进行可视化
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        HAS_MATPLOTLIB = True
    except ImportError:
        print("\n错误: 需要安装matplotlib")
        print("请运行: pip install matplotlib")
        return
    
    print("\n使用matplotlib渲染SMPL mesh...")
    
    # 计算行走方向（从root translation的变化）
    if T > 1:
        root_positions = trans_tensor.cpu().numpy()  # (T, 3)
        directions = np.diff(root_positions, axis=0)  # (T-1, 3)
        directions_xy = directions[:, :2]  # (T-1, 2)
        avg_direction = np.mean(directions_xy, axis=0)  # (2,)
        if np.linalg.norm(avg_direction) > 1e-6:
            avg_direction = avg_direction / np.linalg.norm(avg_direction)
        else:
            avg_direction = np.array([1.0, 0.0])
    else:
        avg_direction = np.array([1.0, 0.0])
    
    # 计算垂直于行走方向的视角角度
    walk_angle = np.arctan2(avg_direction[1], avg_direction[0])
    camera_azimuth = np.degrees(walk_angle) + 90
    
    print(f"  检测到的行走方向: {avg_direction}")
    print(f"  相机视角 (azimuth): {camera_azimuth:.1f}度")
    
    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴范围
    all_vertices = vertices.reshape(-1, 3)
    x_min, x_max = all_vertices[:, 0].min() - 0.5, all_vertices[:, 0].max() + 0.5
    y_min, y_max = all_vertices[:, 1].min() - 0.5, all_vertices[:, 1].max() + 0.5
    z_min, z_max = all_vertices[:, 2].min() - 0.1, all_vertices[:, 2].max() + 0.5
    
    def update_frame(frame_idx):
        ax.clear()
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        ax.set_axis_off()  # 隐藏坐标轴
        
        # 获取当前帧的mesh
        frame_vertices = vertices[frame_idx]  # (6890, 3)
        triangles = frame_vertices[faces]  # (13776, 3, 3)
        
        # 计算每个面的法向量用于光照效果
        # 这样可以增强3D感
        num_triangles = len(triangles)
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0]
        )
        # 归一化法向量
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除零
        face_normals = face_normals / norms
        
        # 定义光源方向（从右前上方照射，增强3D感）
        # 根据相机视角调整光源方向，使其始终从合适角度照射
        light_azimuth_rad = np.radians(camera_azimuth)
        light_dir = np.array([
            np.cos(light_azimuth_rad) * 0.5,
            np.sin(light_azimuth_rad) * 0.5,
            0.8
        ])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        # 计算每个面的光照强度（点积）
        # 值在-1到1之间，我们映射到0.3到1.0之间
        light_intensity = np.dot(face_normals, light_dir)
        light_intensity = (light_intensity + 1.0) / 2.0  # 映射到0-1
        light_intensity = 0.3 + 0.7 * light_intensity  # 映射到0.3-1.0
        
        # 为每个面创建颜色（基于光照强度）
        base_color = np.array([0.7, 0.7, 0.9])  # 浅蓝色
        face_colors = base_color * light_intensity[:, np.newaxis]  # (N, 3)
        # 添加alpha通道
        face_colors_rgba = np.column_stack([face_colors, np.ones(num_triangles)])  # (N, 4)
        
        # 渲染mesh，使用基于光照的颜色
        # 添加细边缘线以增强3D轮廓感
        collection = Poly3DCollection(
            triangles,
            facecolors=face_colors_rgba,
            edgecolors=(0.3, 0.3, 0.3, 0.3),  # 深灰色细边缘，增强轮廓
            linewidths=0.3,
            alpha=1.0,
            shade=False  # 我们自己计算光照
        )
        ax.add_collection3d(collection)
        
        # 设置视角：调整elevation和azimuth让3D效果更明显
        # 从侧面看（垂直于行走方向），稍微俯视以看到更多3D细节
        # 增加elevation角度让视角更立体，能看到前后深度
        # 稍微调整azimuth让视角不是完全侧面，能看到一些正面
        ax.view_init(elev=30, azim=camera_azimuth + 10)  # 稍微偏移10度，增强3D感
        ax.set_box_aspect([1, 1, 1])
        
        # 确保坐标轴比例正确，避免挤压
        # 计算实际的坐标范围
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        max_range = max(x_range, y_range, z_range)
        
        # 设置相等的范围，避免某个方向被压缩
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_z = (z_min + z_max) / 2
        
        ax.set_xlim([center_x - max_range/2, center_x + max_range/2])
        ax.set_ylim([center_y - max_range/2, center_y + max_range/2])
        ax.set_zlim([center_z - max_range/2, center_z + max_range/2])
        
        # 设置背景为白色以增强对比
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.05)
        ax.yaxis.pane.set_alpha(0.05)
        ax.zaxis.pane.set_alpha(0.05)
        
        # 设置背景色
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
    
    print(f"\n创建动画 ({T} 帧)...")
    anim = FuncAnimation(fig, update_frame, frames=T, interval=1000/fps, repeat=True)
    
    print(f"\n保存视频到: {output_path}")
    os.makedirs(osp.dirname(output_path) if osp.dirname(output_path) else '.', exist_ok=True)
    
    try:
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='SMPL Visualizer'), bitrate=3000)
        anim.save(output_path, writer=writer)
    except Exception as e:
        print(f"使用FFMpeg保存失败: {e}")
        print("尝试使用pillow writer保存为GIF...")
        try:
            from matplotlib.animation import PillowWriter
            gif_path = output_path.replace('.mp4', '.gif')
            writer = PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer)
            print(f"已保存为GIF: {gif_path}")
        except Exception as e2:
            print(f"保存失败: {e2}")
            return
    
    plt.close()
    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)
    print(f"输出视频: {output_path}")
    print(f"帧数: {T}")
    print(f"FPS: {fps}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="可视化SMPL参数文件")
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="SMPL参数文件路径 (.npy)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="recovered_smpl24_params.mp4",
        help="输出视频路径 (.mp4)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="最大渲染帧数（None表示全部）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="计算设备 (cpu/cuda)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="输出视频帧率"
    )
    
    args = parser.parse_args()
    
    visualize_smpl_params(
        params_path=args.params,
        output_path=args.output,
        max_frames=args.max_frames,
        device=args.device,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
