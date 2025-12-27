# 运动重定向与可视化流程说明

本文档介绍如何使用Skeleton-Aware网络进行运动重定向，以及如何可视化重定向结果。

## 📁 文件位置

### 主要脚本

1. **重定向脚本**: `tokenhsi/data/dataset_amass_loco/retarget_with_new_network.py`
   - 功能：将phys_humanoid_v3的15关节运动重定向到SMPL的24关节参数
   - 特点：考虑骨骼长度比例，确保重定向后的模型比例正确

2. **可视化脚本**: `tokenhsi/data/dataset_amass_loco/visualize_smpl24.py`
   - 功能：将SMPL参数文件可视化为MP4视频
   - 特点：使用SMPL mesh渲染，包含光照和阴影效果

## 🚀 使用流程

### 前置要求

1. **环境设置**
   ```bash
   # 激活conda环境
   conda activate tokenhsi
   
   # 确保已安装所需依赖
   # - torch
   # - numpy
   # - matplotlib
   # - smplx (用于SMPL模型)
   ```

2. **SMPL模型文件**
   确保SMPL模型文件已正确放置在：
   ```
   body_models/smpl/
   ├── SMPL_NEUTRAL.pkl
   ├── SMPL_MALE.pkl
   └── SMPL_FEMALE.pkl
   ```

### 步骤1: 运动重定向

使用 `retarget_with_new_network.py` 将phys_humanoid_v3的运动数据重定向为SMPL参数。

#### 基本用法

```bash
cd /home/artanis/TokenHSI/tokenhsi/data/dataset_amass_loco

python retarget_with_new_network.py \
    --input "motions/ACCAD/ACCAD+__+Female1Walking_c3d+__+B3_-_walk1_stageii/phys_humanoid_v3/ref_motion.npy" \
    --output "recovered_smpl24_params.npy" \
    --device "cpu" \
    --num_iters 200 \
    --lr 1e-2
```

#### 参数说明

- `--input`: **必需**，输入ref_motion.npy文件路径
  - 可以是相对路径（相对于脚本目录）或绝对路径
  - 示例：`"motions/ACCAD/ACCAD+__+Female1Walking_c3d+__+B3_-_walk1_stageii/phys_humanoid_v3/ref_motion.npy"`
  - 或绝对路径：`"/home/artanis/TokenHSI/tokenhsi/data/dataset_amass_loco/motions/.../ref_motion.npy"`

- `--output`: 输出SMPL参数文件路径（默认：`recovered_smpl24_params.npy`）
  - 可以是相对路径或绝对路径

- `--device`: 计算设备（默认：`cuda`如果可用，否则`cpu`）
  - 选项：`cpu` 或 `cuda`

- `--num_iters`: 优化迭代次数（默认：200）
  - 更多迭代通常能获得更好的结果，但需要更长时间

- `--lr`: 学习率（默认：1e-2）
  - 如果优化不稳定，可以尝试降低学习率（如5e-3）

- `--no_network_init`: 不使用网络初始化（仅使用优化方法）
  - 添加此标志将跳过网络预测步骤

#### 使用自定义输入文件

```bash
# 使用相对路径
python retarget_with_new_network.py \
    --input "motions/YOUR_DATASET/YOUR_MOTION/phys_humanoid_v3/ref_motion.npy" \
    --output "your_output.npy"

# 使用绝对路径
python retarget_with_new_network.py \
    --input "/absolute/path/to/ref_motion.npy" \
    --output "/absolute/path/to/output.npy"
```

#### 输出说明

脚本会生成一个包含以下内容的`.npy`文件：
- `poses`: (T, 72) - SMPL姿态参数（3维全局方向 + 69维身体姿态）
- `trans`: (T, 3) - 全局平移
- `fps`: 帧率

### 步骤2: 可视化结果

使用 `visualize_smpl24.py` 将SMPL参数文件可视化为MP4视频。

#### 基本用法

```bash
cd /home/artanis/TokenHSI/tokenhsi/data/dataset_amass_loco

python visualize_smpl24.py \
    --params "recovered_smpl24_params.npy" \
    --output "recovered_smpl24_params.mp4" \
    --device "cpu"
```

#### 参数说明

- `--params`: **必需**，SMPL参数文件路径（.npy文件）
  - 这是步骤1生成的输出文件

- `--output`: 输出视频路径（默认：`recovered_smpl24_params.mp4`）
  - 必须是`.mp4`格式

- `--device`: 计算设备（默认：`cpu`）
  - 选项：`cpu` 或 `cuda`

- `--max_frames`: 最大渲染帧数（默认：None，表示渲染所有帧）
  - 例如：`--max_frames 100` 只渲染前100帧

- `--fps`: 输出视频帧率（默认：30）

#### 完整示例

```bash
# 可视化所有帧
python visualize_smpl24.py \
    --params "recovered_smpl24_params.npy" \
    --output "recovered_smpl24_params.mp4"

# 只可视化前100帧
python visualize_smpl24.py \
    --params "recovered_smpl24_params.npy" \
    --output "recovered_smpl24_params.mp4" \
    --max_frames 100

# 使用GPU加速（如果可用）
python visualize_smpl24.py \
    --params "recovered_smpl24_params.npy" \
    --output "recovered_smpl24_params.mp4" \
    --device "cuda"
```

## 📋 完整工作流示例

### 示例1: 处理单个运动文件

```bash
# 1. 进入工作目录
cd /home/artanis/TokenHSI/tokenhsi/data/dataset_amass_loco

# 2. 运行重定向
python retarget_with_new_network.py \
    --input "motions/ACCAD/ACCAD+__+Female1Walking_c3d+__+B3_-_walk1_stageii/phys_humanoid_v3/ref_motion.npy" \
    --output "recovered_smpl24_params.npy" \
    --device "cpu" \
    --num_iters 200

# 3. 可视化结果
python visualize_smpl24.py \
    --params "recovered_smpl24_params.npy" \
    --output "recovered_smpl24_params.mp4" \
    --device "cpu"
```

### 示例2: 批量处理多个文件

```bash
#!/bin/bash
cd /home/artanis/TokenHSI/tokenhsi/data/dataset_amass_loco

# 定义输入文件列表
INPUT_FILES=(
    "motions/ACCAD/ACCAD+__+Female1Walking_c3d+__+B3_-_walk1_stageii/phys_humanoid_v3/ref_motion.npy"
    # 添加更多文件...
)

# 处理每个文件
for input_file in "${INPUT_FILES[@]}"; do
    # 生成输出文件名
    output_name=$(basename "$(dirname "$(dirname "$input_file")")")
    output_npy="${output_name}_smpl24_params.npy"
    output_mp4="${output_name}_smpl24_params.mp4"
    
    echo "处理: $input_file"
    
    # 重定向
    python retarget_with_new_network.py \
        --input "$input_file" \
        --output "$output_npy" \
        --device "cpu"
    
    # 可视化
    python visualize_smpl24.py \
        --params "$output_npy" \
        --output "$output_mp4" \
        --device "cpu"
    
    echo "完成: $output_mp4"
done
```

## 🔍 技术细节

### 重定向方法特点

1. **骨骼长度比例匹配**
   - 获取SMPL T-pose的默认骨骼长度作为基准
   - 计算phys_humanoid_v3的平均骨骼长度（多帧平均）
   - 计算比例因子，确保相对比例正确
   - 优化betas参数以调整SMPL体型

2. **双重损失函数**
   - 位置损失：匹配15个共享关节的位置
   - 骨骼比例损失：确保骨骼长度比例正确
   - 骨骼绝对长度损失：作为辅助约束

3. **优化参数**
   - `global_orient`: 全局方向（3维）
   - `body_pose`: 身体姿态（69维）
   - `transl`: 全局平移（3维）
   - `betas`: 体型参数（10维）

### 可视化特点

1. **SMPL Mesh渲染**
   - 使用完整的SMPL mesh（6890个顶点，13776个面）
   - 包含光照和阴影效果
   - 自动检测行走方向并调整相机角度

2. **相机设置**
   - 相机方向垂直于行走方向
   - 自动调整elevation和azimuth以获得最佳视角

## ⚠️ 常见问题

### 1. 找不到输入文件

**错误**: `FileNotFoundError: 找不到输入文件: ...`

**解决**: 
- 检查文件路径是否正确
- 使用绝对路径避免路径问题
- 确保文件确实存在

### 2. 网络预测失败

**警告**: `⚠ 网络预测失败，回退到标准优化方法`

**说明**: 这是正常的，脚本会自动回退到纯优化方法。优化方法仍然能够产生良好的结果。

### 3. 内存不足

**解决**:
- 使用`--device cpu`而不是cuda
- 减少`--num_iters`迭代次数
- 使用`--max_frames`限制可视化帧数

### 4. 可视化视频质量不佳

**解决**:
- 增加优化迭代次数（`--num_iters 300`或更多）
- 调整学习率（`--lr 5e-3`）
- 检查输入运动数据质量

## 📝 输出文件格式

### SMPL参数文件 (.npy)

```python
{
    'poses': np.ndarray,  # (T, 72) float32
    'trans': np.ndarray,   # (T, 3) float32
    'fps': float          # 帧率
}
```

### 视频文件 (.mp4)

- 格式：MP4
- 编码：H.264
- 包含所有帧的SMPL mesh渲染

## 🔗 相关文件

- `tokenhsi/data/dataset_amass_loco/skeleton_aware_network.py`: Skeleton-Aware网络实现
- `tokenhsi/data/dataset_amass_loco/recover_smpl24_from_ref_motion.py`: 基础重定向函数
- `tokenhsi/data/dataset_amass_loco/retarget_from_rollout.py`: 处理rollout数据的重定向脚本

## 📚 参考文献

本实现基于以下论文：
- **Skeleton-Aware Networks for Deep Motion Retargeting** (Aberman et al., 2020)

## 💡 提示

1. **性能优化**: 如果使用GPU，设置`--device cuda`可以显著加速
2. **批量处理**: 对于多个文件，建议编写脚本批量处理
3. **结果检查**: 建议先可视化检查结果，确认重定向质量
4. **参数调整**: 如果结果不理想，可以尝试调整`--num_iters`和`--lr`参数

---

如有问题，请检查：
1. 环境是否正确配置
2. 输入文件格式是否正确
3. SMPL模型文件是否存在
4. 路径是否正确
