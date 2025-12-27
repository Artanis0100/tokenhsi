"""
Skeleton-Aware Network for Motion Retargeting
Based on: Skeleton-Aware Networks for Deep Motion Retargeting (Aberman et al., 2020)

This module implements the skeleton-aware neural network framework from the paper,
which uses differentiable convolution, pooling, and unpooling operators that are
skeleton-aware, meaning they explicitly account for the skeleton's hierarchical 
structure and joint adjacency.

Key Components:
1. Skeleton-Aware Convolution: Temporal convolution with reflected padding
2. Average Skeletal Pooling (AP): Reduces skeleton to primal skeleton via edge merging
3. Skeletal Unpooling (UP): Expands from primal skeleton to target skeleton
4. Temporal Linear Upsampling (UpS): Upsamples along temporal dimension
5. Encoder-Decoder architecture with shared latent space

Reference: https://doi.org/10.1145/3386569.3392462
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


def build_adjacency_matrix(parent_indices, num_joints):
    """
    构建骨架的邻接矩阵（用于图卷积）
    Args:
        parent_indices: (J,) 每个关节的父关节索引，-1表示根关节
        num_joints: 关节数量
    Returns:
        adj_matrix: (J, J) 邻接矩阵，1表示有连接，0表示无连接
    """
    adj_matrix = torch.zeros(num_joints, num_joints, dtype=torch.float32)
    
    for child_idx, parent_idx in enumerate(parent_indices):
        if parent_idx >= 0:
            # 父子连接（双向）
            adj_matrix[child_idx, parent_idx] = 1.0
            adj_matrix[parent_idx, child_idx] = 1.0
        # 自连接
        adj_matrix[child_idx, child_idx] = 1.0
    
    return adj_matrix


class GraphConvolution(nn.Module):
    """
    图卷积层：基于骨架结构的图卷积操作
    参考：Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks"
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj_matrix):
        """
        Args:
            x: (B, T, J, C_in) 关节特征
            adj_matrix: (J, J) 邻接矩阵
        Returns:
            out: (B, T, J, C_out) 输出特征
        """
        B, T, J, C_in = x.shape
        
        # 归一化邻接矩阵（对称归一化）
        adj_matrix = adj_matrix.to(x.device)
        # 添加自连接并归一化
        degree = adj_matrix.sum(dim=1, keepdim=True)  # (J, 1)
        degree_sqrt_inv = torch.pow(degree + 1e-8, -0.5)
        adj_norm = degree_sqrt_inv * adj_matrix * degree_sqrt_inv.t()
        
        # 图卷积: (B, T, J, C_in) -> (B*T, J, C_in) -> (B*T, J, C_out) -> (B, T, J, C_out)
        x_reshaped = x.reshape(B * T, J, C_in)  # (B*T, J, C_in)
        support = torch.matmul(x_reshaped, self.weight)  # (B*T, J, C_out)
        output = torch.matmul(adj_norm, support)  # (B*T, J, C_out)
        
        if self.bias is not None:
            output = output + self.bias
        
        output = output.reshape(B, T, J, self.out_channels)
        return output


class SkeletonAttention(nn.Module):
    """
    骨架注意力机制：学习关节之间的重要性权重
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        """
        Args:
            x: (B, T, J, C) 关节特征
        Returns:
            out: (B, T, J, C) 注意力增强的特征
        """
        B, T, J, C = x.shape
        
        # 重塑为 (B*T, J, C)
        x_reshaped = x.reshape(B * T, J, C)
        
        # 计算query, key, value
        q = self.query(x_reshaped).reshape(B * T, J, self.num_heads, self.head_dim).transpose(1, 2)  # (B*T, H, J, d)
        k = self.key(x_reshaped).reshape(B * T, J, self.num_heads, self.head_dim).transpose(1, 2)  # (B*T, H, J, d)
        v = self.value(x_reshaped).reshape(B * T, J, self.num_heads, self.head_dim).transpose(1, 2)  # (B*T, H, J, d)
        
        # 计算注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B*T, H, J, J)
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.matmul(attn, v)  # (B*T, H, J, d)
        out = out.transpose(1, 2).reshape(B * T, J, C)  # (B*T, J, C)
        out = self.out_proj(out)  # (B*T, J, C)
        
        # 重塑回 (B, T, J, C)
        out = out.reshape(B, T, J, C)
        
        # 残差连接
        out = out + x
        
        return out


class TemporalLSTM(nn.Module):
    """
    时间建模：使用LSTM处理时间序列
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        output_size = hidden_size * 2 if bidirectional else hidden_size
        self.out_proj = nn.Linear(output_size, input_size)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, J, C) 关节特征
        Returns:
            out: (B, T, J, C) 时间增强的特征
        """
        B, T, J, C = x.shape
        
        # 对每个关节独立处理时间序列
        # (B, T, J, C) -> (B*J, T, C)
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * J, T, C)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x_reshaped)  # (B*J, T, H)
        
        # 投影回原始维度
        out = self.out_proj(lstm_out)  # (B*J, T, C)
        
        # 重塑回 (B, T, J, C)
        out = out.reshape(B, J, T, C).permute(0, 2, 1, 3)
        
        # 残差连接
        out = out + x
        
        return out


class ImprovedSkeletonAwareConv(nn.Module):
    """
    改进的骨架感知卷积：结合图卷积、注意力和时间建模
    """
    def __init__(self, in_channels, out_channels, kernel_size=15, use_gcn=True, use_attention=True, use_lstm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_gcn = use_gcn
        self.use_attention = use_attention
        self.use_lstm = use_lstm
        
        # 时间卷积
        self.temporal_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding=kernel_size // 2, padding_mode='reflect'
        )
        
        # 图卷积
        if use_gcn:
            self.gcn = GraphConvolution(in_channels, out_channels)
        
        # 注意力机制
        if use_attention:
            self.attention = SkeletonAttention(out_channels if use_gcn else in_channels)
        
        # 时间LSTM
        if use_lstm:
            self.temporal_lstm = TemporalLSTM(out_channels if use_gcn else in_channels, out_channels)
        
        # 特征融合
        num_modalities = 1 + (1 if use_gcn else 0) + (1 if use_lstm else 0)
        if num_modalities > 1:
            self.fusion = nn.Linear(out_channels * num_modalities, out_channels)
        else:
            self.fusion = nn.Identity()
        
        # 归一化
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x, parent_indices=None, adj_matrix=None):
        """
        Args:
            x: (B, T, J, C) 关节特征
            parent_indices: (J,) 父关节索引（用于构建邻接矩阵）
            adj_matrix: (J, J) 预计算的邻接矩阵（可选）
        Returns:
            out: (B, T, J, C_out) 输出特征
        """
        B, T, J, C = x.shape
        
        features = []
        
        # 1. 时间卷积
        x_temporal = x.permute(0, 2, 3, 1)  # (B, J, C, T)
        x_temporal = x_temporal.reshape(B * J, C, T)
        x_temporal = self.temporal_conv(x_temporal)  # (B*J, C_out, T)
        x_temporal = x_temporal.reshape(B, J, self.out_channels, T)
        x_temporal = x_temporal.permute(0, 3, 1, 2)  # (B, T, J, C_out)
        features.append(x_temporal)
        
        # 2. 图卷积
        if self.use_gcn:
            if adj_matrix is None and parent_indices is not None:
                adj_matrix = build_adjacency_matrix(parent_indices, J)
            elif adj_matrix is None:
                # 如果没有提供骨架信息，使用全连接图
                adj_matrix = torch.ones(J, J, device=x.device, dtype=torch.float32)
            
            x_gcn = self.gcn(x, adj_matrix)
            features.append(x_gcn)
        
        # 3. 时间LSTM
        if self.use_lstm:
            x_lstm_input = x_gcn if self.use_gcn else x
            x_lstm = self.temporal_lstm(x_lstm_input)
            # 如果LSTM输出维度不匹配，需要投影
            if x_lstm.shape[-1] != self.out_channels:
                x_lstm = F.linear(x_lstm, self.temporal_lstm.out_proj.weight[:, :self.out_channels])
            features.append(x_lstm)
        
        # 融合多模态特征
        if len(features) > 1:
            x_fused = torch.cat(features, dim=-1)  # (B, T, J, C_out * num_modalities)
            out = self.fusion(x_fused)  # (B, T, J, C_out)
        else:
            out = features[0]
        
        # 归一化
        out = self.norm(out)
        
        # 注意力机制
        if self.use_attention:
            out = self.attention(out)
        
        return out


class SkeletonAwareConv1D(nn.Module):
    """
    Skeleton-Aware 1D Convolution (from paper)
    
    Performs temporal convolution along the time dimension for each joint independently.
    Uses reflected padding as specified in the paper.
    
    The convolution is skeleton-aware in the sense that it operates on joint features
    while preserving the joint structure, allowing the network to learn temporal patterns
    that respect the skeleton hierarchy.
    """
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Temporal convolution with reflected padding (as per paper)
        # Padding is set to maintain temporal resolution when stride=1
        padding = kernel_size // 2 if stride == 1 else 0
        self.temporal_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, padding_mode='reflect'
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, J, C) joint features [batch, time, joints, channels]
        Returns:
            out: (B, T', J, C_out) output features (T' may differ if stride > 1)
        """
        B, T, J, C = x.shape
        
        # Reshape for temporal convolution: (B, T, J, C) -> (B*J, C, T)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * J, C, T)
        
        # Apply temporal convolution: (B*J, C, T) -> (B*J, C_out, T')
        x_conv = self.temporal_conv(x_reshaped)  # (B*J, C_out, T')
        
        # Reshape back: (B*J, C_out, T') -> (B, T', J, C_out)
        T_out = x_conv.shape[2]
        out = x_conv.reshape(B, J, self.out_channels, T_out).permute(0, 3, 1, 2)
        
        return out


class AverageSkeletalPooling(nn.Module):
    """
    Average Skeletal Pooling (AP) - from paper
    
    Reduces skeleton complexity by merging joints through edge merging operations.
    Maps from a higher-joint skeleton to a lower-joint (primal) skeleton.
    
    The pooling operation averages features from source joints that map to the same
    target joint in the primal skeleton. This is a key operation for creating a
    common latent space shared by homeomorphic skeletons.
    """
    def __init__(self, joint_mapping):
        """
        Args:
            joint_mapping: dict mapping from source joint indices to target joint indices
                          {source_idx: target_idx, ...}
        """
        super().__init__()
        self.joint_mapping = joint_mapping
        
        # Pre-compute reverse mapping for efficiency
        self.target_to_sources = defaultdict(list)
        for src_idx, tgt_idx in joint_mapping.items():
            self.target_to_sources[tgt_idx].append(src_idx)
        self.target_to_sources = dict(self.target_to_sources)
        
        # Get sorted target joint indices
        self.target_joints = sorted(set(joint_mapping.values()))
        self.num_target_joints = len(self.target_joints)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, J_source, C) source skeleton features
        Returns:
            out: (B, T, J_target, C) pooled features
        """
        B, T, J_source, C = x.shape
        
        # Pool features for each target joint
        pooled_features = []
        for target_idx in self.target_joints:
            # Get all source joints that map to this target
            source_indices = self.target_to_sources.get(target_idx, [])
            
            if source_indices:
                # Average pooling across mapped joints
                # Extract features: (B, T, len(source_indices), C)
                mapped_features = x[:, :, source_indices, :]
                # Average over joint dimension: (B, T, C)
                pooled = mapped_features.mean(dim=2)
            else:
                # Zero padding if no mapping exists (shouldn't happen with proper mapping)
                pooled = torch.zeros(B, T, C, device=x.device, dtype=x.dtype)
            
            pooled_features.append(pooled)
        
        # Stack: (B, T, J_target, C)
        out = torch.stack(pooled_features, dim=2)
        return out


class SkeletalUnpooling(nn.Module):
    """
    Skeletal Unpooling (UP) - from paper
    
    Expands from primal skeleton (lower-joint) to target skeleton (higher-joint).
    This is the inverse operation of skeletal pooling, mapping from the common
    latent space back to the target skeleton structure.
    
    For joints that don't have a direct mapping, the features are copied from
    the corresponding source joint or interpolated based on skeleton hierarchy.
    """
    def __init__(self, joint_mapping, target_num_joints):
        """
        Args:
            joint_mapping: dict mapping from source joint indices to target joint indices
                          {source_idx: target_idx, ...}
            target_num_joints: number of joints in target skeleton
        """
        super().__init__()
        self.joint_mapping = joint_mapping
        self.target_num_joints = target_num_joints
        
        # Pre-compute source to target mapping
        self.source_to_target = {src: tgt for src, tgt in joint_mapping.items()}
        self.mapped_targets = set(joint_mapping.values())
    
    def forward(self, x):
        """
        Args:
            x: (B, T, J_source, C) source skeleton features (from primal skeleton)
        Returns:
            out: (B, T, J_target, C) unpooled features
        """
        B, T, J_source, C = x.shape
        
        # Initialize output with zeros
        out = torch.zeros(B, T, self.target_num_joints, C, device=x.device, dtype=x.dtype)
        
        # Map known joints: copy features from source to target
        for source_idx, target_idx in self.joint_mapping.items():
            if source_idx < J_source and target_idx < self.target_num_joints:
                out[:, :, target_idx, :] = x[:, :, source_idx, :]
        
        # For unmapped target joints, use interpolation based on hierarchy
        # Simple strategy: use the nearest mapped joint or average of all mapped joints
        unmapped_targets = [j for j in range(self.target_num_joints) if j not in self.mapped_targets]
        
        if len(unmapped_targets) > 0:
            # Use average of all mapped joints as default (can be improved with hierarchy)
            if len(self.mapped_targets) > 0:
                mapped_features = [out[:, :, j, :] for j in sorted(self.mapped_targets)]
                avg_feature = torch.stack(mapped_features, dim=2).mean(dim=2)  # (B, T, C)
                for target_idx in unmapped_targets:
                    out[:, :, target_idx, :] = avg_feature
        
        return out


class TemporalLinearUpsampling(nn.Module):
    """
    Temporal Linear Upsampling (UpS) - from paper
    
    Performs linear interpolation along the temporal dimension to upsample
    the temporal resolution. This is used in the decoder to restore temporal
    resolution after downsampling in the encoder.
    """
    def __init__(self, scale_factor=2, mode='linear'):
        """
        Args:
            scale_factor: upsampling factor (typically 2)
            mode: 'linear' or 'nearest'
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x):
        """
        Args:
            x: (B, T, J, C) input features
        Returns:
            out: (B, T*scale_factor, J, C) upsampled features
        """
        B, T, J, C = x.shape
        
        # Reshape for interpolation: (B, T, J, C) -> (B*J*C, T)
        # We need to upsample along temporal dimension for each joint and channel independently
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * J * C, 1, T)  # (B*J*C, 1, T) - 3D for interpolate
        
        # Upsample along temporal dimension (last dimension)
        # Use 'trilinear' for 3D input or 'linear' for 2D, but we have 3D so use 'trilinear'
        # Actually, for 3D input with shape (N, C, D), we should use 'trilinear'
        # But we want 1D interpolation, so reshape to (B*J*C, T) and use 'linear' mode
        x_reshaped_2d = x_reshaped.squeeze(1)  # (B*J*C, T)
        x_reshaped_2d = x_reshaped_2d.unsqueeze(1)  # (B*J*C, 1, T) - 2D for linear interpolation
        
        out = F.interpolate(
            x_reshaped_2d,
            size=T * self.scale_factor,
            mode='linear',  # 1D linear interpolation
            align_corners=False
        )  # (B*J*C, 1, T*scale_factor)
        
        out = out.squeeze(1)  # (B*J*C, T*scale_factor)
        T_out = out.shape[1]
        
        # Reshape back: (B*J*C, T*scale_factor) -> (B, T*scale_factor, J, C)
        out = out.reshape(B, J, C, T_out).permute(0, 3, 1, 2)
        
        return out


class MotionRetargetingNetwork(nn.Module):
    """
    Encoder-Decoder network for motion retargeting following the paper's architecture.
    
    Architecture (based on Table 3 in paper):
    - Encoder: Conv + LReLU + AP (Average Skeletal Pooling)
    - Decoder: UP (Skeletal Unpooling) + UpS (Temporal Upsampling) + Conv + LReLU
    
    The network encodes motion to a shared latent space (primal skeleton) and
    decodes to the target skeleton structure.
    """
    def __init__(
        self,
        input_joints=15,
        output_joints=24,
        joint_mapping=None,  # mapping from input joint indices to output joint indices
        input_channels=3,  # joint positions (x, y, z)
        output_channels=3,
        hidden_channels=128,
        kernel_size=15,
    ):
        super().__init__()
        self.input_joints = input_joints
        self.output_joints = output_joints
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        
        # Default joint mapping: phys_humanoid_v3 -> SMPL
        if joint_mapping is None:
            # This maps phys_humanoid_v3 joints [0..14] to SMPL joint indices
            # Must match generate_motion.py exactly: [0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]
            self.joint_mapping = {
                i: smpl_idx for i, smpl_idx in enumerate(
                    [0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]
                )
            }
        else:
            self.joint_mapping = joint_mapping
        
        # Compute primal skeleton size (number of unique target joints in mapping)
        primal_joints = len(set(self.joint_mapping.values()))
        
        # Encoder: Input -> Primal Skeleton
        # EAQ: Encoder A to Q (primal skeleton)
        # First layer: Conv + LReLU + AP
        self.encoder_conv1 = SkeletonAwareConv1D(
            input_channels, hidden_channels // 2, kernel_size=kernel_size, stride=2
        )
        self.encoder_pool1 = AverageSkeletalPooling(self.joint_mapping)
        # After first pooling: input_joints -> primal_joints
        
        # Second layer: Conv + LReLU + AP (to smaller primal representation)
        self.encoder_conv2 = SkeletonAwareConv1D(
            hidden_channels // 2, hidden_channels, kernel_size=kernel_size, stride=2
        )
        # Create mapping from primal to even smaller representation
        # This creates a hierarchical reduction similar to the paper
        primal_to_smaller = {}
        # Reduce to ~7 joints (as in paper Table 3: 28->18->7)
        if primal_joints > 7:
            # Create a reduction mapping that groups nearby joints
            reduction_factor = max(1, primal_joints // 7)
            for i in range(primal_joints):
                primal_to_smaller[i] = min(i // reduction_factor, 6)  # Cap at 6 (0-6 = 7 joints)
        else:
            # Identity mapping if already small enough
            primal_to_smaller = {i: i for i in range(primal_joints)}
        self.encoder_pool2 = AverageSkeletalPooling(primal_to_smaller)
        
        # Decoder: Primal Skeleton -> Output
        # DAQ: Decoder Q to A
        # First decoder layer: UP + UpS + Conv + LReLU
        smaller_joints = len(set(primal_to_smaller.values()))
        # Create reverse mapping: smaller -> primal
        # Group all primal joints that map to the same smaller joint
        smaller_to_primal_groups = defaultdict(list)
        for primal_idx, smaller_idx in primal_to_smaller.items():
            smaller_to_primal_groups[smaller_idx].append(primal_idx)
        # For unpooling, we need a mapping from smaller index to one primal index
        # We'll use the first primal index in each group
        reverse_smaller_to_primal = {smaller_idx: primal_group[0] 
                                     for smaller_idx, primal_group in smaller_to_primal_groups.items()}
        self.decoder_unpool1 = SkeletalUnpooling(reverse_smaller_to_primal, primal_joints)
        self.decoder_upsample1 = TemporalLinearUpsampling(scale_factor=2)
        self.decoder_conv1 = SkeletonAwareConv1D(
            hidden_channels, hidden_channels // 2, kernel_size=kernel_size, stride=1
        )
        
        # Second decoder layer: UP + UpS + Conv
        # Map from primal skeleton (which contains SMPL joint indices) to all 24 SMPL joints
        # The primal skeleton joints are the unique SMPL joint indices from joint_mapping
        # We need to map from these indices to all 24 output joints
        primal_to_output = {}
        primal_joint_list = sorted(set(self.joint_mapping.values()))
        for i, smpl_joint_idx in enumerate(primal_joint_list):
            # Map from primal index i to SMPL joint index smpl_joint_idx
            primal_to_output[i] = smpl_joint_idx
        self.decoder_unpool2 = SkeletalUnpooling(primal_to_output, output_joints)
        self.decoder_upsample2 = TemporalLinearUpsampling(scale_factor=2)
        self.decoder_conv2 = SkeletonAwareConv1D(
            hidden_channels // 2, output_channels, kernel_size=kernel_size, stride=1
        )
        
        # Activation: LeakyReLU as per paper
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
    
    def forward(self, joint_positions):
        """
        Args:
            joint_positions: (B, T, J_input, C_input) input joint positions
        Returns:
            output_positions: (B, T', J_output, C_output) predicted output joint positions
                             (T' may differ from T due to upsampling)
        """
        x = joint_positions
        
        # Encoder: Input -> Primal Skeleton
        # Conv + LReLU + AP
        x = self.encoder_conv1(x)  # (B, T//2, J_input, hidden_channels//2)
        x = self.leaky_relu(x)
        x = self.encoder_pool1(x)  # (B, T//2, primal_joints, hidden_channels//2)
        
        x = self.encoder_conv2(x)  # (B, T//4, primal_joints, hidden_channels)
        x = self.leaky_relu(x)
        x = self.encoder_pool2(x)  # (B, T//4, smaller_joints, hidden_channels)
        
        # Decoder: Primal Skeleton -> Output
        # UP + UpS + Conv + LReLU
        x = self.decoder_unpool1(x)  # (B, T//4, primal_joints, hidden_channels)
        x = self.decoder_upsample1(x)  # (B, T//2, primal_joints, hidden_channels)
        x = self.decoder_conv1(x)  # (B, T//2, primal_joints, hidden_channels//2)
        x = self.leaky_relu(x)
        
        x = self.decoder_unpool2(x)  # (B, T//2, output_joints, hidden_channels//2)
        x = self.decoder_upsample2(x)  # (B, T, output_joints, hidden_channels//2)
        x = self.decoder_conv2(x)  # (B, T, output_joints, output_channels)
        
        return x


class SMPLJointPredictor(nn.Module):
    """
    High-level wrapper that predicts all 24 SMPL joint positions from 15 phys_humanoid_v3 joints.
    This can be used as an initialization or approximation before optimization.
    """
    def __init__(self, hidden_channels=128, kernel_size=15):
        super().__init__()
        self.retargeting_net = MotionRetargetingNetwork(
            input_joints=15,
            output_joints=24,
            input_channels=3,
            output_channels=3,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )
    
    def forward(self, phys_joint_positions):
        """
        Args:
            phys_joint_positions: (T, 15, 3) or (B, T, 15, 3) phys_humanoid_v3 joint positions
        Returns:
            smpl_joint_positions: (T, 24, 3) or (B, T, 24, 3) predicted SMPL joint positions
        """
        # Handle both batched and unbatched inputs
        if phys_joint_positions.dim() == 3:
            phys_joint_positions = phys_joint_positions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        smpl_joint_positions = self.retargeting_net(phys_joint_positions)
        
        if squeeze_output:
            smpl_joint_positions = smpl_joint_positions.squeeze(0)
        
        return smpl_joint_positions


def create_default_predictor(pretrained_path=None, device='cpu', hidden_channels=128, kernel_size=15):
    """
    Create a default SMPL joint predictor.
    If pretrained_path is provided, load weights from that path.
    Otherwise, return untrained model (can be used with optimization).
    
    Args:
        pretrained_path: Path to pretrained model weights
        device: Device to load model on
        hidden_channels: Number of hidden channels (default: 128)
        kernel_size: Temporal convolution kernel size (default: 15)
    """
    model = SMPLJointPredictor(hidden_channels=hidden_channels, kernel_size=kernel_size)
    
    if pretrained_path is not None and os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        model.eval()
    
    return model.to(device)


# For backward compatibility and easier usage
if __name__ == "__main__":
    # Test the network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Testing Skeleton-Aware Network (Improved Implementation)")
    print("=" * 60)
    
    model = SMPLJointPredictor(hidden_channels=128, kernel_size=15).to(device)
    
    # Test with dummy data
    T = 100
    test_input = torch.randn(T, 15, 3).to(device)
    
    print(f"\nInput shape: {test_input.shape}")
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"\n✓ Network test passed!")
    print(f"\nKey Improvements (based on paper):")
    print("  1. Skeleton-Aware Convolution with reflected padding")
    print("  2. Average Skeletal Pooling (AP) for primal skeleton reduction")
    print("  3. Skeletal Unpooling (UP) for skeleton expansion")
    print("  4. Temporal Linear Upsampling (UpS) for temporal resolution restoration")
    print("  5. Encoder-Decoder architecture following paper's Table 3")

