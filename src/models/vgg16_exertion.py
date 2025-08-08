import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VGG16ExertionModel(nn.Module):
    """
    基于VGG16结构的运动强度检测模型
    支持MFB、MFCC和wav2vec2特征的融合
    """
    
    def __init__(self, 
                 mfcc_dim=40,  # 修正为实际的MFCC维度
                 wav2vec2_dim=768,
                 mfb_dim=40,   # MFB特征维度
                 num_classes=5,
                 dropout_rate=0.5,
                 use_mfcc=True,
                 use_wav2vec2=True,
                 use_mfb=False):
        super(VGG16ExertionModel, self).__init__()
        
        self.use_mfcc = use_mfcc
        self.use_wav2vec2 = use_wav2vec2
        self.use_mfb = use_mfb
        self.num_classes = num_classes
        
        # 计算输入维度
        input_dim = 0
        if use_mfcc:
            input_dim += mfcc_dim
        if use_wav2vec2:
            input_dim += wav2vec2_dim
        if use_mfb:
            input_dim += mfb_dim
            
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        
        # VGG16特征提取器（修正为1D卷积）
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(self.input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 全局平均池化后的特征维度
        # 使用adaptive_avg_pool1d后，特征维度为512
        self.feature_dim = 512
        
        # 分类器（调整维度）
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # 使用PyTorch默认的Kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 使用PyTorch默认的Xavier初始化
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # 回归层特殊初始化：权重标准差0.01，偏置1.5
                    if m.out_features == self.num_classes:
                        nn.init.normal_(m.weight, mean=0, std=0.01)
                        nn.init.constant_(m.bias, 1.5)
                    else:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, mfcc=None, wav2vec2=None, mfb=None):
        """
        前向传播
        Args:
            mfcc: MFCC特征 (batch_size, time_steps, mfcc_dim)
            wav2vec2: wav2vec2特征 (batch_size, time_steps, wav2vec2_dim)
            mfb: MFB特征 (batch_size, time_steps, mfb_dim)
        """
        # 特征融合
        features_list = []
        current_input_dim = 0
        
        if self.use_mfcc and mfcc is not None:
            features_list.append(mfcc)
            current_input_dim += mfcc.shape[-1]
        if self.use_wav2vec2 and wav2vec2 is not None:
            features_list.append(wav2vec2)
            current_input_dim += wav2vec2.shape[-1]
        if self.use_mfb and mfb is not None:
            features_list.append(mfb)
            current_input_dim += mfb.shape[-1]
        
        if not features_list:
            raise ValueError("至少需要一种特征输入")
        
        # 检查输入维度是否匹配
        if current_input_dim != self.input_dim:
            raise ValueError(f"输入维度不匹配: 期望 {self.input_dim}, 实际 {current_input_dim}")
        
        # 检查时间步长是否一致
        if len(features_list) > 1:
            time_steps = [f.shape[1] for f in features_list]
            if len(set(time_steps)) > 1:
                # 如果时间步长不同，需要插值到相同长度
                min_time_steps = min(time_steps)
                for i in range(len(features_list)):
                    if features_list[i].shape[1] != min_time_steps:
                        # 使用插值调整时间步长
                        features_list[i] = F.interpolate(
                            features_list[i].transpose(1, 2), 
                            size=min_time_steps, 
                            mode='linear', 
                            align_corners=False
                        ).transpose(1, 2)
        
        # 在特征维度上拼接 (batch_size, time_steps, total_dim)
        x = torch.cat(features_list, dim=2)
        
        # 转换为卷积输入格式 (batch_size, channels, time)
        x = x.transpose(1, 2)
        
        # VGG特征提取
        x = self.features(x)
        
        # 全局平均池化 (batch_size, 512, time) -> (batch_size, 512)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        
        # 分类
        x = self.classifier(x)
        
        return x
    
    def get_feature_dim(self):
        """获取特征维度"""
        return self.feature_dim


class VGG16ExertionModelRTX4070(VGG16ExertionModel):
    """
    针对RTX4070优化的VGG16模型
    - 使用混合精度训练
    - 优化内存使用
    - 针对12GB显存优化
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 启用自动混合精度
        self.autocast_enabled = True
        
        # 优化内存使用
        self.gradient_checkpointing = False  # 可以根据需要启用
    
    def forward(self, mfcc=None, wav2vec2=None, mfb=None):
        """优化的前向传播"""
        if self.autocast_enabled and torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                return super().forward(mfcc, wav2vec2, mfb)
        else:
            return super().forward(mfcc, wav2vec2, mfb)


def create_model(config):
    """
    根据配置创建模型
    Args:
        config: 配置字典，包含模型参数
    """
    model_config = {
        'mfcc_dim': config.get('mfcc_dim', 40),  # 实际MFCC维度
        'wav2vec2_dim': config.get('wav2vec2_dim', 768),
        'mfb_dim': config.get('mfb_dim', 40),    # MFB特征维度
        'num_classes': config.get('num_classes', 5),  # 5个类别（0,1,2,3,4）
        'dropout_rate': config.get('dropout_rate', 0.5),
        'use_mfcc': config.get('use_mfcc', True),
        'use_wav2vec2': config.get('use_wav2vec2', True),
        'use_mfb': config.get('use_mfb', False),
    }
    
    if config.get('optimize_for_rtx4070', True):
        model = VGG16ExertionModelRTX4070(**model_config)
    else:
        model = VGG16ExertionModel(**model_config)
    
    # 启用torch.compile优化（如果配置中启用）
    if config.get('enable_compile', False) and hasattr(torch, 'compile'):
        try:
            print("启用torch.compile优化")
            model = torch.compile(model, mode='max-autotune')
        except Exception as e:
            print(f"torch.compile启用失败: {e}")
    
    return model


def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_millions': total_params / 1e6,
        'trainable_params_millions': trainable_params / 1e6
    } 