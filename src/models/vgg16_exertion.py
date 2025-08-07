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
                 mfcc_dim=13, 
                 wav2vec2_dim=768,
                 num_classes=5,
                 dropout_rate=0.5,
                 use_mfcc=True,
                 use_wav2vec2=True):
        super(VGG16ExertionModel, self).__init__()
        
        self.use_mfcc = use_mfcc
        self.use_wav2vec2 = use_wav2vec2
        
        # 计算输入维度
        input_dim = 0
        if use_mfcc:
            input_dim += mfcc_dim
        if use_wav2vec2:
            input_dim += wav2vec2_dim
            
        self.input_dim = input_dim
        
        # VGG16特征提取器
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            # Block 4
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            # Block 5
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
        )
        
        # 全局平均池化后的特征维度
        # 使用adaptive_avg_pool1d后，特征维度为512
        self.feature_dim = 512
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, mfcc=None, wav2vec2=None):
        """
        前向传播
        Args:
            mfcc: MFCC特征 (batch_size, 300, mfcc_dim)
            wav2vec2: wav2vec2特征 (batch_size, 300, wav2vec2_dim)
        """
        # 特征融合
        features_list = []
        
        if self.use_mfcc and mfcc is not None:
            features_list.append(mfcc)
        if self.use_wav2vec2 and wav2vec2 is not None:
            features_list.append(wav2vec2)
        
        if not features_list:
            raise ValueError("至少需要一种特征输入")
        
        # 横向拼接特征 (batch_size, 300, total_dim)
        # 每个特征都是 (batch_size, 300, feature_dim)
        # 在特征维度上拼接
        x = torch.cat(features_list, dim=2)
        
        # 转换为卷积输入格式 (batch_size, channels, time)
        x = x.transpose(1, 2)
        
        # VGG特征提取
        x = self.features(x)
        
        # 全局平均池化
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        
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
    
    def forward(self, mfcc=None, wav2vec2=None):
        """优化的前向传播"""
        if self.autocast_enabled and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                return super().forward(mfcc, wav2vec2)
        else:
            return super().forward(mfcc, wav2vec2)


def create_model(config):
    """
    根据配置创建模型
    Args:
        config: 配置字典，包含模型参数
    """
    model_config = {
        'mfcc_dim': config.get('mfcc_dim', 13),
        'wav2vec2_dim': config.get('wav2vec2_dim', 768),
        'num_classes': config.get('num_classes', 5),
        'dropout_rate': config.get('dropout_rate', 0.5),
        'use_mfcc': config.get('use_mfcc', True),
        'use_wav2vec2': config.get('use_wav2vec2', True),
    }
    
    if config.get('optimize_for_rtx4070', True):
        model = VGG16ExertionModelRTX4070(**model_config)
    else:
        model = VGG16ExertionModel(**model_config)
    
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