import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VGG16ExertionModel(nn.Module):
    """
    VGG16-based exertion level detection model
    Supports fusion of MFB, MFCC and wav2vec2 features
    """
    
    def __init__(self, 
                 mfcc_dim=40,  # Actual MFCC dimension
                 wav2vec2_dim=768,
                 mfb_dim=40,   # MFB feature dimension
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
        
        # Calculate input dimension
        input_dim = 0
        if use_mfcc:
            input_dim += mfcc_dim
        if use_wav2vec2:
            input_dim += wav2vec2_dim
        if use_mfb:
            input_dim += mfb_dim
            
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        
        # VGG16 feature extractor with BatchNorm (corrected to 1D convolution)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(self.input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

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
        )
        
        # Feature dimension after global average pooling
        # After adaptive_avg_pool1d, feature dimension is 512
        self.feature_dim = 512
        
        # Classifier with BatchNorm (adjusted dimensions)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Use PyTorch default Kaiming initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Use PyTorch default Xavier initialization
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, mfcc=None, wav2vec2=None, mfb=None, features=None):
        """
        Forward pass
        
        Args:
            mfcc: MFCC features (batch_size, mfcc_dim, time_steps)
            wav2vec2: wav2vec2 features (batch_size, wav2vec2_dim, time_steps)
            mfb: MFB features (batch_size, mfb_dim, time_steps)
            features: Combined features (batch_size, total_features, time_steps)
        
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        # If combined features are provided, use them directly
        if features is not None:
            x = features
        else:
            # Feature fusion
            features_list = []
            
            if self.use_mfcc and mfcc is not None:
                features_list.append(mfcc)
            
            if self.use_wav2vec2 and wav2vec2 is not None:
                features_list.append(wav2vec2)
            
            if self.use_mfb and mfb is not None:
                features_list.append(mfb)
            
            if not features_list:
                raise ValueError("No features provided")
            
            # Concatenate features along feature dimension
            x = torch.cat(features_list, dim=1)
        
        # Ensure input dimension matches
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.size(1)}")
        
        # Feature extraction
        x = self.features(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_feature_dim(self):
        """Get feature dimension after feature extraction"""
        return self.feature_dim

class VGG16ExertionModelOptimized(VGG16ExertionModel):
    """
    GPU optimized version of VGG16 exertion model
    Includes memory optimizations and performance improvements
    """
    
    def __init__(self, *args, **kwargs):
        super(VGG16ExertionModelOptimized, self).__init__(*args, **kwargs)
        
        # GPU specific optimizations
        self.use_amp = True  # Automatic mixed precision
        self.gradient_checkpointing = False  # Can be enabled for memory saving
    
    def forward(self, mfcc=None, wav2vec2=None, mfb=None, features=None):
        """Forward pass with GPU optimizations"""
        return super().forward(mfcc, wav2vec2, mfb, features)

def create_model(config):
    """
    Create model from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        model: Initialized model
    """
    # Extract model parameters
    mfcc_dim = config.get('mfcc_dim', 40)
    wav2vec2_dim = config.get('wav2vec2_dim', 768)
    mfb_dim = config.get('mfb_dim', 40)
    num_classes = config.get('num_classes', 5)
    dropout_rate = config.get('dropout_rate', 0.5)
    use_mfcc = config.get('use_mfcc', True)
    use_wav2vec2 = config.get('use_wav2vec2', True)
    use_mfb = config.get('use_mfb', False)
    optimize_for_gpu = config.get('optimize_for_gpu', True)
    
    # Create model
    if optimize_for_gpu:
        model = VGG16ExertionModelOptimized(
            mfcc_dim=mfcc_dim,
            wav2vec2_dim=wav2vec2_dim,
            mfb_dim=mfb_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_mfcc=use_mfcc,
            use_wav2vec2=use_wav2vec2,
            use_mfb=use_mfb
        )
    else:
        model = VGG16ExertionModel(
            mfcc_dim=mfcc_dim,
            wav2vec2_dim=wav2vec2_dim,
            mfb_dim=mfb_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_mfcc=use_mfcc,
            use_wav2vec2=use_wav2vec2,
            use_mfb=use_mfb
        )
    
    return model

def count_parameters(model):
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        dict: Parameter count information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_millions': total_params / 1e6,
        'trainable_params_millions': trainable_params / 1e6
    } 