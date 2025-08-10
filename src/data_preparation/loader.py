import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class ExertionDataset(Dataset):
    """Dataset for exertion level prediction"""
    
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        
        # Load clip labels - 支持ConfigManager和普通配置对象
        if hasattr(config, 'get'):
            # ConfigManager对象
            label_file = config.get('data.label_file', 'data/processed/04_final/labels.csv')
            feature_dir = config.get('data.feature_dir', 'data/processed/04_final/features')
            self.mfcc_dim = config.get('model.mfcc_dim', 40)
            self.wav2vec2_dim = config.get('model.wav2vec2_dim', 768)
        else:
            # 普通配置对象
            label_file = getattr(config, 'final_labels_file', 'data/processed/04_final/labels.csv')
            feature_dir = getattr(config, 'final_features_dir', 'data/processed/04_final/features')
            self.mfcc_dim = getattr(config, 'mfcc_dim', 40)
            self.wav2vec2_dim = getattr(config, 'wav2vec2_dim', 768)
        
        clip_labels = pd.read_csv(label_file)
        self.labels_df = clip_labels[clip_labels['split'] == split].reset_index(drop=True)
        
        # Set feature directory
        self.feature_dir = os.path.join(feature_dir, split)
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        # Get clip information
        row = self.labels_df.iloc[idx]
        clip_id = row['clip_id']
        exertion_level = row['exertion_level']
        
        # Load features
        feature_path = os.path.join(self.feature_dir, f"{clip_id}.npz")
        features = np.load(feature_path)
        
        # Extract features separately
        mfcc_features = features['mfcc'] if 'mfcc' in features else np.zeros((300, self.mfcc_dim))
        wav2vec2_features = features['wav2vec2'] if 'wav2vec2' in features else np.zeros((300, self.wav2vec2_dim))
        
        # Transpose to (features, time) format for Conv1d
        mfcc_features = mfcc_features.T  # (40, 300)
        wav2vec2_features = wav2vec2_features.T  # (768, 300)
        
        # Add channel dimension for MFCC: (F, T) -> (1, F, T) for Conv2d
        mfcc_features = mfcc_features[np.newaxis, :, :]  # (1, 40, 300)
        
        # Transpose wav2vec2 for Transformer: (D, T) -> (T, D) for [B, T, D] format
        wav2vec2_features = wav2vec2_features.T  # (300, 768)
        
        # Convert to tensors
        mfcc_tensor = torch.FloatTensor(mfcc_features)  # (1, 40, 300)
        wav2vec2_tensor = torch.FloatTensor(wav2vec2_features)  # (300, 768)
        # Convert labels from 1-5 to 0-4 for PyTorch
        label_tensor = torch.LongTensor([exertion_level - 1])
        
        return mfcc_tensor, wav2vec2_tensor, label_tensor.squeeze()

def create_data_loaders(config, batch_size=32, num_workers=4):
    """Create data loaders for all splits"""
    
    # 从配置中获取批大小和工作进程数
    if hasattr(config, 'get'):
        # ConfigManager对象
        batch_size = config.get('training.batch_size', batch_size)
        num_workers = config.get('training.num_workers', num_workers)
        pin_memory = config.get('training.pin_memory', True)
    else:
        # 普通配置对象
        batch_size = getattr(config, 'batch_size', batch_size)
        num_workers = getattr(config, 'num_workers', num_workers)
        pin_memory = getattr(config, 'pin_memory', True)
    
    # Create datasets
    train_dataset = ExertionDataset(config, split='train')
    val_dataset = ExertionDataset(config, split='val')
    test_dataset = ExertionDataset(config, split='test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
