#!/usr/bin/env python3
"""
Data loader for exertion level detection
Handles training and test set separation without data leakage
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json

class NoLeakageAudioDataset(Dataset):
    """
    Audio dataset without data leakage
    Supports MFCC and wav2vec2 features
    """
    
    def __init__(self, 
                 audio_dir=None, 
                 labels_file=None, 
                 feature_dir=None,
                 use_mfcc=True, 
                 use_wav2vec2=True,
                 wav2vec2_layers=[4, 8, 12],
                 target_frames=300,
                 transform=None):
        """
        Initialize dataset
        
        Args:
            audio_dir: Audio file directory
            labels_file: Label file path
            feature_dir: Feature file directory (if None, extract features in real-time)
            use_mfcc: Whether to use MFCC features
            use_wav2vec2: Whether to use wav2vec2 features
            wav2vec2_layers: wav2vec2 layer selection
            target_frames: Target frame count
            transform: Data transformation
        """
        self.audio_dir = audio_dir
        self.labels_file = labels_file
        self.feature_dir = feature_dir
        self.use_mfcc = use_mfcc
        self.use_wav2vec2 = use_wav2vec2
        self.wav2vec2_layers = wav2vec2_layers
        self.target_frames = target_frames
        self.transform = transform
        
        # Check data format
        if feature_dir and labels_file:
            # Use pre-extracted feature format (.npz files)
            self._load_npz_features()
        elif audio_dir and labels_file:
            # Use original audio format
            self._load_audio_files()
        else:
            raise ValueError("Must provide feature_dir+labels_file or audio_dir+labels_file")
    
    def _load_npz_features(self):
        """Load .npz format pre-extracted features"""
        # Load labels
        self.labels_df = pd.read_csv(self.labels_file)
        
        # Get feature file list
        self.feature_files = []
        for _, row in self.labels_df.iterrows():
            segment_id = row['segment_id']
            feature_file = os.path.join(self.feature_dir, f"{segment_id}.npz")
            
            if os.path.exists(feature_file):
                self.feature_files.append({
                    'segment_id': segment_id,
                    'feature_file': feature_file,
                    'exertion_level': row['exertion_level']
                })
        
        print(f"Dataset loaded: {len(self.feature_files)} feature files")
        
        # Show label distribution
        level_counts = self.labels_df['exertion_level'].value_counts().sort_index()
        print("Label distribution:")
        for level, count in level_counts.items():
            print(f"  Level {level}: {count} samples")
        
        # Check class imbalance
        min_samples = min(level_counts.values)
        max_samples = max(level_counts.values)
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        if imbalance_ratio > 10:
            print(f"\nWARNING: Severe class imbalance detected!")
            print(f"  Ratio between most and least frequent classes: {imbalance_ratio:.1f}")
            print(f"  Level {level_counts.idxmin()} has only {min_samples} samples")
            print(f"  Consider using weighted loss or data augmentation")
    
    def _load_audio_files(self):
        """Load original audio files"""
        # Load labels
        self.labels_df = pd.read_csv(self.labels_file)
        
        # Get audio file list
        self.audio_files = []
        for _, row in self.labels_df.iterrows():
            segment_id = row['segment_id']
            audio_file = os.path.join(self.audio_dir, f"{segment_id}.wav")
            
            if os.path.exists(audio_file):
                self.audio_files.append({
                    'segment_id': segment_id,
                    'audio_file': audio_file,
                    'exertion_level': row['exertion_level']
                })
        
        print(f"Dataset loaded: {len(self.audio_files)} audio files")
        
        # Show label distribution
        level_counts = self.labels_df['exertion_level'].value_counts().sort_index()
        print("Label distribution:")
        for level, count in level_counts.items():
            print(f"  Level {level}: {count} samples")
    
    def __len__(self):
        if hasattr(self, 'feature_files'):
            return len(self.feature_files)
        else:
            return len(self.audio_files)
    
    def __getitem__(self, idx):
        if hasattr(self, 'feature_files'):
            return self._get_npz_item(idx)
        else:
            return self._get_audio_item(idx)
    
    def _get_npz_item(self, idx):
        """Get item from .npz feature file"""
        item = self.feature_files[idx]
        segment_id = item['segment_id']
        feature_file = item['feature_file']
        exertion_level = item['exertion_level']
        
        # Load features
        features = np.load(feature_file)
        mfcc = features['mfcc'].astype(np.float32)
        wav2vec2 = features['wav2vec2'].astype(np.float32)
        
        # Convert to tensor
        mfcc = torch.tensor(mfcc)
        wav2vec2 = torch.tensor(wav2vec2)
        # Convert labels from 1-5 to 0-4 for PyTorch
        label = torch.tensor(exertion_level - 1, dtype=torch.long)
        
        return {
            'mfcc': mfcc,
            'wav2vec2': wav2vec2,
            'label': label,
            'segment_id': segment_id
        }
    
    def _get_audio_item(self, idx):
        """Get item from original audio file"""
        item = self.audio_files[idx]
        segment_id = item['segment_id']
        audio_file = item['audio_file']
        exertion_level = item['exertion_level']
        
        # Extract features in real-time
        mfcc, wav2vec2 = self._extract_features_realtime(audio_file)
        
        # Convert to tensor
        mfcc = torch.tensor(mfcc, dtype=torch.float32)
        wav2vec2 = torch.tensor(wav2vec2, dtype=torch.float32)
        # Convert labels from 1-5 to 0-4 for PyTorch
        label = torch.tensor(exertion_level - 1, dtype=torch.long)
        
        return {
            'mfcc': mfcc,
            'wav2vec2': wav2vec2,
            'label': label,
            'segment_id': segment_id
        }
    
    def _load_precomputed_features(self, session_id):
        """Load precomputed features for a session"""
        feature_file = os.path.join(self.feature_dir, f"{session_id}.npz")
        
        if os.path.exists(feature_file):
            features = np.load(feature_file)
            return features
        else:
            return None
    
    def _extract_features_realtime(self, audio_path):
        """Extract features from audio file in real-time"""
        # This would implement real-time feature extraction
        # For now, return dummy features
        mfcc = np.random.randn(13, self.target_frames).astype(np.float32)
        wav2vec2 = np.random.randn(768, self.target_frames).astype(np.float32)
        return mfcc, wav2vec2

def create_train_val_split(train_dataset, val_size=0.1, random_state=42):
    """
    Create train/validation split
    
    Args:
        train_dataset: Training dataset
        val_size: Validation set size ratio
        random_state: Random seed
    
    Returns:
        train_dataset, val_dataset: Split datasets
    """
    # Get all indices
    indices = list(range(len(train_dataset)))
    
    # Split indices
    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, random_state=random_state, stratify=None
    )
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    return train_subset, val_subset

class FeatureNormalizer:
    """Feature normalization utility"""
    
    def __init__(self):
        self.mfcc_scaler = StandardScaler()
        self.wav2vec2_scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, train_dataset):
        """Fit normalizers on training data"""
        print("Fitting feature normalizers...")
        
        mfcc_features = []
        wav2vec2_features = []
        
        # Collect features from training dataset
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            
            if 'mfcc' in item and item['mfcc'] is not None:
                mfcc = item['mfcc'].numpy()
                if len(mfcc.shape) == 3:
                    mfcc = mfcc.reshape(-1, mfcc.shape[-1])
                mfcc_features.append(mfcc)
            
            if 'wav2vec2' in item and item['wav2vec2'] is not None:
                wav2vec2 = item['wav2vec2'].numpy()
                if len(wav2vec2.shape) == 3:
                    wav2vec2 = wav2vec2.reshape(-1, wav2vec2.shape[-1])
                wav2vec2_features.append(wav2vec2)
        
        # Fit scalers
        if mfcc_features:
            mfcc_combined = np.vstack(mfcc_features)
            self.mfcc_scaler.fit(mfcc_combined)
            print(f"MFCC scaler fitted on {mfcc_combined.shape[0]} samples")
        
        if wav2vec2_features:
            wav2vec2_combined = np.vstack(wav2vec2_features)
            self.wav2vec2_scaler.fit(wav2vec2_combined)
            print(f"wav2vec2 scaler fitted on {wav2vec2_combined.shape[0]} samples")
        
        self.fitted = True
    
    def transform(self, mfcc=None, wav2vec2=None):
        """Transform features using fitted scalers"""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transformation")
        
        if mfcc is not None:
            mfcc_shape = mfcc.shape
            mfcc_reshaped = mfcc.reshape(-1, mfcc.shape[-1])
            mfcc_normalized = self.mfcc_scaler.transform(mfcc_reshaped)
            mfcc = mfcc_normalized.reshape(mfcc_shape)
        
        if wav2vec2 is not None:
            wav2vec2_shape = wav2vec2.shape
            wav2vec2_reshaped = wav2vec2.reshape(-1, wav2vec2.shape[-1])
            wav2vec2_normalized = self.wav2vec2_scaler.transform(wav2vec2_reshaped)
            wav2vec2 = wav2vec2_normalized.reshape(wav2vec2_shape)
        
        return mfcc, wav2vec2
    
    def save(self, filepath):
        """Save fitted normalizers"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'mfcc_scaler': self.mfcc_scaler,
                'wav2vec2_scaler': self.wav2vec2_scaler,
                'fitted': self.fitted
            }, f)
    
    def load(self, filepath):
        """Load fitted normalizers"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.mfcc_scaler = data['mfcc_scaler']
            self.wav2vec2_scaler = data['wav2vec2_scaler']
            self.fitted = data['fitted']

def collate_fn(batch):
    """
    Custom collate function for batching
    
    Args:
        batch: List of dataset items
    
    Returns:
        dict: Batched data
    """
    # Separate features and labels
    mfcc_list = []
    wav2vec2_list = []
    labels_list = []
    segment_ids = []
    
    for item in batch:
        if 'mfcc' in item and item['mfcc'] is not None:
            mfcc_list.append(item['mfcc'])
        if 'wav2vec2' in item and item['wav2vec2'] is not None:
            wav2vec2_list.append(item['wav2vec2'])
        if 'label' in item:
            labels_list.append(item['label'])
        if 'segment_id' in item:
            segment_ids.append(item['segment_id'])
    
    # Stack features
    mfcc = torch.stack(mfcc_list) if mfcc_list else None
    wav2vec2 = torch.stack(wav2vec2_list) if wav2vec2_list else None
    labels = torch.stack(labels_list) if labels_list else None
    
    return {
        'mfcc': mfcc,
        'wav2vec2': wav2vec2,
        'labels': labels,
        'segment_ids': segment_ids
    }

def create_dataloaders(config):
    """
    Create data loaders from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        train_loader, val_loader: Data loaders
    """
    # Extract configuration
    feature_dir = config.get('feature_dir', 'data/features')
    labels_file = config.get('labels_file', 'data/labels.csv')
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    val_size = config.get('val_size', 0.1)
    random_state = config.get('random_state', 42)
    
    # Create dataset
    dataset = NoLeakageAudioDataset(
        feature_dir=feature_dir,
        labels_file=labels_file,
        use_mfcc=config.get('use_mfcc', True),
        use_wav2vec2=config.get('use_wav2vec2', True),
        wav2vec2_layers=config.get('wav2vec2_layers', [4])
    )
    
    # Create train/validation split
    train_dataset, val_dataset = create_train_val_split(
        dataset, val_size=val_size, random_state=random_state
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
