import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class DataNormalizer:
    """Data normalization module"""
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
    
    def fit_scalers(self):
        """Fit scalers on training data"""
        print("=== Data Normalization ===")
        print("Fitting scalers on training data...")
        
        # Load clip labels
        clip_labels = pd.read_csv(self.config.final_labels_file)
        train_clips = clip_labels[clip_labels['split'] == 'train']['clip_id'].tolist()
        
        # Initialize scalers
        if self.config.use_mfcc:
            self.scalers['mfcc'] = StandardScaler()
        if self.config.use_wav2vec2:
            self.scalers['wav2vec2'] = StandardScaler()
        
        # Collect features in batches to fit scalers
        batch_size = 50
        for i in tqdm(range(0, len(train_clips), batch_size), desc="Collecting training features"):
            batch_clips = train_clips[i:i+batch_size]
            batch_mfcc_array = []
            batch_wav2vec2_array = []
            
            for clip_id in batch_clips:
                clip_path = os.path.join(self.config.features_dir, "train", f"{clip_id}.npy")
                if os.path.exists(clip_path):
                    clip_data = np.load(clip_path, allow_pickle=True).item()
                    
                    if self.config.use_mfcc and 'mfcc' in clip_data:
                        batch_mfcc_array.append(clip_data['mfcc'])
                    
                    if self.config.use_wav2vec2 and 'wav2vec2' in clip_data:
                        batch_wav2vec2_array.append(clip_data['wav2vec2'])
            
            # Fit scalers on batch
            if batch_mfcc_array:
                batch_mfcc = np.vstack(batch_mfcc_array)
                self.scalers['mfcc'].partial_fit(batch_mfcc)
            
            if batch_wav2vec2_array:
                batch_wav2vec2 = np.vstack(batch_wav2vec2_array)
                self.scalers['wav2vec2'].partial_fit(batch_wav2vec2)
        
        # Save scalers
        with open(self.config.scalers_file, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        print("Scalers fitted successfully!")
        print(f"Scalers saved to: {self.config.scalers_file}")
    
    def normalize_features(self):
        """Normalize all features using fitted scalers"""
        print("Normalizing features...")
        
        # Load clip labels
        clip_labels = pd.read_csv(self.config.final_labels_file)
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.config.final_features_dir, split)
            os.makedirs(split_dir, exist_ok=True)
        
        # Normalize each split
        for split in ['train', 'val', 'test']:
            print(f"Normalizing {split} set features...")
            split_clips = clip_labels[clip_labels['split'] == split]
            
            for _, row in tqdm(split_clips.iterrows(), total=len(split_clips), desc=f"Normalizing {split} features"):
                clip_id = row['clip_id']
                
                # Load original features
                input_path = os.path.join(self.config.features_dir, split, f"{clip_id}.npy")
                output_path = os.path.join(self.config.final_features_dir, split, f"{clip_id}.npz")
                
                if os.path.exists(input_path):
                    clip_data = np.load(input_path, allow_pickle=True).item()
                    normalized_features = {}
                    
                    # Normalize MFCC features
                    if self.config.use_mfcc and 'mfcc' in clip_data:
                        normalized_features['mfcc'] = self.scalers['mfcc'].transform(clip_data['mfcc'])
                    
                    # Normalize wav2vec2 features
                    if self.config.use_wav2vec2 and 'wav2vec2' in clip_data:
                        normalized_features['wav2vec2'] = self.scalers['wav2vec2'].transform(clip_data['wav2vec2'])
                    
                    # Save normalized features
                    np.savez(output_path, **normalized_features)
        
        print("Feature normalization completed!")
        print(f"Normalized features saved to: {self.config.final_features_dir}")
    
    def normalize_all(self):
        """Run complete normalization process"""
        self.fit_scalers()
        self.normalize_features()
