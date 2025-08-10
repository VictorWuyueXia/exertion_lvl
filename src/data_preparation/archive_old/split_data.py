#!/usr/bin/env python3
"""
Data splitting script
Split training and test sets from cleaned data
"""

import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import yaml
from tqdm import tqdm

def load_config():
    """Load configuration from yaml file"""
    config_path = "config/vgg16_wav2vec2_layer4_mfcc_exertion.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    else:
        print(f"WARNING: Config file {config_path} not found, using default values")
        return {
            'data': {
                'split': {
                    'test_size': 0.1,
                    'val_size': 0.1,
                    'random_state': 42
                }
            }
        }

def create_data_split():
    """
    Create training and test set splits
    Split from cleaned data level
    """
    
    print("Creating data split...")
    
    # Load configuration
    config = load_config()
    test_size = config['data']['split']['test_size']
    random_state = config['data']['split']['random_state']
    
    print(f"Using test_size: {test_size} from config")
    
    # Read cleaned metadata
    metadata_path = "data/metadata_cleaned.csv"
    if not os.path.exists(metadata_path):
        print(f"ERROR: Cleaned metadata file {metadata_path} does not exist")
        return None, None
    
    df = pd.read_csv(metadata_path)
    
    # Extract Session Name and Exertion columns
    labels_df = df[['Session Name', 'Exertion']].copy()
    labels_df.columns = ['session_id', 'exertion_level']
    
    # Stratified split by exertion level
    print("Starting stratified split...")
    
    # Split for each exertion level separately
    train_sessions = []
    test_sessions = []
    
    unique_levels = sorted(labels_df['exertion_level'].unique())
    for level in tqdm(unique_levels, desc="Splitting by exertion level"):
        level_data = labels_df[labels_df['exertion_level'] == level]
        level_sessions = level_data['session_id'].tolist()
        
        # Ensure each label has at least one session in test set
        if len(level_sessions) == 1:
            # If only one session, put in training set
            level_train = level_sessions
            level_test = []
        elif len(level_sessions) == 2:
            # If two sessions, put one in each set
            level_train = [level_sessions[0]]
            level_test = [level_sessions[1]]
        else:
            # Calculate test set size, ensure at least 1, target from config
            min_test = 1
            target_test_size = max(min_test, int(len(level_sessions) * test_size))
            
            # Ensure we don't take too many for test set
            max_test_size = len(level_sessions) - 1
            test_size_actual = min(target_test_size, max_test_size)
            
            level_train, level_test = train_test_split(
                level_sessions, 
                test_size=test_size_actual, 
                random_state=random_state,
                shuffle=True
            )
        
        train_sessions.extend(level_train)
        test_sessions.extend(level_test)
    
    # Create split information
    split_info = {
        'train_sessions': train_sessions,
        'test_sessions': test_sessions,
        'total_train': len(train_sessions),
        'total_test': len(test_sessions),
        'split_ratio': len(test_sessions) / (len(train_sessions) + len(test_sessions)),
        'config_test_size': test_size
    }
    
    # Save split information
    split_info_path = "data/split_info.json"
    with open(split_info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    print(f"Split completed: {len(train_sessions)} train, {len(test_sessions)} test ({split_info['split_ratio']:.1%})")
    
    return train_sessions, test_sessions

def create_train_test_directories():
    """Create training and test set directory structure"""
    
    # Create main directories
    train_dir = "data/train_set"
    test_dir = "data/test_set"
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['audio', 'features', 'labels', 'segmented_audio']
    for subdir in subdirs:
        os.makedirs(os.path.join(train_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(test_dir, subdir), exist_ok=True)
    
    return train_dir, test_dir

def copy_audio_files(train_sessions, test_sessions, train_dir, test_dir):
    """Copy audio files to corresponding training and test set directories"""
    
    cleaned_audio_dir = "data/cleaned_audio"
    if not os.path.exists(cleaned_audio_dir):
        print(f"ERROR: Cleaned audio directory {cleaned_audio_dir} does not exist")
        return
    
    # Copy training set audio
    train_audio_dir = os.path.join(train_dir, "audio")
    train_count = 0
    for session_id in tqdm(train_sessions, desc="Copying training audio"):
        audio_file = f"{session_id}.wav"
        src_path = os.path.join(cleaned_audio_dir, audio_file)
        dst_path = os.path.join(train_audio_dir, audio_file)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            train_count += 1
    
    # Copy test set audio
    test_audio_dir = os.path.join(test_dir, "audio")
    test_count = 0
    for session_id in tqdm(test_sessions, desc="Copying test audio"):
        audio_file = f"{session_id}.wav"
        src_path = os.path.join(cleaned_audio_dir, audio_file)
        dst_path = os.path.join(test_audio_dir, audio_file)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            test_count += 1
    
    print(f"Audio files copied: {train_count} train, {test_count} test")

def create_labels_files(train_sessions, test_sessions, train_dir, test_dir):
    """Create label files for training and test sets"""
    
    # Extract label information from cleaned metadata
    metadata_path = "data/metadata_cleaned.csv"
    if not os.path.exists(metadata_path):
        print(f"ERROR: Cleaned metadata file {metadata_path} does not exist")
        return
    
    # Read cleaned metadata
    df = pd.read_csv(metadata_path)
    
    # Extract Session Name and Exertion columns as labels
    labels_df = df[['Session Name', 'Exertion']].copy()
    labels_df.columns = ['session_id', 'exertion_level']
    
    # Create training set labels
    train_labels = labels_df[labels_df['session_id'].isin(train_sessions)].copy()
    train_labels_path = os.path.join(train_dir, "labels", "exertion_labels.csv")
    train_labels.to_csv(train_labels_path, index=False)
    
    # Create test set labels
    test_labels = labels_df[labels_df['session_id'].isin(test_sessions)].copy()
    test_labels_path = os.path.join(test_dir, "labels", "exertion_labels.csv")
    test_labels.to_csv(test_labels_path, index=False)
    
    print(f"Label files created: {len(train_labels)} train, {len(test_labels)} test")

def main():
    """Main function"""
    print("=== Data Splitting ===")
    
    # 1. Create data split
    train_sessions, test_sessions = create_data_split()
    if train_sessions is None:
        return
    
    # 2. Create directory structure
    train_dir, test_dir = create_train_test_directories()
    
    # 3. Copy audio files
    copy_audio_files(train_sessions, test_sessions, train_dir, test_dir)
    
    # 4. Create label files
    create_labels_files(train_sessions, test_sessions, train_dir, test_dir)
    
    print("Data split completed")

if __name__ == "__main__":
    main()
