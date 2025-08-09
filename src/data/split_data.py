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
from tqdm import tqdm

def create_data_split():
    """
    Create training and test set splits
    Split from cleaned data level
    """
    
    print("Creating data split...")
    
    # Read cleaned metadata
    metadata_path = "data/metadata_cleaned.csv"
    if not os.path.exists(metadata_path):
        print(f"ERROR: Cleaned metadata file {metadata_path} does not exist")
        print("Please run data cleaning step first")
        return None, None
    
    print("Reading cleaned metadata...")
    df = pd.read_csv(metadata_path)
    
    # Extract Session Name and Exertion columns
    labels_df = df[['Session Name', 'Exertion']].copy()
    labels_df.columns = ['session_id', 'exertion_level']
    
    # Check exertion level distribution
    print("\nOriginal data exertion level distribution:")
    level_counts = labels_df['exertion_level'].value_counts().sort_index()
    for level, count in level_counts.items():
        print(f"Level {level}: {count} original audio files")
    
    # Stratified split by exertion level
    print("\nStarting stratified split...")
    
    # Split for each exertion level separately
    train_sessions = []
    test_sessions = []
    
    for level in sorted(labels_df['exertion_level'].unique()):
        level_data = labels_df[labels_df['exertion_level'] == level]
        level_sessions = level_data['session_id'].tolist()
        
        print(f"Level {level}: {len(level_sessions)} original audio files")
        
        # Ensure each label has at least one audio in test set
        if len(level_sessions) == 1:
            # If only one audio, put in training set
            level_train = level_sessions
            level_test = []
        else:
            # Calculate test set size, ensure at least 1, maximum 10%
            test_size = max(1, min(len(level_sessions) // 10, len(level_sessions) - 1))
            test_ratio = test_size / len(level_sessions)
            
            level_train, level_test = train_test_split(
                level_sessions, 
                test_size=test_ratio, 
                random_state=42,
                shuffle=True
            )
        
        train_sessions.extend(level_train)
        test_sessions.extend(level_test)
        
        print(f"  -> Training: {len(level_train)} files, Test: {len(level_test)} files")
    
    # Create split information
    split_info = {
        'train_sessions': train_sessions,
        'test_sessions': test_sessions,
        'total_train': len(train_sessions),
        'total_test': len(test_sessions),
        'split_ratio': len(test_sessions) / (len(train_sessions) + len(test_sessions))
    }
    
    # Save split information
    split_info_path = "data/split_info.json"
    with open(split_info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    print(f"\nSplit info saved to: {split_info_path}")
    print(f"Total training set: {len(train_sessions)} original audio files")
    print(f"Total test set: {len(test_sessions)} original audio files")
    print(f"Split ratio: {split_info['split_ratio']:.2%}")
    
    return train_sessions, test_sessions

def create_train_test_directories():
    """Create training and test set directory structure"""
    
    print("\nCreating directory structure...")
    
    # Create main directories
    train_dir = "data/train_set"
    test_dir = "data/test_set"
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['audio', 'features', 'labels']
    for subdir in subdirs:
        os.makedirs(os.path.join(train_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(test_dir, subdir), exist_ok=True)
    
    print(f"Created directory: {train_dir}")
    print(f"Created directory: {test_dir}")
    
    return train_dir, test_dir

def copy_audio_files(train_sessions, test_sessions, train_dir, test_dir):
    """Copy audio files to corresponding training and test set directories"""
    
    print("\nCopying audio files...")
    
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
        else:
            print(f"WARNING: Training audio file not found {audio_file}")
    
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
        else:
            print(f"WARNING: Test audio file not found {audio_file}")
    
    print(f"Copy completed: Training {train_count} audio files, Test {test_count} audio files")

def create_labels_files(train_sessions, test_sessions, train_dir, test_dir):
    """Create label files for training and test sets"""
    
    print("\nCreating label files...")
    
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
    
    print(f"Training labels: {len(train_labels)} samples -> {train_labels_path}")
    print(f"Test labels: {len(test_labels)} samples -> {test_labels_path}")
    
    # Show label distribution
    print("\nTraining set label distribution:")
    train_level_counts = train_labels['exertion_level'].value_counts().sort_index()
    for level, count in train_level_counts.items():
        print(f"  Level {level}: {count} files")
    
    print("\nTest set label distribution:")
    test_level_counts = test_labels['exertion_level'].value_counts().sort_index()
    for level, count in test_level_counts.items():
        print(f"  Level {level}: {count} files")

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
    
    print("\n=== Data Split Completed ===")
    print("Training directory: data/train_set/")
    print("Test directory: data/test_set/")

if __name__ == "__main__":
    main()
