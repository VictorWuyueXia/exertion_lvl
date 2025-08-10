#!/usr/bin/env python3
"""
Feature extraction script
Extract features for training and test sets
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import pandas as pd
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import librosa
import soundfile as sf
import warnings

# Suppress warnings to avoid interrupting progress bars
warnings.filterwarnings("ignore", category=UserWarning)

def extract_mfcc_features(audio_path, sr=16000, n_mfcc=20, hop_length=320):
    """Extract MFCC features from audio file"""
    # Load audio
    y, sr_orig = sf.read(audio_path)
    if sr_orig != sr:
        y = torchaudio.functional.resample(torch.tensor(y), orig_freq=sr_orig, new_freq=sr).numpy()
    
    # Extract MFCC with optimized parameters
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=1024)
    return mfcc.T  # Transpose to get (time, features)

def extract_wav2vec2_features(audio_path, model, feature_extractor, target_frames=300):
    """Extract wav2vec2 features from audio file"""
    # Load audio
    y, sr_orig = sf.read(audio_path)
    if sr_orig != 16000:
        y = torchaudio.functional.resample(torch.tensor(y), orig_freq=sr_orig, new_freq=16000).numpy()
    
    # Prepare input for wav2vec2
    inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
    
    # Move inputs to the same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Use layer 4 features (index 4)
        features = outputs.hidden_states[4].squeeze(0)  # Remove batch dimension
    
    # Resample to target frames if necessary
    if features.shape[0] != target_frames:
        features = torch.nn.functional.interpolate(
            features.unsqueeze(0).transpose(1, 2), 
            size=target_frames, 
            mode='linear'
        ).squeeze(0).transpose(0, 1)
    
    # Move to CPU and convert to numpy
    return features.cpu().numpy()

def extract_features_for_dataset(audio_dir, output_dir, labels_file, description, batch_size=32):
    """Extract features for specified dataset"""
    
    print(f"Starting {description} feature extraction...")
    
    # Check input directories
    if not os.path.exists(audio_dir):
        print(f"ERROR: Audio directory {audio_dir} does not exist")
        return False
    
    if not os.path.exists(labels_file):
        print(f"ERROR: Label file {labels_file} does not exist")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get audio file list
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    if not audio_files:
        print(f"{description} audio directory is empty")
        return False
    
    print(f"Found {len(audio_files)} audio files")
    
    # Load wav2vec2 model (load only once)
    print("Loading wav2vec2 model...")
    model_name = "facebook/wav2vec2-base"
    model = Wav2Vec2Model.from_pretrained(model_name)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU for feature extraction")
    else:
        print("Using CPU for feature extraction")
    
    # Load labels
    labels_df = pd.read_csv(labels_file)
    
    # Extract features
    extracted_count = 0
    for audio_file in tqdm(audio_files, desc=f"Extracting {description} features"):
        segment_id = audio_file.replace('.wav', '')
        audio_path = os.path.join(audio_dir, audio_file)
        
        # Check if label exists
        if segment_id not in labels_df['segment_id'].values:
            continue
        
        try:
            # Extract MFCC features
            mfcc_features = extract_mfcc_features(audio_path)
            
            # Extract wav2vec2 features
            wav2vec2_features = extract_wav2vec2_features(audio_path, model, feature_extractor)
            
            # Save features
            feature_file = os.path.join(output_dir, f"{segment_id}.npz")
            np.savez_compressed(
                feature_file,
                mfcc=mfcc_features,
                wav2vec2=wav2vec2_features
            )
            
            extracted_count += 1
            
        except Exception as e:
            continue

    print(f"{description} feature extraction completed!")
    print(f"Successfully extracted features for {extracted_count} files")
    print(f"Features saved to: {output_dir}")
    
    return True

def main():
    """Main function"""
    print("=== Feature Extraction ===")
    
    # Training set feature extraction
    train_success = extract_features_for_dataset(
        audio_dir="data/train_set/segmented_audio",
        output_dir="data/train_set/features",
        labels_file="data/train_set/segment_exertion_labels.csv",
        description="Training Set"
    )
    
    # Test set feature extraction
    test_success = extract_features_for_dataset(
        audio_dir="data/test_set/segmented_audio",
        output_dir="data/test_set/features",
        labels_file="data/test_set/segment_exertion_labels.csv",
        description="Test Set"
    )
    
    if train_success and test_success:
        print("\n=== Feature Extraction Completed ===")
        print("Training features: data/train_set/features/")
        print("Test features: data/test_set/features/")
    else:
        print("\nERROR: Errors occurred during feature extraction")

if __name__ == "__main__":
    main()
