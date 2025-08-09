#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training script for exertion level detection model
Complete pipeline integrating data processing, model training and evaluation
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_manager import load_config_from_args, ConfigManager
from src.data.loader import AudioFeatureDataset, get_session_metadata
from src.training.trainer import ExertionTrainer
from src.evaluation.evaluator import ExertionEvaluator
from src.models.vgg16_exertion import create_model, count_parameters

def setup_environment(config: ConfigManager):
    """Setup training environment"""
    print("Setting up training environment...")
    
    # Set random seed
    seed = config.get('system.seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set GPU
    device = config.get('system.gpu.device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Set GPU memory usage fraction
        memory_fraction = config.get('system.gpu.memory_fraction', 0.95)
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        # Enable GPU optimizations
        if config.get('system.gpu.enable_benchmark', True):
            torch.backends.cudnn.benchmark = True
        
        if config.get('system.gpu.enable_flash_attention', True):
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
        
        # Enable TF32 (improve computational efficiency)
        if config.get('system.gpu.enable_tf32', True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable automatic mixed precision
        if config.get('system.gpu.enable_amp', True):
            print("Automatic mixed precision training enabled")
        
        # Enable torch.compile
        if config.get('system.gpu.enable_compile', False):
            print("torch.compile optimization enabled")
        
        print(f"GPU memory usage: {memory_fraction*100:.0f}%")
        print(f"cuDNN benchmark: {config.get('system.gpu.enable_benchmark', True)}")
        print(f"TF32: {config.get('system.gpu.enable_tf32', True)}")
        print(f"Flash Attention: {config.get('system.gpu.enable_flash_attention', True)}")
    
    return device

def check_data_availability(config: ConfigManager):
    """Check data availability"""
    print("Checking data availability...")
    
    # Check audio files
    audio_dir = config.get('data.audio_dir', 'data/audio')
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory does not exist: {audio_dir}")
        return False
    
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    if len(audio_files) == 0:
        print(f"Error: No WAV files found in audio directory: {audio_dir}")
        return False
    
    print(f"Found {len(audio_files)} audio files")
    
    # Check label file
    label_file = config.get('data.label_file', 'data/general_information.csv')
    if not os.path.exists(label_file):
        print(f"Error: Label file does not exist: {label_file}")
        return False
    
    labels_df = pd.read_csv(label_file)
    print(f"Found {len(labels_df)} label records")
    
    # Check feature directory
    feature_dir = config.get('data.feature_dir', 'data/features')
    if not os.path.exists(feature_dir):
        print("Feature directory does not exist, need to generate feature files first")
        return False
    
    # Check feature files
    feature_files = []
    for root, dirs, files in os.walk(feature_dir):
        feature_files.extend([os.path.join(root, f) for f in files if f.endswith('.npy')])
    
    if len(feature_files) == 0:
        print("No feature files found in feature directory, need to generate feature files first")
        return False
    
    print(f"Found {len(feature_files)} feature files")
    
    return True

def generate_features_if_needed(config: ConfigManager):
    """Generate feature files if needed"""
    feature_dir = config.get('data.feature_dir', 'data/features')
    
    if not os.path.exists(feature_dir):
        print("Feature directory does not exist, starting feature generation...")
        os.makedirs(feature_dir, exist_ok=True)
        
        # Run feature extraction
        from src.data.get_features import generate_feature_dir_with_labels
        
        # Read label data
        label_file = config.get('data.label_file', 'data/general_information.csv')
        labels_df = pd.read_csv(label_file)
        
        # Generate features
        audio_dir = config.get('data.audio_dir', 'data/audio')
        wav2vec2_layers = config.get('data.wav2vec2_layers', [4])
        
        generate_feature_dir_with_labels(
            audio_dir=audio_dir,
            features_dir=feature_dir,
            labels_df=labels_df,
            sr_target=16000,
            sr_feature=20,
            selected_layers=wav2vec2_layers,
            batch_processing=True
        )
        
        print("Feature files generation completed")
    else:
        print("Feature directory exists, skipping feature generation")

def prepare_dataset(config: ConfigManager):
    """Prepare dataset"""
    print("Preparing dataset...")
    
    # Get session metadata
    feature_dir = config.get('data.feature_dir', 'data/features')
    metadata_df = get_session_metadata(feature_dir)
    print(f"Found {len(metadata_df)} sessions")
    
    # Read label data
    label_file = config.get('data.label_file', 'data/general_information.csv')
    labels_df = pd.read_csv(label_file)
    labels_df['segment_id'] = labels_df['Session Name']
    print(f"Found {len(labels_df)} label records")
    
    # Create dataset
    dataset = AudioFeatureDataset(
        metadata_df=metadata_df,
        feature_dir=feature_dir,
        labels_df=labels_df,
        use_acoustic=config.get('data.features.use_mfcc', True),
        use_mfb=config.get('data.features.use_mfb', False),
        use_embed=config.get('data.features.use_wav2vec2', True),
        selected_wav2vec2_layers=config.get('data.wav2vec2_layers', [4])
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    return dataset

def train_model(config: ConfigManager, dataset):
    """Train model"""
    print("Starting model training...")
    
    # Create trainer
    trainer = ExertionTrainer(config.to_dict())
    
    # Save configuration
    trainer._save_config()
    
    # Create model and display parameter count
    model = create_model(config.to_dict())
    param_info = count_parameters(model)
    print(f"Model parameters: {param_info['total_params_millions']:.2f}M")
    print(f"Trainable parameters: {param_info['trainable_params_millions']:.2f}M")
    
    # Cross-validation training
    cv_results = trainer.cross_validation_train(dataset, config.to_dict())
    
    return trainer, cv_results

def evaluate_model(trainer, cv_results, config: ConfigManager):
    """Evaluate model"""
    print("Starting model evaluation...")
    
    # Create evaluator
    evaluator = ExertionEvaluator(trainer.result_dir, config.to_dict())
    
    # Evaluate cross-validation results
    overall_results = evaluator.evaluate_cross_validation(cv_results)
    
    # Print main results
    print("\n" + "="*50)
    print("Training completed! Main results:")
    print("="*50)
    print(f"Average accuracy: {overall_results['avg_accuracy']:.4f} ± {overall_results['std_accuracy']:.4f}")
    print(f"Fold accuracies: {[f'{acc:.4f}' for acc in overall_results['fold_accuracies']]}")
    print(f"Overall F1 score: {overall_results['overall_metrics']['f1_macro']:.4f}")
    print(f"Overall AUC: {overall_results['overall_metrics']['auc']:.4f}")
    print(f"Results saved to: {trainer.result_dir}")
    print("="*50)
    
    return overall_results

def main():
    """Main function"""
    print("Exertion level detection model training started")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Load configuration
    config = load_config_from_args()
    print(f"Configuration loaded from: {config.config_path}")
    
    # Setup environment
    device = setup_environment(config)
    
    # Check data availability
    if not check_data_availability(config):
        print("Data check failed, exiting training")
        return
    
    # Generate feature files (if needed)
    skip_feature_generation = config.get('data.skip_feature_generation', False)
    if not skip_feature_generation:
        generate_features_if_needed(config)
    else:
        print("Skipping feature generation step")
    
    # Prepare dataset
    dataset = prepare_dataset(config)
    
    # Train model
    trainer, cv_results = train_model(config, dataset)
    
    # Evaluate model
    overall_results = evaluate_model(trainer, cv_results, config)
    
    print("\nTraining pipeline completed!")
    print(f"All results saved to: {trainer.result_dir}")

if __name__ == "__main__":
    main()
