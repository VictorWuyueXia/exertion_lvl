#!/usr/bin/env python3
"""
Simple training runner for exertion level detection model
This script provides a clean interface to run the training pipeline
"""

import os
import sys
import torch
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main training function"""
    print("="*60)
    print("EXERTION LEVEL DETECTION MODEL TRAINING")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("CUDA not available, using CPU")
    
    # Check if we're in the correct environment
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    
    # Import and run training
    try:
        from src.train import main as train_main
        train_main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed and the project structure is correct")
        return
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
