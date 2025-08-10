#!/usr/bin/env python3
"""
Data preparation script for exertion level classification
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data_preparation import run_data_preparation

def main():
    parser = argparse.ArgumentParser(description="Data preparation for exertion level classification")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/vgg16_wav2vec2_layer4_mfcc_exertion.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--archive-old", 
        action="store_true",
        help="Archive old processed data before running"
    )
    
    args = parser.parse_args()
    
    # Archive old processed data if requested
    if args.archive_old:
        old_processed_dir = "data/processed"
        if os.path.exists(old_processed_dir):
            import shutil
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = f"data/processed_archive_{timestamp}"
            print(f"Archiving old processed data to: {archive_dir}")
            shutil.move(old_processed_dir, archive_dir)
    
    # Run data preparation
    print("Starting data preparation pipeline...")
    pipeline = run_data_preparation(args.config)
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main()
