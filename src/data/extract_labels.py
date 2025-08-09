#!/usr/bin/env python3
"""
Label extraction script
Create corresponding labels for segmented audio in training and test sets
"""

import pandas as pd
import os
from tqdm import tqdm

def create_segment_labels_for_dataset(audio_dir, labels_file, output_file, description):
    """Create labels for segmented audio in specified dataset"""
    
    print(f"\nStarting {description}...")
    
    # Read label file
    if not os.path.exists(labels_file):
        print(f"ERROR: Label file {labels_file} does not exist")
        return False
    
    labels_df = pd.read_csv(labels_file)
    
    # Check segmented audio directory
    if not os.path.exists(audio_dir):
        print(f"ERROR: Segmented audio directory {audio_dir} does not exist")
        return False
    
    # Get all segmented audio files
    segmented_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    if not segmented_files:
        print(f"{description} segmented audio directory is empty")
        return False
    
    print(f"Found {len(segmented_files)} segmented audio files")
    
    # Create segmented label mapping
    segment_labels = []
    
    for filename in tqdm(segmented_files, desc=f"Processing {description} labels"):
        # Parse filename: session_id_stride_N.wav
        # Example: d31_P01_6_0_clip_1_stride_1.wav
        session_id = filename.replace('.wav', '')
        
        # Find corresponding original session_id (remove _stride_N part)
        original_session_id = None
        for label_session in labels_df['session_id']:
            if session_id.startswith(label_session):
                original_session_id = label_session
                break
        
        if original_session_id is not None:
            exertion_level = labels_df[labels_df['session_id'] == original_session_id]['exertion_level'].iloc[0]
            segment_labels.append({
                'segment_id': session_id,
                'original_session_id': original_session_id,
                'exertion_level': exertion_level
            })
        else:
            print(f"WARNING: Cannot find label for {session_id}")
    
    # Save segmented labels
    segment_labels_df = pd.DataFrame(segment_labels)
    segment_labels_df.to_csv(output_file, index=False)
    
    print(f"{description} label creation completed!")
    print(f"Segmented label file saved to: {output_file}")
    print(f"Successfully mapped segments: {len(segment_labels_df)}")
    
    # Show segmented label distribution
    print(f"{description} label distribution:")
    level_counts = segment_labels_df['exertion_level'].value_counts().sort_index()
    for level, count in level_counts.items():
        print(f"  Level {level}: {count} segments")
    
    return True

def main():
    """Main function"""
    print("=== Label Extraction ===")
    
    # Training set label extraction
    train_success = create_segment_labels_for_dataset(
        audio_dir="data/train_set/segmented_audio",
        labels_file="data/train_set/labels/exertion_labels.csv",
        output_file="data/train_set/segment_exertion_labels.csv",
        description="Training Set"
    )
    
    # Test set label extraction
    test_success = create_segment_labels_for_dataset(
        audio_dir="data/test_set/segmented_audio",
        labels_file="data/test_set/labels/exertion_labels.csv",
        output_file="data/test_set/segment_exertion_labels.csv",
        description="Test Set"
    )
    
    if train_success and test_success:
        print("\n=== Label Extraction Completed ===")
        print("Training segmented labels: data/train_set/segment_exertion_labels.csv")
        print("Test segmented labels: data/test_set/segment_exertion_labels.csv")
    else:
        print("\nERROR: Errors occurred during label extraction")

if __name__ == "__main__":
    main()
