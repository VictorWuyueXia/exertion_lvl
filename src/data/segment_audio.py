#!/usr/bin/env python3
"""
Audio segmentation script
Segment audio for training and test sets
"""

import os
import torch
import numpy as np
import torchaudio
from tqdm import tqdm
import soundfile as sf

# --- Config ---
train_audio_dir = "data/train_set/audio"
test_audio_dir = "data/test_set/audio"
train_out_dir = "data/train_set/segmented_audio"
test_out_dir = "data/test_set/segmented_audio"

os.makedirs(train_out_dir, exist_ok=True)
os.makedirs(test_out_dir, exist_ok=True)

sr_target = 16000  # audio
window_sec = 15.0
stride_sec = 1.0

# --- Helper ---
def segment_indices(total_len, window_len, stride_len):
    return [(start, min(start + window_len, total_len))
            for start in range(0, total_len, stride_len)
            if start + window_len <= total_len]

def segment_audio_directory(input_dir, output_dir, description):
    """Segment audio in specified directory"""
    print(f"\nStarting {description}...")
    
    if not os.path.exists(input_dir):
        print(f"ERROR: Audio directory {input_dir} does not exist")
        return False
    
    segment_counts = []
    processed_count = 0
    skipped_count = 0

    audio_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    
    for fname in tqdm(sorted(audio_files), desc=f"Processing {description}"):
        session_id = os.path.splitext(fname)[0]
        audio_path = os.path.join(input_dir, fname)

        # Load audio
        y, sr = sf.read(audio_path)
        if sr != sr_target:
            y = torchaudio.functional.resample(torch.tensor(y), orig_freq=sr, new_freq=sr_target).numpy()
        audio_len = len(y)

        # Compute segment indices
        audio_window = int(window_sec * sr_target)
        audio_stride = int(stride_sec * sr_target)

        audio_segments = segment_indices(audio_len, audio_window, audio_stride)
        
        num_segments = len(audio_segments)
        segment_counts.append({"session_id": session_id, "num_segments": num_segments})

        # Save segments
        for i, (a_start, a_end) in enumerate(audio_segments, start=1):
            seg_name = f"{session_id}_stride_{i}"
            
            # Save audio segment
            audio_segment = y[a_start:a_end]
            sf.write(os.path.join(output_dir, f"{seg_name}.wav"), audio_segment, sr_target)
        
        processed_count += 1

    print(f"{description} completed!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Skipped: {skipped_count} files")
    print(f"Total segments: {sum([seg['num_segments'] for seg in segment_counts])}")
    print(f"Segmented audio saved in: {output_dir}")
    
    return True

def main():
    """Main function"""
    print("=== Audio Segmentation ===")
    
    # Check if training and test set directories exist
    if not os.path.exists(train_audio_dir):
        print(f"ERROR: Training audio directory {train_audio_dir} does not exist")
        print("Please run data splitting step first")
        return
    
    if not os.path.exists(test_audio_dir):
        print(f"ERROR: Test audio directory {test_audio_dir} does not exist")
        print("Please run data splitting step first")
        return
    
    # Segment training set
    if not segment_audio_directory(train_audio_dir, train_out_dir, "Training Set Audio Segmentation"):
        return
    
    # Segment test set
    if not segment_audio_directory(test_audio_dir, test_out_dir, "Test Set Audio Segmentation"):
        return
    
    print("\n=== Audio Segmentation Completed ===")
    print("Training segmented audio: data/train_set/segmented_audio/")
    print("Test segmented audio: data/test_set/segmented_audio/")

if __name__ == "__main__":
    main()
