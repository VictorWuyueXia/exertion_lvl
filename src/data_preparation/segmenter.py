import os
import torch
import numpy as np
import torchaudio
import soundfile as sf
from tqdm import tqdm
from pathlib import Path

class AudioSegmenter:
    """Audio segmentation module"""
    
    def __init__(self, config):
        self.config = config
        
    def segment_indices(self, total_len, window_len, stride_len):
        """Generate segment indices"""
        return [(start, min(start + window_len, total_len))
                for start in range(0, total_len, stride_len)
                if start + window_len <= total_len]
    
    def segment_audio_directory(self, input_dir, output_dir, description):
        """Segment audio in specified directory"""
        print(f"Starting {description}...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        audio_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
        segment_counts = []
        
        for fname in tqdm(sorted(audio_files), desc=f"Processing {description}"):
            session_id = os.path.splitext(fname)[0]
            audio_path = os.path.join(input_dir, fname)
            
            # Load audio
            y, sr = sf.read(audio_path)
            if sr != self.config.sample_rate:
                y = torchaudio.functional.resample(
                    torch.tensor(y), orig_freq=sr, new_freq=self.config.sample_rate
                ).numpy()
            
            audio_len = len(y)
            
            # Compute segment indices
            audio_window = int(self.config.clip_duration * self.config.sample_rate)
            audio_stride = int(self.config.clip_stride * self.config.sample_rate)
            
            audio_segments = self.segment_indices(audio_len, audio_window, audio_stride)
            num_segments = len(audio_segments)
            segment_counts.append({"session_id": session_id, "num_segments": num_segments})
            
            # Save segments
            for i, (a_start, a_end) in enumerate(audio_segments, start=1):
                seg_name = f"{session_id}_segment_{i:03d}"
                
                # Save audio segment
                audio_segment = y[a_start:a_end]
                sf.write(os.path.join(output_dir, f"{seg_name}.wav"), audio_segment, self.config.sample_rate)
        
        total_segments = sum([seg['num_segments'] for seg in segment_counts])
        print(f"{description} completed!")
        print(f"Processed {len(audio_files)} files")
        print(f"Total segments: {total_segments}")
        print(f"Segmented audio saved in: {output_dir}")
        
        return segment_counts
    
    def segment_all_data(self):
        """Segment audio for all data splits"""
        print("=== Audio Segmentation ===")
        
        # Segment training set
        train_segments = self.segment_audio_directory(
            self.config.train_dir,
            os.path.join(self.config.train_dir, "segments"),
            "Training Set Audio Segmentation"
        )
        
        # Segment validation set
        val_segments = self.segment_audio_directory(
            self.config.val_dir,
            os.path.join(self.config.val_dir, "segments"),
            "Validation Set Audio Segmentation"
        )
        
        # Segment test set
        test_segments = self.segment_audio_directory(
            self.config.test_dir,
            os.path.join(self.config.test_dir, "segments"),
            "Test Set Audio Segmentation"
        )
        
        return train_segments, val_segments, test_segments
