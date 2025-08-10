import os
import pandas as pd
from tqdm import tqdm

class LabelManager:
    """Label management for clip generation"""
    
    def __init__(self, config):
        self.config = config
    
    def create_clip_labels(self):
        """Create labels for all generated clips"""
        print("Creating clip labels...")
        
        # Load session labels
        labels_df = pd.read_csv(self.config.splits_labels_file)
        
        clip_labels = []
        
        # Process each split
        for split in ['train', 'val', 'test']:
            split_sessions = labels_df[labels_df['split'] == split]
            features_dir = os.path.join(self.config.features_dir, split)
            
            if not os.path.exists(features_dir):
                continue
            
            # Get all clip files in this split
            clip_files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
            
            for clip_file in tqdm(clip_files, desc=f"Processing {split} clips"):
                clip_id = os.path.splitext(clip_file)[0]
                
                # Extract session_id from clip_id (format: session_id_clip_XXX)
                session_id = '_'.join(clip_id.split('_')[:-2])
                
                # Find corresponding session label
                session_label = split_sessions[split_sessions['session_id'] == session_id]
                
                if not session_label.empty:
                    exertion_level = session_label.iloc[0]['exertion_level']
                    
                    clip_labels.append({
                        'clip_id': clip_id,
                        'session_id': session_id,
                        'exertion_level': exertion_level,
                        'split': split
                    })
        
        # Create DataFrame and save
        clip_labels_df = pd.DataFrame(clip_labels)
        clip_labels_df.to_csv(self.config.final_labels_file, index=False)
        
        print("Clip labels created!")
        print(f"Total clips: {len(clip_labels_df)}")
        
        # Print statistics
        for split in ['train', 'val', 'test']:
            split_clips = clip_labels_df[clip_labels_df['split'] == split]
            if not split_clips.empty:
                print(f"{split.capitalize()} set: {len(split_clips)} clips")
                level_counts = split_clips['exertion_level'].value_counts().sort_index()
                print(f"  Exertion level distribution: {dict(level_counts)}")
