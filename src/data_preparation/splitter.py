import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DataSplitter:
    """Data splitting module"""
    
    def __init__(self, config):
        self.config = config
    
    def split_data(self):
        """Split cleaned data into train, validation, and test sets"""
        print("Loading cleaned metadata...")
        cleaned_metadata = pd.read_csv(self.config.cleaned_metadata_file)
        
        print("Splitting data...")
        # First split: separate test set
        train_val_data, test_data = train_test_split(
            cleaned_metadata, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=cleaned_metadata['exertion_level'] if self.config.use_stratified_split else None
        )
        
        # Second split: separate validation set from training set
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=self.config.val_size / (1 - self.config.test_size),
            random_state=self.config.random_state,
            stratify=train_val_data['exertion_level'] if self.config.use_stratified_split else None
        )
        
        # Copy audio files to respective directories
        print("Copying files to split directories...")
        
        # Copy training files
        for _, row in tqdm(train_data.iterrows(), total=len(train_data), desc="Copying training files"):
            session_id = row['session_id']
            src_path = os.path.join(self.config.cleaned_audio_dir, f"{session_id}.wav")
            dst_path = os.path.join(self.config.train_dir, f"{session_id}.wav")
            shutil.copy2(src_path, dst_path)
        
        # Copy validation files
        for _, row in tqdm(val_data.iterrows(), total=len(val_data), desc="Copying validation files"):
            session_id = row['session_id']
            src_path = os.path.join(self.config.cleaned_audio_dir, f"{session_id}.wav")
            dst_path = os.path.join(self.config.val_dir, f"{session_id}.wav")
            shutil.copy2(src_path, dst_path)
        
        # Copy test files
        for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Copying test files"):
            session_id = row['session_id']
            src_path = os.path.join(self.config.cleaned_audio_dir, f"{session_id}.wav")
            dst_path = os.path.join(self.config.test_dir, f"{session_id}.wav")
            shutil.copy2(src_path, dst_path)
        
        # Create combined labels file
        train_data['split'] = 'train'
        val_data['split'] = 'val'
        test_data['split'] = 'test'
        
        combined_labels = pd.concat([train_data, val_data, test_data], ignore_index=True)
        combined_labels.to_csv(self.config.splits_labels_file, index=False)
        
        print("Data splitting completed!")
        print(f"Training sessions: {len(train_data)}")
        print(f"Validation sessions: {len(val_data)}")
        print(f"Test sessions: {len(test_data)}")
        print(f"Split data saved to: {self.config.splits_dir}")
        
        # Print exertion level distribution
        for split_name, split_data in [('Training', train_data), ('Validation', val_data), ('Test', test_data)]:
            level_counts = split_data['exertion_level'].value_counts().sort_index()
            print(f"{split_name} set exertion level distribution: {dict(level_counts)}")
