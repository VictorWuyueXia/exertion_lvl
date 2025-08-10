import os
import shutil
import pandas as pd
import wave
from tqdm import tqdm

class DataCleaner:
    """Data cleaning module"""
    
    def __init__(self, config):
        self.config = config
    
    def get_wav_duration(self, audio_path):
        """Get duration of wav file in seconds"""
        with wave.open(audio_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration
    
    def clean_data(self):
        """Clean raw data based on criteria"""
        metadata = pd.read_csv(self.config.metadata_file)
        
        # Filter based on duration and naming
        valid_sessions = []
        
        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing sessions"):
            session = row['Session Name']
            
            # Check if audio file exists
            audio_path = os.path.join(self.config.raw_data_dir, f"{session}.wav")
            if not os.path.exists(audio_path):
                continue
            
            # Get duration
            duration = self.get_wav_duration(audio_path)
            
            # Apply duration filter (keep sessions between 15 seconds)
            if duration < 15:
                continue
            
            # Apply naming filter (exclude sessions with specific patterns)
            if any(pattern in session.lower() for pattern in ['test', 'debug', 'invalid']):
                continue
            
            # Create row with correct column names
            valid_row = {
                'session_id': session,
                'exertion_level': row['Exertion'],
                'duration': duration
            }
            valid_sessions.append(valid_row)
        
        # Create cleaned metadata
        cleaned_metadata = pd.DataFrame(valid_sessions)
        cleaned_metadata.to_csv(self.config.cleaned_metadata_file, index=False)
        
        # Copy valid audio files
        print("Copying valid audio files...")
        for _, row in tqdm(cleaned_metadata.iterrows(), total=len(cleaned_metadata), desc="Copying audio files"):
            session = row['session_id']
            src_path = os.path.join(self.config.raw_data_dir, f"{session}.wav")
            dst_path = os.path.join(self.config.cleaned_audio_dir, f"{session}.wav")
            shutil.copy2(src_path, dst_path)
        
