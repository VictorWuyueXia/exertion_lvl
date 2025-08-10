import os
import yaml

class DataConfig:
    """Configuration class for data preparation pipeline"""
    
    def __init__(self, config_path="config/vgg16_wav2vec2_layer4_mfcc_exertion.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Data splitting parameters
        self.test_size = config['data']['split']['test_size']
        self.val_size = config['data']['split']['val_size']
        self.random_state = config['data']['split']['random_state']
        self.use_stratified_split = config['data']['split']['use_stratified_split']
        
        # Feature extraction parameters
        self.use_mfcc = config['data']['features']['use_mfcc']
        self.use_wav2vec2 = config['data']['features']['use_wav2vec2']
        self.wav2vec2_layers = config['data']['features']['wav2vec2_layers']
        self.target_frames = config['data']['target_frames']
        self.normalize_data = config['data']['normalize_data']
        
        # Model parameters
        self.mfcc_dim = config['model']['mfcc_dim']
        self.wav2vec2_dim = config['model']['wav2vec2_dim']
        
        # Audio parameters
        self.sample_rate = 16000
        
        # Paths
        self.raw_data_dir = "data/audio"
        self.metadata_file = "data/general_information.csv"
        self.processed_dir = "data/processed"
        
        # Cleaned data
        self.cleaned_dir = os.path.join(self.processed_dir, "01_cleaned")
        self.cleaned_audio_dir = os.path.join(self.cleaned_dir, "audio")
        self.cleaned_metadata_file = os.path.join(self.cleaned_dir, "metadata.csv")
        
        # Data splits
        self.splits_dir = os.path.join(self.processed_dir, "02_splits")
        self.train_dir = os.path.join(self.splits_dir, "train")
        self.val_dir = os.path.join(self.splits_dir, "val")
        self.test_dir = os.path.join(self.splits_dir, "test")
        self.splits_labels_file = os.path.join(self.splits_dir, "labels.csv")
        
        # Features
        self.features_dir = os.path.join(self.processed_dir, "03_features")
        
        # Final data
        self.final_dir = os.path.join(self.processed_dir, "04_final")
        self.final_features_dir = os.path.join(self.final_dir, "features")
        self.final_labels_file = os.path.join(self.final_dir, "labels.csv")
        self.scalers_file = os.path.join(self.final_dir, "scalers.pkl")
        self.processing_report_file = os.path.join(self.final_dir, "processing_report.txt")
        
        # Cleanup settings
        self.cleanup_intermediate = True
