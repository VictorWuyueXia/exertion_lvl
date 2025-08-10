import os
import time
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path

from .config import DataConfig
from .cleaner import DataCleaner
from .splitter import DataSplitter
from .feature_extractor import FeatureExtractor
from .label_manager import LabelManager
from .normalizer import DataNormalizer

class DataPreparationPipeline:
    """Complete data preparation pipeline"""
    
    def __init__(self, config_path="config/vgg16_wav2vec2_layer4_mfcc_exertion.yaml"):
        self.config = DataConfig(config_path)
        self.start_time = None
    
    def run(self):
        """Run complete data preparation pipeline"""
        self.start_time = time.time()
        
        
        # Step 1: Create directories
        print("Step 1: Creating directories...")
        self._create_directories()
        print("Directories created successfully!")
        print()
        
        # Step 2: Data cleaning
        print("Step 2: Data cleaning...")
        cleaner = DataCleaner(self.config)
        cleaner.clean_data()
        print()
        
        # Step 3: Data splitting
        print("Step 3: Data splitting...")
        splitter = DataSplitter(self.config)
        splitter.split_data()
        print()
        
        # Step 4: Feature extraction
        print("Step 4: Feature extraction...")
        extractor = FeatureExtractor(self.config)
        extractor.extract_all_features()
        print()
        
        # Step 5: Label management
        print("Step 5: Label management...")
        label_manager = LabelManager(self.config)
        label_manager.create_clip_labels()
        print()
        
        # Step 6: Data normalization
        print("Step 6: Data normalization...")
        normalizer = DataNormalizer(self.config)
        normalizer.normalize_all()
        print()
        
        # Step 7: Clean up intermediate data
        print("Cleaning up intermediate data...")
        self.cleanup_intermediate_data()
        print("Intermediate data cleanup completed!")
        print()
        
        # Generate final report
        self._generate_report()
        
        print("=" * 60)
        print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
    
    def _create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.config.cleaned_dir,
            self.config.cleaned_audio_dir,
            self.config.splits_dir,
            self.config.train_dir,
            self.config.val_dir,
            self.config.test_dir,
            self.config.features_dir,
            self.config.final_dir,
            self.config.final_features_dir,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def cleanup_intermediate_data(self):
        """Clean up intermediate data to save storage"""
        if self.config.cleanup_intermediate:
            # Remove large intermediate directories
            if os.path.exists(self.config.cleaned_audio_dir):
                print(f"Removed: {self.config.cleaned_audio_dir}")
                import shutil
                shutil.rmtree(self.config.cleaned_audio_dir)
            
            for split_dir in [self.config.train_dir, self.config.val_dir, self.config.test_dir]:
                if os.path.exists(split_dir):
                    print(f"Removed: {split_dir}")
                    import shutil
                    shutil.rmtree(split_dir)
            
            if os.path.exists(self.config.features_dir):
                print(f"Removed: {self.config.features_dir}")
                import shutil
                shutil.rmtree(self.config.features_dir)
    
    def _generate_report(self):
        """Generate final processing report"""
        end_time = time.time()
        processing_time = end_time - self.start_time
        
        # Load final statistics
        import pandas as pd
        clip_labels = pd.read_csv(self.config.final_labels_file)
        total_clips = len(clip_labels)
        
        # Count clips per split
        train_clips = len(clip_labels[clip_labels['split'] == 'train'])
        val_clips = len(clip_labels[clip_labels['split'] == 'val'])
        test_clips = len(clip_labels[clip_labels['split'] == 'test'])
        
        # Load cleaned metadata for session count
        cleaned_metadata = pd.read_csv(self.config.cleaned_metadata_file)
        total_sessions = len(cleaned_metadata)
        
        # Generate report
        report_content = f"""FINAL PROCESSING REPORT
        ----------------------------------------
        Processing completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}
        Total processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)

        DATA STATISTICS:
        Cleaned sessions: {total_sessions}
        Total clips: {total_clips}

        CLIP STATISTICS:
        Training clips: {train_clips}
        Validation clips: {val_clips}
        Test clips: {test_clips}

        CONFIGURATION:
        Sample rate: {self.config.sample_rate} Hz
        Target frames: {self.config.target_frames}
        MFCC dimensions: {self.config.mfcc_dim}
        Use MFCC: {self.config.use_mfcc}
        Use wav2vec2: {self.config.use_wav2vec2}
        wav2vec2 layers: {self.config.wav2vec2_layers}
        Data normalization: {self.config.normalize_data}

        FINAL DATA STRUCTURE:
        Final features: {self.config.final_features_dir}
        Final labels: {self.config.final_labels_file}
        Scalers: {self.config.scalers_file}
        """
        
        # Save report
        with open(self.config.processing_report_file, 'w') as f:
            f.write(report_content)
        
        # Print summary
        print("FINAL PROCESSING REPORT")
        print("-" * 40)
        print(f"Processing completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
        print()
        print("DATA STATISTICS:")
        print(f"  Cleaned sessions: {total_sessions}")
        print(f"  Training clips: {train_clips}")
        print(f"  Validation clips: {val_clips}")
        print(f"  Test clips: {test_clips}")
        print()
        print("CONFIGURATION:")
        print(f"  Sample rate: {self.config.sample_rate} Hz")
        print(f"  Target frames: {self.config.target_frames}")
        print(f"  MFCC dimensions: {self.config.mfcc_dim}")
        print(f"  Use MFCC: {self.config.use_mfcc}")
        print(f"  Use wav2vec2: {self.config.use_wav2vec2}")
        print(f"  wav2vec2 layers: {self.config.wav2vec2_layers}")
        print(f"  Data normalization: {self.config.normalize_data}")
        print()
        print("FINAL DATA STRUCTURE:")
        print(f"  Final features: {self.config.final_features_dir}")
        print(f"  Final labels: {self.config.final_labels_file}")
        print(f"  Scalers: {self.config.scalers_file}")
        print()
        print(f"Detailed report saved to: {self.config.processing_report_file}")

def run_data_preparation(config_path="config/vgg16_wav2vec2_layer4_mfcc_exertion.yaml"):
    """Convenience function to run data preparation"""
    pipeline = DataPreparationPipeline(config_path)
    pipeline.run()
    return pipeline
