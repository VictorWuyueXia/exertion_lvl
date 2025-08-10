from .config import DataConfig
from .cleaner import DataCleaner
from .splitter import DataSplitter
from .feature_extractor import FeatureExtractor
from .label_manager import LabelManager
from .normalizer import DataNormalizer
from .loader import ExertionDataset, create_data_loaders
from .pipeline import DataPreparationPipeline

__all__ = [
    'DataConfig',
    'DataCleaner',
    'DataSplitter',
    'FeatureExtractor',
    'LabelManager',
    'DataNormalizer',
    'ExertionDataset',
    'create_data_loaders',
    'DataPreparationPipeline',
    'run_data_preparation'
]

def run_data_preparation(config_path="config/vgg16_wav2vec2_layer4_mfcc_exertion.yaml"):
    """Run the complete data preparation pipeline"""
    pipeline = DataPreparationPipeline(config_path)
    pipeline.run()
