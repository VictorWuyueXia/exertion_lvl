import os
import torch
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torchaudio
import warnings
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from tqdm import tqdm
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class FeatureExtractor:
    """Feature extraction module - downsample sessions to 20Hz, extract features, align and clip"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load wav2vec2 model
        if self.config.use_wav2vec2:
            print("Loading wav2vec2 model...")
            model_name = "facebook/wav2vec2-base"
            self.wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name)
            self.wav2vec2_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.wav2vec2_model.to(self.device)
            self.wav2vec2_model.eval()
    
    def downsample_audio(self, audio_path):
        """Downsample audio to 20Hz for consistent feature extraction"""
        y, sr = sf.read(audio_path)
        
        # Resample to 16kHz
        target_sr = 16000
        if sr != target_sr:
            y = torchaudio.functional.resample(
                torch.tensor(y), orig_freq=sr, new_freq=target_sr
            ).numpy()
        
        return y, target_sr
    
    def extract_mfcc(self, audio_resampled, sr_resampled):
        """从16kHz音频中提取50Hz的MFCC特征"""
        target_hz = 50  # 从20Hz提高到50Hz
        hop_length = int(sr_resampled / target_hz)  # 16000/50=320
        n_fft = 400  # 25ms窗口

        # 提取MFCC
        mfcc = librosa.feature.mfcc(
            y=audio_resampled,
            sr=sr_resampled,
            n_mfcc=self.config.mfcc_dim,
            hop_length=hop_length,
            n_fft=n_fft
        )
        # 转置为 (时间, 特征)
        mfcc = mfcc.T

        return mfcc

    def extract_wav2vec2(self, audio_resampled, sr_resampled):
        """基于16kHz音频提取wav2vec2特征，保持50Hz"""
        # 确保输入音频为16kHz
        if sr_resampled != 16000:
            audio_16k = torchaudio.functional.resample(
                torch.tensor(audio_resampled), orig_freq=sr_resampled, new_freq=16000
            ).numpy()
        else:
            audio_16k = audio_resampled

        # 准备wav2vec2输入
        inputs = self.wav2vec2_feature_extractor(audio_16k, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 提取特征
        with torch.no_grad():
            outputs = self.wav2vec2_model(**inputs, output_hidden_states=True)
            features = outputs.hidden_states[self.config.wav2vec2_layers[0]].squeeze(0)

        # features.shape: (帧数, 特征维度)，帧率约为16kHz/320=50Hz
        features_np = features.cpu().numpy()  # (T_50hz, D)
        return features_np  # 直接返回50Hz特征

    def create_clips_from_aligned_features(self, mfcc_features, wav2vec2_features, session_id, output_dir):
        """基于50Hz对齐特征，创建15秒clip，步长5秒"""
        # 两种特征的时间维度应一致
        assert mfcc_features.shape[0] == wav2vec2_features.shape[0], "Features must have same time dimension"

        total_frames = mfcc_features.shape[0]
        clip_frames = 15 * 50  # 15秒 * 50Hz = 750帧
        stride_frames = 5 * 50  # 5秒 * 50Hz = 250帧

        clips = []

        # 检查帧数是否足够
        if total_frames < clip_frames:
            print(f"Warning: Session {session_id} has only {total_frames} frames, need {clip_frames} for 15s clip")
            return clips

        for start_frame in range(0, total_frames - clip_frames + 1, stride_frames):
            end_frame = start_frame + clip_frames

            # 截取clip特征
            mfcc_clip = mfcc_features[start_frame:end_frame]
            wav2vec2_clip = wav2vec2_features[start_frame:end_frame]

            # 保存clip
            clip_id = f"{session_id}_clip_{len(clips)+1:03d}"
            clip_path = os.path.join(output_dir, f"{clip_id}.npy")

            # 保存两个特征到一个文件
            np.save(clip_path, {
                'mfcc': mfcc_clip,
                'wav2vec2': wav2vec2_clip
            })

            clips.append(clip_id)

        return clips
    
    def extract_features_for_split(self, split_dir, output_dir, description):
        """Extract features for all sessions in a split"""
        print(f"Starting {description} feature extraction...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get session audio files
        session_files = [f for f in os.listdir(split_dir) if f.endswith('.wav')]
        all_clips = []
        
        for session_file in tqdm(session_files, desc=f"Processing {description} sessions"):
            session_id = os.path.splitext(session_file)[0]
            audio_path = os.path.join(split_dir, session_file)
            
            # Downsample audio
            audio_resampled, sr_resampled = self.downsample_audio(audio_path)
            
            # Extract MFCC features
            mfcc_features = self.extract_mfcc(audio_resampled, sr_resampled)
            
            # Extract wav2vec2 features
            wav2vec2_features = self.extract_wav2vec2(audio_resampled, sr_resampled)
            
            # Ensure both features have same time dimension
            min_frames = min(mfcc_features.shape[0], wav2vec2_features.shape[0])
            mfcc_features = mfcc_features[:min_frames]
            wav2vec2_features = wav2vec2_features[:min_frames]
            
            # Create clips from aligned features
            clips = self.create_clips_from_aligned_features(
                mfcc_features, wav2vec2_features, session_id, output_dir
            )
            all_clips.extend(clips)
        
        print(f"{description} feature extraction completed!")
        print(f"Processed {len(session_files)} sessions")
        print(f"Created {len(all_clips)} clips")
    
    def extract_all_features(self):
        """Extract features for all data splits"""
        print("=== Feature Extraction ===")
        
        # Extract features for training set
        train_output_dir = os.path.join(self.config.features_dir, "train")
        self.extract_features_for_split(
            self.config.train_dir, train_output_dir, "Training Set"
        )
        
        # Extract features for validation set
        val_output_dir = os.path.join(self.config.features_dir, "val")
        self.extract_features_for_split(
            self.config.val_dir, val_output_dir, "Validation Set"
        )
        
        # Extract features for test set
        test_output_dir = os.path.join(self.config.features_dir, "test")
        self.extract_features_for_split(
            self.config.test_dir, test_output_dir, "Test Set"
        )
