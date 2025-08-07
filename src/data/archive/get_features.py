# --- Optimized Feature Extraction for RTX 4070 ---
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
import torchaudio.compliance.kaldi as kaldi
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

# Global variables for model loading (loaded once per process)
processor = None
model = None
device = None

def initialize_wav2vec2_model():
    """Initialize Wav2Vec2 model on GPU - optimized for RTX 4070"""
    global processor, model, device
    if model is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base", cache_dir=".cache")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", cache_dir=".cache", output_hidden_states=True)
        model = model.to(device)
        model.eval()
        
        # Enable optimizations for RTX 4070 (Ada Lovelace architecture)
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, 'enabled'):
            torch.backends.cudnn.enabled = True
        
        # Enable memory efficient attention if available (for Ada Lovelace)
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # Set memory fraction to prevent OOM on 12GB RTX 4070
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available memory

def get_optimal_batch_size():
    """Determine optimal batch size based on GPU memory - optimized for RTX 4070"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        if gpu_memory >= 24:  # RTX 4090/4080/3090/3080 Ti
            return 6
        elif gpu_memory >= 16:  # RTX 4080/3080
            return 4
        elif gpu_memory >= 12:  # RTX 4070 Ti/4070/3080
            return 3
        elif gpu_memory >= 8:   # RTX 3070/3060 Ti
            return 2
        else:
            return 1
    return 1

def clear_gpu_cache():
    """Clear GPU cache to prevent memory fragmentation on RTX 4070"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_memory_usage():
    """Get current GPU memory usage for monitoring"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3     # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return {
            'allocated_gb': allocated,
            'cached_gb': cached,
            'total_gb': total,
            'free_gb': total - cached
        }
    return None

# --- Optimized Acoustic Features (GPU accelerated) ---
class OptimizedAcousticExtractor:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    def extract_all_features_batch(self, audio_batch, sr=16000, target_frames=300):
        """Extract all acoustic features in a single GPU batch operation"""
        if isinstance(audio_batch, np.ndarray):
            audio_batch = torch.tensor(audio_batch, dtype=torch.float32, device=self.device)
        elif isinstance(audio_batch, list):
            # Pad sequences to same length
            max_len = max(len(a) for a in audio_batch)
            padded = []
            for audio in audio_batch:
                if isinstance(audio, np.ndarray):
                    audio = torch.tensor(audio, dtype=torch.float32)
                if len(audio) < max_len:
                    audio = F.pad(audio, (0, max_len - len(audio)))
                padded.append(audio)
            audio_batch = torch.stack(padded).to(self.device)
        
        if audio_batch.dim() == 1:
            audio_batch = audio_batch.unsqueeze(0)
        
        batch_size = audio_batch.shape[0]
        results = []
        
        for i in range(batch_size):
            audio = audio_batch[i]
            features = self._extract_single_audio_features(audio, sr, target_frames)
            results.append(features)
            
        return results
    
    def _extract_single_audio_features(self, audio, sr, target_frames):
        """Extract features for single audio on GPU"""
        audio_length = audio.shape[-1]
        hop_length = max(1, audio_length // target_frames)
        
        # All computations on GPU
        features = {}
        
        # 1. MFB (Mel Filter Bank) using Kaldi
        if audio.dim() == 1:
            audio_2d = audio.unsqueeze(0)
        else:
            audio_2d = audio
            
        mfb = kaldi.fbank(
            waveform=audio_2d,
            num_mel_bins=40,
            sample_frequency=sr,
            frame_length=25.0,
            frame_shift=hop_length * 1000 / sr,
            dither=0.0,
            energy_floor=0.0,
            use_energy=False
        )
        features['mfb'] = self._align_frames(mfb, target_frames)
        
        # 2. MFCC using torchaudio (GPU accelerated)
        mfcc_transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=40,
            melkwargs={
                "n_fft": 1024,
                "hop_length": hop_length,
                "n_mels": 128,
                "f_max": 8000
            }
        ).to(self.device)
        
        mfcc = mfcc_transform(audio.unsqueeze(0)).squeeze(0).T
        features['mfcc'] = self._align_frames(mfcc, target_frames)
        
        # 3. Mel Spectrogram
        mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            hop_length=hop_length,
            n_mels=40,
            f_max=8000
        ).to(self.device)
        
        mel_spec = mel_transform(audio.unsqueeze(0)).squeeze(0).T
        features['mel'] = self._align_frames(mel_spec, target_frames)
        
        # 4. RMS Energy (vectorized)
        # Reshape audio for frame-wise RMS
        n_frames = audio_length // hop_length
        if n_frames > 0:
            frames = audio[:n_frames * hop_length].view(n_frames, hop_length)
            rms = torch.sqrt(torch.mean(frames ** 2, dim=1, keepdim=True))
            features['rms'] = self._align_frames(rms, target_frames)
        else:
            features['rms'] = torch.zeros(target_frames, 1, device=self.device)
        
        # 5. PSD (Power Spectral Density)
        stft = torch.stft(
            audio,
            n_fft=1024,
            hop_length=hop_length,
            window=torch.hann_window(1024, device=self.device),
            return_complex=True
        )
        power_spectrum = (stft.abs() ** 2)
        psd = power_spectrum.sum(dim=0, keepdim=True).T
        features['psd'] = self._align_frames(psd, target_frames)
        
        # Convert to numpy and combine
        for key in features:
            features[key] = features[key].cpu().numpy()
            
        # Combine features
        combined = np.concatenate([
            features['mfb'], features['mel'], features['mfcc'], 
            features['rms'], features['psd']
        ], axis=1)
        features['combined'] = combined
        
        return features
    
    def _align_frames(self, feature, target_frames):
        """Align feature tensor to target frames"""
        current_frames = feature.shape[0]
        if current_frames == target_frames:
            return feature
        elif current_frames < target_frames:
            # Pad with last frame
            pad_length = target_frames - current_frames
            if feature.dim() == 1:
                last_frame = feature[-1:].repeat(pad_length)
                return torch.cat([feature, last_frame], dim=0)
            else:
                last_frame = feature[-1:].repeat(pad_length, 1)
                return torch.cat([feature, last_frame], dim=0)
        else:
            # Truncate or interpolate
            if current_frames <= target_frames * 1.5:  # Close enough, just truncate
                return feature[:target_frames]
            else:  # Downsample using interpolation
                return F.interpolate(
                    feature.T.unsqueeze(0), 
                    size=target_frames, 
                    mode='linear', 
                    align_corners=False
                ).squeeze(0).T

# --- Optimized Wav2Vec2 extraction with batching for RTX 4070 ---
def extract_wav2vec2_features_batch(audio_list, sr_list, selected_layers=(4,), sr_target=16000, batch_size=None):
    """Extract Wav2Vec2 features in batches - optimized for RTX 4070 memory management"""
    initialize_wav2vec2_model()
    
    if batch_size is None:
        batch_size = get_optimal_batch_size()
    
    # Clear GPU cache before processing
    clear_gpu_cache()
    
    # Resample all audio to target sr and prepare batches
    processed_audio = []
    for audio, sr in zip(audio_list, sr_list):
        if sr != sr_target:
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            audio = torchaudio.functional.resample(audio_tensor, sr, sr_target).numpy()
        processed_audio.append(audio)
    
    results = []
    
    # Process in batches
    for i in range(0, len(processed_audio), batch_size):
        batch_audio = processed_audio[i:i + batch_size]
        
        # Find max length in batch and pad
        max_len = max(len(audio) for audio in batch_audio)
        padded_batch = []
        attention_masks = []
        
        for audio in batch_audio:
            padded = np.pad(audio, (0, max_len - len(audio)), mode='constant')
            padded_batch.append(padded)
            # Create attention mask
            mask = np.ones(len(audio))
            mask = np.pad(mask, (0, max_len - len(audio)), mode='constant')
            attention_masks.append(mask)
        
        # Process batch
        inputs = processor(
            padded_batch, 
            sampling_rate=sr_target, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Use autocast for mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
            
            # Extract selected layers for each audio in batch
            for j, audio_idx in enumerate(range(len(batch_audio))):
                audio_results = []
                for layer_idx in selected_layers:
                    # Get the embeddings for this audio (remove padding)
                    original_length = len(processed_audio[i + j])
                    # Calculate the corresponding length in the feature space
                    feature_length = outputs.hidden_states[layer_idx].shape[1]
                    audio_feature_length = int(feature_length * len(processed_audio[i + j]) / max_len)
                    
                    layer_features = outputs.hidden_states[layer_idx][j, :audio_feature_length].cpu().numpy()
                    audio_results.append(layer_features)
                results.append(audio_results)
        
        # Clear GPU cache after each batch to prevent memory accumulation
        clear_gpu_cache()
    
    return results

def load_and_resample_audio_optimized(wav_path, sr_target=16000):
    """Optimized audio loading with GPU acceleration - memory efficient for RTX 4070"""
    waveform, sr = torchaudio.load(wav_path)
    
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if sr != sr_target:
        # Use GPU for resampling if available
        if torch.cuda.is_available():
            waveform = waveform.cuda()
            waveform = torchaudio.functional.resample(waveform, sr, sr_target)
            waveform = waveform.cpu()
            # Clear GPU cache after resampling
            clear_gpu_cache()
        else:
            waveform = torchaudio.functional.resample(waveform, sr, sr_target)
    
    return waveform.squeeze(0).numpy(), sr_target

def process_single_file(args):
    """Process a single file - used for multiprocessing"""
    fname, audio_dir, features_dir, sr_target, sr_feature, selected_layers = args
    
    session_id = os.path.splitext(fname)[0]
    wav_path = os.path.join(audio_dir, fname)
    session_dir = os.path.join(features_dir, session_id)
    wav2vec_dir = os.path.join(session_dir, "wav2vec2")
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(wav2vec_dir, exist_ok=True)
    
    try:
        # Load audio
        y, sr = load_and_resample_audio_optimized(wav_path, sr_target)
        
        # Calculate target frames
        duration = len(y) / sr
        target_frames = int(duration * sr_feature)
        
        # Extract acoustic features using GPU
        acoustic_extractor = OptimizedAcousticExtractor()
        features = acoustic_extractor.extract_all_features_batch([y], sr, target_frames)[0]
        
        # Save acoustic features
        for feature_name, feature_data in features.items():
            if feature_name != 'combined':  # Save individual features
                np.save(os.path.join(session_dir, f"{feature_name}.npy"), feature_data)
        
        # Extract Wav2Vec2 features
        wav2vec_features = extract_wav2vec2_features_batch([y], [sr], selected_layers, sr_target, batch_size=1)[0]
        
        # Save Wav2Vec2 features
        for actual_layer_idx, layer_features in zip(selected_layers, wav2vec_features):
            np.save(os.path.join(wav2vec_dir, f"wav2vec2_layer{actual_layer_idx}.npy"), layer_features)
        
        return f"Success: {session_id}"
        
    except Exception as e:
        return f"Failed on {session_id}: {e}"

def generate_feature_dir_optimized(audio_dir, features_dir, sr_target=16000, sr_feature=20, 
                                 max_files=None, selected_layers=(4,), 
                                 n_processes=None, batch_processing=False):
    """
    Optimized feature generation with multiprocessing and GPU batching options
    Optimized for RTX 4070 with memory management
    
    Args:
        batch_processing: If True, process multiple files in GPU batches (better for many small files)
                         If False, use multiprocessing (better for fewer large files)
    """
    os.makedirs(features_dir, exist_ok=True)
    
    # Initialize GPU and clear cache
    if torch.cuda.is_available():
        clear_gpu_cache()
        print(f"GPU Memory Usage: {get_memory_usage()}")
    
    if max_files is not None:
        filenames = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])[:max_files]
    else:
        filenames = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    
    if n_processes is None:
        n_processes = min(mp.cpu_count() // 2, 4)  # Conservative for GPU sharing
    
    if batch_processing and len(filenames) > 10:
        # Batch processing mode - better for many small files
        process_files_in_batches(filenames, audio_dir, features_dir, sr_target, sr_feature, selected_layers)
    else:
        # Multiprocessing mode - better for fewer large files
        args_list = [
            (fname, audio_dir, features_dir, sr_target, sr_feature, selected_layers)
            for fname in filenames
        ]
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            results = list(tqdm(executor.map(process_single_file, args_list), 
                              total=len(args_list), desc="Processing files"))
        
        # Print results
        for result in results:
            print(result)

def process_files_in_batches(filenames, audio_dir, features_dir, sr_target, sr_feature, selected_layers):
    """Process files in GPU batches for maximum efficiency - optimized for RTX 4070"""
    batch_size = get_optimal_batch_size()
    print(f"Using batch size: {batch_size} for RTX 4070")
    initialize_wav2vec2_model()
    acoustic_extractor = OptimizedAcousticExtractor()
    
    for i in tqdm(range(0, len(filenames), batch_size), desc="Processing batches"):
        batch_files = filenames[i:i + batch_size]
        
        # Monitor memory usage every few batches
        if i % (batch_size * 5) == 0 and torch.cuda.is_available():
            mem_info = get_memory_usage()
            print(f"Batch {i//batch_size}: GPU Memory - {mem_info['allocated_gb']:.2f}GB allocated, {mem_info['free_gb']:.2f}GB free")
        
        # Load batch of audio files
        audio_batch = []
        sr_batch = []
        session_ids = []
        target_frames_batch = []
        
        for fname in batch_files:
            session_id = os.path.splitext(fname)[0]
            wav_path = os.path.join(audio_dir, fname)
            session_ids.append(session_id)
            
            try:
                y, sr = load_and_resample_audio_optimized(wav_path, sr_target)
                duration = len(y) / sr
                target_frames = int(duration * sr_feature)
                
                audio_batch.append(y)
                sr_batch.append(sr)
                target_frames_batch.append(target_frames)
                
            except Exception as e:
                print(f"Failed to load {session_id}: {e}")
                continue
        
        if not audio_batch:
            continue
        
        # Process acoustic features in batch
        try:
            # For now, process individually due to different target_frames
            # Could be optimized further by grouping by similar target_frames
            for j, (audio, sr, target_frames, session_id) in enumerate(zip(audio_batch, sr_batch, target_frames_batch, session_ids)):
                session_dir = os.path.join(features_dir, session_id)
                wav2vec_dir = os.path.join(session_dir, "wav2vec2")
                os.makedirs(session_dir, exist_ok=True)
                os.makedirs(wav2vec_dir, exist_ok=True)
                
                # Acoustic features
                features = acoustic_extractor.extract_all_features_batch([audio], sr, target_frames)[0]
                for feature_name, feature_data in features.items():
                    if feature_name != 'combined':
                        np.save(os.path.join(session_dir, f"{feature_name}.npy"), feature_data)
            
            # Process Wav2Vec2 in batch
            wav2vec_results = extract_wav2vec2_features_batch(audio_batch, sr_batch, selected_layers, sr_target)
            
            # Save Wav2Vec2 results
            for session_id, wav2vec_features in zip(session_ids, wav2vec_results):
                wav2vec_dir = os.path.join(features_dir, session_id, "wav2vec2")
                for actual_layer_idx, layer_features in zip(selected_layers, wav2vec_features):
                    np.save(os.path.join(wav2vec_dir, f"wav2vec2_layer{actual_layer_idx}.npy"), layer_features)
            
        except Exception as e:
            print(f"Batch processing failed: {e}")
            continue

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    print("开始特征提取 (RTX 4070优化)...")
    
    # Check GPU availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 检查输入目录
    audio_dir = "data/segmented_audio"
    features_dir = "data/features"
    
    if not os.path.exists(audio_dir):
        print(f"错误: 音频目录 {audio_dir} 不存在")
        print("请先运行数据清洗和分段步骤")
        exit(1)
    
    print(f"从 {audio_dir} 提取特征到 {features_dir}")
    
    generate_feature_dir_optimized(
        audio_dir=audio_dir,
        features_dir=features_dir, 
        sr_target=16000,
        sr_feature=20,
        max_files=None,  # 处理所有文件
        selected_layers=(4, 12),  # 提取第4层和第12层的特征
        n_processes=4,  # 减少进程数以避免GPU内存冲突
        batch_processing=True  # 使用批处理模式
    )