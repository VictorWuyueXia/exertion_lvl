import os
import torch
import numpy as np
import torchaudio
from tqdm import tqdm
import soundfile as sf

# --- Config ---
audio_dir = "data/cleaned_audio"
out_audio_dir = "data/segmented_audio"

os.makedirs(out_audio_dir, exist_ok=True)

sr_target = 16000  # audio
window_sec = 15.0
stride_sec = 1.0

# --- Helper ---
def segment_indices(total_len, window_len, stride_len):
    return [(start, min(start + window_len, total_len))
            for start in range(0, total_len, stride_len)
            if start + window_len <= total_len]

# --- Main ---
if __name__ == "__main__":
    print("开始音频分段...")
    
    if not os.path.exists(audio_dir):
        print(f"错误: 音频目录 {audio_dir} 不存在")
        exit(1)
    
    segment_counts = []
    processed_count = 0
    skipped_count = 0

    for fname in tqdm(sorted(os.listdir(audio_dir)), desc="处理音频文件"):
        if not fname.endswith(".wav"):
            continue

        session_id = os.path.splitext(fname)[0]
        audio_path = os.path.join(audio_dir, fname)

        # Load audio
        try:
            y, sr = sf.read(audio_path)
            if sr != sr_target:
                y = torchaudio.functional.resample(torch.tensor(y), orig_freq=sr, new_freq=sr_target).numpy()
            audio_len = len(y)
        except Exception as e:
            print(f"加载音频文件失败 {session_id}: {e}")
            skipped_count += 1
            continue

        # Compute segment indices
        audio_window = int(window_sec * sr_target)
        audio_stride = int(stride_sec * sr_target)

        audio_segments = segment_indices(audio_len, audio_window, audio_stride)
        
        num_segments = len(audio_segments)
        segment_counts.append({"session_id": session_id, "num_segments": num_segments})

        # 保存分段
        for i, (a_start, a_end) in enumerate(audio_segments, start=1):
            seg_name = f"{session_id}_stride_{i}"
            
            try:
                # 保存音频分段
                audio_segment = y[a_start:a_end]
                sf.write(os.path.join(out_audio_dir, f"{seg_name}.wav"), audio_segment, sr_target)
                
            except Exception as e:
                print(f"保存分段失败 {seg_name}: {e}")
                continue
        
        processed_count += 1
        print(f"处理完成: {session_id} - {num_segments} 个分段")

    print(f"\n分段处理完成!")
    print(f"处理成功: {processed_count} 个文件")
    print(f"跳过: {skipped_count} 个文件")
    print(f"总分段数: {sum([seg['num_segments'] for seg in segment_counts])}")
    print(f"分段音频保存在: {out_audio_dir}") 