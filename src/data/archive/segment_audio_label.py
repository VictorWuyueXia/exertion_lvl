import os
import csv
import torch
import numpy as np
import torchaudio
from tqdm import tqdm
import soundfile as sf


# --- Config ---
audio_dir = "cleaned_audio"
label_dir = "samplewise_labels"
out_audio_dir = "segmented_audio"
out_label_dir = "segmented_labels"

os.makedirs(out_audio_dir, exist_ok=True)
os.makedirs(out_label_dir, exist_ok=True)

sr_target = 16000  # audio
label_sr = 20      # label
window_sec = 15.0
stride_sec = 1.0

# --- Helper ---
def segment_indices(total_len, window_len, stride_len):
    return [(start, min(start + window_len, total_len))
            for start in range(0, total_len, stride_len)
            if start + window_len <= total_len]

# --- Main ---
if __name__ == "__main__":
    print("开始音频和标签分段...")
    
    if not os.path.exists(audio_dir):
        print(f"错误: 音频目录 {audio_dir} 不存在")
        exit(1)
    
    if not os.path.exists(label_dir):
        print(f"错误: 标签目录 {label_dir} 不存在")
        exit(1)
    
    segment_counts = []
    processed_count = 0
    skipped_count = 0

    for fname in tqdm(sorted(os.listdir(audio_dir)), desc="处理音频文件"):
        if not fname.endswith(".wav"):
            continue

        session_id = os.path.splitext(fname)[0]
        audio_path = os.path.join(audio_dir, fname)
        label_path_npy = os.path.join(label_dir, session_id + ".npy")
        label_path_csv = os.path.join(label_dir, session_id + ".csv")

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

        # Load labels
        labels = None
        if os.path.exists(label_path_npy):
            try:
                labels = np.load(label_path_npy)
            except Exception as e:
                print(f"加载npy标签失败 {session_id}: {e}")
        elif os.path.exists(label_path_csv):
            try:
                labels = np.loadtxt(label_path_csv, delimiter=",")
            except Exception as e:
                print(f"加载csv标签失败 {session_id}: {e}")
        
        if labels is None:
            print(f"缺少标签文件: {session_id}")
            skipped_count += 1
            continue

        label_len = len(labels)
        
        # Unified length of audio and label
        audio_duration = len(y) / sr_target
        label_duration = len(labels) / label_sr
        final_duration = min(audio_duration, label_duration)

        # Compute sample counts to match
        max_audio_samples = int(final_duration * sr_target)
        max_label_samples = int(final_duration * label_sr)

        # Truncate both
        y = y[:max_audio_samples]
        labels = labels[:max_label_samples]

        audio_len = len(y)
        label_len = len(labels)
        
        # 验证长度匹配
        expected_label_len = round(audio_len / sr_target * label_sr)
        if abs(label_len - expected_label_len) > 1:  # 允许1个样本的误差
            print(f"长度不匹配: {session_id} - 音频={audio_len}, 标签={label_len}, 期望={expected_label_len}")
            skipped_count += 1
            continue

        # Compute segment indices
        audio_window = int(window_sec * sr_target)
        audio_stride = int(stride_sec * sr_target)
        label_window = int(window_sec * label_sr)
        label_stride = int(stride_sec * label_sr)

        audio_segments = segment_indices(audio_len, audio_window, audio_stride)
        label_segments = segment_indices(label_len, label_window, label_stride)
        
        num_segments = len(audio_segments)
        segment_counts.append({"session_id": session_id, "num_segments": num_segments})

        # 保存分段
        for i, ((a_start, a_end), (l_start, l_end)) in enumerate(zip(audio_segments, label_segments), start=1):
            seg_name = f"{session_id}_stride_{i}"
            
            try:
                # 保存音频分段
                audio_segment = y[a_start:a_end]
                sf.write(os.path.join(out_audio_dir, f"{seg_name}.wav"), audio_segment, sr_target)
                
                # 保存标签分段
                label_segment = labels[l_start:l_end]
                np.save(os.path.join(out_label_dir, f"{seg_name}_label.npy"), label_segment)
                
            except Exception as e:
                print(f"保存分段失败 {seg_name}: {e}")
                continue
        
        processed_count += 1
        print(f"处理完成: {session_id} - {num_segments} 个分段")

    print(f"\n分段处理完成!")
    print(f"处理成功: {processed_count} 个文件")
    print(f"跳过: {skipped_count} 个文件")
    print(f"总分段数: {sum([seg['num_segments'] for seg in segment_counts])}")
    
    # 保存分段统计信息
    if segment_counts:
        import pandas as pd
        df = pd.DataFrame(segment_counts)
        df.to_csv("segment_statistics.csv", index=False)
        print("分段统计信息已保存到 segment_statistics.csv")


