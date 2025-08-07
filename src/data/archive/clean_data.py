import os
import shutil
import pandas as pd
import wave

# 修改路径为相对路径，适用于当前环境
metadata_path = "data/general_information.csv"
audio_folder = "data/audio"
resp_folder = "data/respiration_belt" if os.path.exists("data/respiration_belt") else None
gt_label_folder = "data/gt_label" if os.path.exists("data/gt_label") else None
manual_cleaned_folder = "data/manually_cleaned_data" if os.path.exists("data/manually_cleaned_data") else None

# 输出目录
cleaned_audio_folder = "cleaned_audio"
cleaned_resp_folder = "cleaned_respiration_belt"
cleaned_gt_label = "cleaned_gt_label"

os.makedirs(cleaned_audio_folder, exist_ok=True)
if resp_folder:
    os.makedirs(cleaned_resp_folder, exist_ok=True)
if gt_label_folder:
    os.makedirs(cleaned_gt_label, exist_ok=True)

# 检查元数据文件是否存在
if not os.path.exists(metadata_path):
    print(f"警告: 元数据文件 {metadata_path} 不存在，跳过数据清洗步骤")
    print("将直接复制音频文件到cleaned_audio目录")
    
    # 直接复制所有音频文件
    if os.path.exists(audio_folder):
        for fname in os.listdir(audio_folder):
            if fname.endswith(".wav"):
                src_audio = os.path.join(audio_folder, fname)
                dst_audio = os.path.join(cleaned_audio_folder, fname)
                shutil.copy2(src_audio, dst_audio)
                print(f"复制音频文件: {fname}")
    else:
        print(f"错误: 音频目录 {audio_folder} 不存在")
    
    exit(0)

# Load metadata
print("加载元数据...")
metadata = pd.read_csv(metadata_path)
metadata["Duration"] = None
metadata["Removed"] = 0  # 初始化移除标记

def get_wav_duration(filepath):
    try:
        with wave.open(filepath, 'rb') as f:
            return f.getnframes() / f.getframerate()
    except:
        return None

print("应用排除标准...")
# Apply exclusion criteria
for idx, row in metadata.iterrows():
    session = row["Session Name"]
    audio_path = os.path.join(audio_folder, f"{session}.wav")
    
    if os.path.exists(audio_path):
        duration = get_wav_duration(audio_path)
        metadata.at[idx, "Duration"] = duration
        
        if duration is not None and duration < 10:
            metadata.at[idx, "Removed"] = 1
            print(f"移除短音频: {session} (时长: {duration:.2f}s)")
        
        if "Non-fluency in English" in metadata.columns and row["Non-fluency in English"] == 1:
            metadata.at[idx, "Removed"] = 1
            print(f"移除非英语流利: {session}")
        
        if "clip_2" in session:
            metadata.at[idx, "Removed"] = 1
            print(f"移除clip_2: {session}")
    else:
        print(f"警告: 音频文件不存在: {audio_path}")

print("保存清洗后的数据...")
# Save clean data to new folders       
included_sessions = []
included_durations = []

for _, row in metadata.iterrows():
    if row["Removed"] == 1:
        continue
    
    session = row["Session Name"]
    included_sessions.append(session)
    included_durations.append(row["Duration"])

    # Copy audio
    src_audio = os.path.join(audio_folder, f"{session}.wav")
    dst_audio = os.path.join(cleaned_audio_folder, f"{session}.wav")
    if os.path.exists(src_audio):
        shutil.copy2(src_audio, dst_audio)
        print(f"复制音频: {session}")

    # Copy respiration if available
    if resp_folder:
        src_resp = os.path.join(resp_folder, f"{session}.csv")
        dst_resp = os.path.join(cleaned_resp_folder, f"{session}.csv")
        if os.path.exists(src_resp):
            shutil.copy2(src_resp, dst_resp)
            print(f"复制呼吸数据: {session}")

# Overwrite with manually cleaned data if available
if manual_cleaned_folder and os.path.exists(manual_cleaned_folder):
    print("应用手动清洗的数据...")
    for fname in os.listdir(manual_cleaned_folder):
        if not (fname.endswith(".wav") or fname.endswith(".csv")):
            continue

        src = os.path.join(manual_cleaned_folder, fname)

        # Remove '_cropped' if present in the filename (before extension)
        base_name, ext = os.path.splitext(fname)
        if base_name.endswith("_cropped"):
            base_name = base_name.replace("_cropped", "")

        # Decide destination folder
        if ext == ".wav":
            dst = os.path.join(cleaned_audio_folder, base_name + ext)
        elif ext == ".csv":
            dst = os.path.join(cleaned_resp_folder, base_name + ext)

        shutil.copy2(src, dst)
        print(f"应用手动清洗: {fname}")

# Save metadata
metadata.to_csv("updated_metadata.csv", index=False)
filtered_metadata = metadata[metadata["Removed"] != 1].reset_index(drop=True)
filtered_metadata.to_csv("metadata_cleaned.csv", index=False)

print(f"数据清洗完成!")
print(f"总会话数: {len(metadata)}")
print(f"保留会话数: {len(filtered_metadata)}")
print(f"移除会话数: {len(metadata) - len(filtered_metadata)}")

# Copy ground truth labels if available
if gt_label_folder:
    print("复制地面真值标签...")
    missing = []
    for session in filtered_metadata["Session Name"]:
        src = os.path.join(gt_label_folder, f"{session}.txt")
        dst = os.path.join(cleaned_gt_label, f"{session}.txt")
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            missing.append(session)
    
    if missing:
        print(f"警告: 缺少标签文件: {missing}")
    else:
        print("所有标签文件已复制")



