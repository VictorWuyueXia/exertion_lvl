import os
import shutil
import pandas as pd
import wave
from tqdm import tqdm

# Path configuration - use correct relative paths
metadata_path = "data/general_information.csv"
audio_folder = "data/audio"
resp_folder = "data/respiration_belt" if os.path.exists("data/respiration_belt") else None
manual_cleaned_folder = "data/manually_cleaned_data" if os.path.exists("data/manually_cleaned_data") else None

# Output directories - place under data directory
cleaned_audio_folder = "data/cleaned_audio"
cleaned_resp_folder = "data/cleaned_respiration_belt"

os.makedirs(cleaned_audio_folder, exist_ok=True)
if resp_folder:
    os.makedirs(cleaned_resp_folder, exist_ok=True)

# Check if metadata file exists
if not os.path.exists(metadata_path):
    print(f"WARNING: Metadata file {metadata_path} does not exist, skipping data cleaning step")
    print("Will directly copy audio files to cleaned_audio directory")
    
    # Directly copy all audio files
    if os.path.exists(audio_folder):
        audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]
        for fname in tqdm(audio_files, desc="Copying audio files"):
            src_audio = os.path.join(audio_folder, fname)
            dst_audio = os.path.join(cleaned_audio_folder, fname)
            shutil.copy2(src_audio, dst_audio)
    else:
        print(f"ERROR: Audio directory {audio_folder} does not exist")
        exit(1)
    
    exit(0)

# Load metadata
print("Loading metadata...")
metadata = pd.read_csv(metadata_path)
metadata["Duration"] = None
metadata["Removed"] = 0  # Initialize removal flag

def get_wav_duration(filepath):
    with wave.open(filepath, 'rb') as f:
        return f.getnframes() / f.getframerate()

print("Applying exclusion criteria...")
# Apply exclusion criteria
for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing sessions"):
    session = row["Session Name"]
    audio_path = os.path.join(audio_folder, f"{session}.wav")
    
    if os.path.exists(audio_path):
        duration = get_wav_duration(audio_path)
        metadata.at[idx, "Duration"] = duration
        
        if duration is not None and duration < 10:
            metadata.at[idx, "Removed"] = 1
        
        if "Non-fluency in English" in metadata.columns and row["Non-fluency in English"] == 1:
            metadata.at[idx, "Removed"] = 1
        
        if "clip_2" in session:
            metadata.at[idx, "Removed"] = 1
    else:
        # Skip files that don't exist without warning
        metadata.at[idx, "Removed"] = 1

print("Saving cleaned data...")
# Save clean data to new folders       
included_sessions = []
included_durations = []

for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Copying files"):
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

    # Copy respiration if available
    if resp_folder:
        src_resp = os.path.join(resp_folder, f"{session}.csv")
        dst_resp = os.path.join(cleaned_resp_folder, f"{session}.csv")
        if os.path.exists(src_resp):
            shutil.copy2(src_resp, dst_resp)

# Overwrite with manually cleaned data if available
if manual_cleaned_folder and os.path.exists(manual_cleaned_folder):
    print("Applying manually cleaned data...")
    manual_files = [f for f in os.listdir(manual_cleaned_folder) 
                   if f.endswith((".wav", ".csv"))]
    
    for fname in tqdm(manual_files, desc="Applying manual cleaning"):
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

# Save metadata (only keep essential)
filtered_metadata = metadata[metadata["Removed"] != 1].reset_index(drop=True)

# Save cleaned metadata
cleaned_metadata_path = "data/metadata_cleaned.csv"
filtered_metadata.to_csv(cleaned_metadata_path, index=False)

print(f"Data cleaning completed!")
print(f"Total sessions: {len(metadata)}")
print(f"Retained sessions: {len(filtered_metadata)}")
print(f"Removed sessions: {len(metadata) - len(filtered_metadata)}")
print(f"Cleaned audio files saved in: {cleaned_audio_folder}") 