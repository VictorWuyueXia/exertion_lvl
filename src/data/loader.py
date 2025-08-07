import os
import numpy as np
import pandas as pd
import soundfile as sf

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from set_seed import seed_worker

# --- 1. Metadata extraction ---
def get_session_metadata(feature_dir):
    """从特征目录中提取会话元数据"""
    session_metadata = []
    
    if not os.path.exists(feature_dir):
        print(f"特征目录不存在: {feature_dir}")
        return pd.DataFrame(session_metadata)

    for session_id in sorted(os.listdir(feature_dir)):
        session_dir = os.path.join(feature_dir, session_id)
        if not os.path.isdir(session_dir):
            continue
            
        # 解析会话ID获取信息
        parts = session_id.split("_")
        if len(parts) >= 4:
            participant_id = parts[1]
            speed = parts[2]
            task = parts[5] if len(parts) > 5 else "unknown"
            stride = parts[-1] if len(parts) > 1 else "1"
        else:
            participant_id = "unknown"
            speed = "unknown"
            task = "unknown"
            stride = "1"

        session_metadata.append({
            "session": session_id,
            "participant": participant_id,
            "speed": speed,
            "task": task,
            "stride": stride
        })
    
    return pd.DataFrame(session_metadata)

# --- 2. Dataset class ---
class AudioFeatureDataset(Dataset):
    def __init__(self, metadata_df, feature_dir, labels_df=None, use_acoustic=True, use_mfb=False, use_embed=False, selected_wav2vec2_layers=(4,)):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.feature_dir = feature_dir
        self.labels_df = labels_df
        self.use_acoustic = use_acoustic
        self.use_mfb = use_mfb
        self.use_embed = use_embed
        self.selected_wav2vec2_layers = selected_wav2vec2_layers    

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        session_id = row["session"]
        session_feature_dir = os.path.join(self.feature_dir, session_id)
        
        # Load features
        acoustic = mfb = None
        embeds = []
        
        if self.use_acoustic:
            acoustic_path = os.path.join(session_feature_dir, "acoustic.npy")
            if os.path.exists(acoustic_path):
                acoustic = np.load(acoustic_path)  # (T1, D1)
        
        if self.use_mfb:
            mfb_path = os.path.join(session_feature_dir, "mfb.npy")
            if os.path.exists(mfb_path):
                mfb = np.load(mfb_path)  # (T2, D2)
            
        if self.use_embed:
            if len(self.selected_wav2vec2_layers) > 1:
                embed_list = []
                for layer_id in self.selected_wav2vec2_layers:
                    layer_path = os.path.join(session_feature_dir, f"wav2vec2/wav2vec2_layer{layer_id}.npy")
                    if os.path.exists(layer_path):
                        embed_list.append(np.load(layer_path))  # each shape: (T, D)
                if embed_list:
                    embeds = np.concatenate(embed_list, axis=1)  # shape: (T, D1 + D2)
            else:
                layer_id = self.selected_wav2vec2_layers[0]
                layer_path = os.path.join(session_feature_dir, f"wav2vec2/wav2vec2_layer{layer_id}.npy")
                if os.path.exists(layer_path):
                    embeds = np.load(layer_path)

        # 获取exertion level标签
        exertion_level = None
        if self.labels_df is not None:
            label_row = self.labels_df[self.labels_df['segment_id'] == session_id]
            if not label_row.empty:
                exertion_level = label_row['exertion_level'].iloc[0]
        
        return {
            'session_id': session_id,
            'acoustic': acoustic,
            'mfb': mfb,
            'embeds': embeds,
            'exertion_level': exertion_level,
            'metadata': row.to_dict()
        }

def collate_multi_feature_batch(batch):
    """处理多特征批次的collate函数"""
    def to_tensor_and_pad(seqs):
        tensors = [torch.tensor(x, dtype=torch.float32) for x in seqs if x is not None]
        if not tensors:
            return None
        return pad_sequence(tensors, batch_first=True, padding_value=0.0)
    
    # 收集所有特征
    acoustic_features = [item['acoustic'] for item in batch if item['acoustic'] is not None]
    mfb_features = [item['mfb'] for item in batch if item['mfb'] is not None]
    embed_features = [item['embeds'] for item in batch if item['embeds'] is not None]
    
    # 转换为tensor并padding
    acoustic_tensor = to_tensor_and_pad(acoustic_features)
    mfb_tensor = to_tensor_and_pad(mfb_features)
    embed_tensor = to_tensor_and_pad(embed_features)
    
    # 收集标签
    exertion_levels = [item['exertion_level'] for item in batch if item['exertion_level'] is not None]
    exertion_tensor = torch.tensor(exertion_levels, dtype=torch.long) if exertion_levels else None
    
    # 收集元数据
    session_ids = [item['session_id'] for item in batch]
    metadata = [item['metadata'] for item in batch]
    
    return {
        'session_ids': session_ids,
        'acoustic': acoustic_tensor,
        'mfb': mfb_tensor,
        'embeds': embed_tensor,
        'exertion_levels': exertion_tensor,
        'metadata': metadata
    }

def greedy_grouped_split(df, n_folds=5, segment_duration=15.0):
    """贪心分组分割，确保同一参与者的数据不会同时出现在训练和测试集中"""
    participants = df['participant'].unique()
    np.random.shuffle(participants)
    
    folds = [[] for _ in range(n_folds)]
    current_fold = 0
    
    for participant in participants:
        participant_sessions = df[df['participant'] == participant].index.tolist()
        folds[current_fold].extend(participant_sessions)
        current_fold = (current_fold + 1) % n_folds
    
    return folds

def get_fixed_fold_split(folds, fold_idx):
    """获取指定fold的训练/测试分割"""
    test_indices = folds[fold_idx]
    train_indices = []
    for i, fold in enumerate(folds):
        if i != fold_idx:
            train_indices.extend(fold)
    
    return train_indices, test_indices

def get_dataloader_from_sessions(session_ids, meta_df, feature_dir, labels_df=None,
                                use_acoustic=True, use_mfb=False, use_embed=False, selected_wav2vec2_layers=(7,),
                                batch_size=4, shuffle=False, num_workers=6, pin_memory=True):
    """从会话ID列表创建数据加载器"""
    # 过滤元数据
    filtered_df = meta_df[meta_df['session'].isin(session_ids)].reset_index(drop=True)
    
    # 创建数据集
    dataset = AudioFeatureDataset(
        filtered_df, 
        feature_dir,
        labels_df=labels_df,
        use_acoustic=use_acoustic,
        use_mfb=use_mfb,
        use_embed=use_embed,
        selected_wav2vec2_layers=selected_wav2vec2_layers
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_multi_feature_batch,
        worker_init_fn=seed_worker
    )
    
    return dataloader

# 示例使用
if __name__ == "__main__":
    # 示例：加载特征数据
    feature_dir = "data/features"
    metadata_df = get_session_metadata(feature_dir)
    
    if len(metadata_df) > 0:
        print(f"找到 {len(metadata_df)} 个会话")
        print("前5个会话:")
        print(metadata_df.head())
        
        # 加载标签数据
        labels_path = "data/segment_exertion_labels.csv"
        labels_df = None
        if os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            print(f"加载标签数据: {len(labels_df)} 个标签")
        else:
            print("警告: 标签文件不存在，将不包含exertion level标签")
        
        # 创建数据集
        dataset = AudioFeatureDataset(
            metadata_df, 
            feature_dir,
            labels_df=labels_df,
            use_acoustic=True,
            use_embed=True,
            selected_wav2vec2_layers=(4, 12)
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 测试加载一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"样本会话ID: {sample['session_id']}")
            print(f"声学特征形状: {sample['acoustic'].shape if sample['acoustic'] is not None else 'None'}")
            print(f"嵌入特征形状: {sample['embeds'].shape if sample['embeds'] is not None else 'None'}")
            print(f"Exertion Level: {sample['exertion_level']}")
    else:
        print("未找到特征数据，请先运行特征提取") 