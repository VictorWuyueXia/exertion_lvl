#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级评估指标
实现三种新的评估方法：
1. 百分比混淆矩阵
2. 容差准确率（预测与真实值相差<=1算正确）
3. 多数投票（前后clip投票）
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import defaultdict
import json
from tqdm import tqdm

class AdvancedMetricsEvaluator:
    """高级评估指标计算器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def load_test_data(self, labels_file, features_dir):
        """加载测试数据"""
        # 读取标签
        df = pd.read_csv(labels_file)
        test_df = df[df['split'] == 'test'].copy()
        test_df = test_df.reset_index(drop=True)
        
        # 加载特征
        test_features = []
        test_labels = []
        test_clip_ids = []
        
        for _, row in tqdm(test_df.iterrows(), desc="Loading test features"):
            clip_id = row['clip_id']
            feature_path = os.path.join(features_dir, 'test', f"{clip_id}.npz")
            
            if os.path.exists(feature_path):
                features = np.load(feature_path)
                mfcc = features['mfcc']
                wav2vec2 = features['wav2vec2']
                
                # 转换为模型输入格式
                # MFCC: [T, F] -> [B, C, F, T] = [1, 1, 40, T]
                mfcc_tensor = torch.FloatTensor(mfcc.T[np.newaxis, np.newaxis, :, :])  # (1, 1, 40, T)
                # wav2vec2: [T, D] -> [B, T, D] = [1, T, 768]
                wav2vec2_tensor = torch.FloatTensor(wav2vec2[np.newaxis, :, :])  # (1, T, 768)
                
                test_features.append((mfcc_tensor, wav2vec2_tensor))
                test_labels.append(row['exertion_level'] - 1)  # 转换为0-4
                test_clip_ids.append(clip_id)
        
        return test_features, test_labels, test_clip_ids, test_df
    
    def predict_single(self, mfcc_tensor, wav2vec2_tensor):
        """单个样本预测"""
        with torch.no_grad():
            mfcc_tensor = mfcc_tensor.to(self.device)
            wav2vec2_tensor = wav2vec2_tensor.to(self.device)
            
            logits, _ = self.model(mfcc_tensor, wav2vec2_tensor)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            
            return pred
    
    def metric_1_percentage_confusion_matrix(self, test_features, test_labels):
        """指标1: 百分比混淆矩阵"""
        print("计算百分比混淆矩阵...")
        
        predictions = []
        for mfcc_tensor, wav2vec2_tensor in tqdm(test_features, desc="Predicting"):
            pred = self.predict_single(mfcc_tensor, wav2vec2_tensor)
            predictions.append(pred)
        
        # 计算混淆矩阵
        cm = confusion_matrix(test_labels, predictions)
        
        # 转换为百分比
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm_percentage = np.nan_to_num(cm_percentage, nan=0.0)
        
        # 绘制百分比混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=range(5), yticklabels=range(5))
        plt.title('Confusion Matrix (Percentage)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('percentage_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算准确率
        accuracy = accuracy_score(test_labels, predictions)
        
        return {
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_percentage': cm_percentage.tolist(),
            'predictions': [int(p) for p in predictions]
        }
    
    def metric_2_tolerance_accuracy(self, test_features, test_labels):
        """指标2: 容差准确率（预测与真实值相差<=1算正确）"""
        print("计算容差准确率...")
        
        predictions = []
        for mfcc_tensor, wav2vec2_tensor in tqdm(test_features, desc="Predicting"):
            pred = self.predict_single(mfcc_tensor, wav2vec2_tensor)
            predictions.append(pred)
        
        # 计算容差准确率
        tolerance_correct = 0
        for true_label, pred_label in zip(test_labels, predictions):
            if abs(true_label - pred_label) <= 1:
                tolerance_correct += 1
        
        tolerance_accuracy = tolerance_correct / len(test_labels)
        
        # 计算不同容差级别的准确率
        tolerance_levels = [0, 1, 2, 3, 4]
        tolerance_accuracies = {}
        
        for tolerance in tolerance_levels:
            correct = sum(1 for true, pred in zip(test_labels, predictions) 
                         if abs(true - pred) <= tolerance)
            tolerance_accuracies[f'tolerance_{tolerance}'] = correct / len(test_labels)
        
        return {
            'tolerance_accuracy': float(tolerance_accuracy),
            'tolerance_accuracies': {k: float(v) for k, v in tolerance_accuracies.items()},
            'predictions': [int(p) for p in predictions]
        }
    
    def metric_3_majority_voting(self, test_features, test_labels, test_clip_ids, test_df):
        """指标3: 多数投票（前后clip投票）"""
        print("计算多数投票...")
        
        # 首先获取所有预测
        all_predictions = []
        for mfcc_tensor, wav2vec2_tensor in tqdm(test_features, desc="Predicting all clips"):
            pred = self.predict_single(mfcc_tensor, wav2vec2_tensor)
            all_predictions.append(pred)
        
        # 创建clip_id到索引的映射
        clip_id_to_idx = {clip_id: idx for idx, clip_id in enumerate(test_clip_ids)}
        
        # 为每个session创建clip序列
        session_clips = defaultdict(list)
        for idx, (clip_id, pred) in enumerate(zip(test_clip_ids, all_predictions)):
            session_id = test_df.iloc[idx]['session_id']
            session_clips[session_id].append((clip_id, pred, idx))
        
        # 对每个session的clip按名称排序
        for session_id in session_clips:
            session_clips[session_id].sort(key=lambda x: x[0])
        
        # 多数投票
        majority_predictions = []
        majority_labels = []
        
        for session_id, clips in session_clips.items():
            for i, (clip_id, pred, idx) in enumerate(clips):
                # 获取前后clip的预测
                votes = [pred]  # 当前clip的预测
                
                # 前一个clip
                if i > 0:
                    prev_pred = clips[i-1][1]
                    votes.append(prev_pred)
                
                # 后一个clip
                if i < len(clips) - 1:
                    next_pred = clips[i+1][1]
                    votes.append(next_pred)
                
                # 多数投票
                majority_pred = max(set(votes), key=votes.count)
                majority_predictions.append(majority_pred)
                majority_labels.append(test_labels[idx])
        
        # 计算准确率
        majority_accuracy = accuracy_score(majority_labels, majority_predictions)
        
        # 计算混淆矩阵
        cm = confusion_matrix(majority_labels, majority_predictions)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(5), yticklabels=range(5))
        plt.title('Majority Voting Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('majority_voting_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'majority_accuracy': float(majority_accuracy),
            'confusion_matrix': cm.tolist(),
            'majority_predictions': [int(p) for p in majority_predictions],
            'majority_labels': [int(l) for l in majority_labels]
        }
    
    def run_all_metrics(self, labels_file, features_dir):
        """运行所有指标"""
        print("开始高级评估...")
        
        # 加载数据
        test_features, test_labels, test_clip_ids, test_df = self.load_test_data(labels_file, features_dir)
        
        # 指标1: 百分比混淆矩阵
        print("\n=== 指标1: 百分比混淆矩阵 ===")
        result1 = self.metric_1_percentage_confusion_matrix(test_features, test_labels)
        print(f"准确率: {result1['accuracy']:.4f}")
        
        # 指标2: 容差准确率
        print("\n=== 指标2: 容差准确率 ===")
        result2 = self.metric_2_tolerance_accuracy(test_features, test_labels)
        print(f"容差准确率 (≤1): {result2['tolerance_accuracy']:.4f}")
        for tolerance, acc in result2['tolerance_accuracies'].items():
            print(f"{tolerance}: {acc:.4f}")
        
        # 指标3: 多数投票
        print("\n=== 指标3: 多数投票 ===")
        result3 = self.metric_3_majority_voting(test_features, test_labels, test_clip_ids, test_df)
        print(f"多数投票准确率: {result3['majority_accuracy']:.4f}")
        
        # 保存结果
        all_results = {
            'metric_1_percentage_confusion_matrix': result1,
            'metric_2_tolerance_accuracy': result2,
            'metric_3_majority_voting': result3
        }
        
        with open('advanced_metrics_results.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: advanced_metrics_results.json")
        
        return all_results

def main():
    """主函数"""
    import sys
    sys.path.append('src')
    
    from models.vgg16_exertion import DualChannelExertionModel
    from utils.config_manager import ConfigManager
    
    # 加载配置
    config = ConfigManager('config/vgg16_wav2vec2_layer4_mfcc_exertion.yaml')
    
    # 创建模型
    model = DualChannelExertionModel(
        mfcc_dim=config.get('model.mfcc_dim', 40),
        wav2vec2_dim=config.get('model.wav2vec2_dim', 768),
        hidden_dim=config.get('model.hidden_dim', 256),
        num_classes=config.get('model.num_classes', 5),
        in_ch=config.get('model.in_ch', 1)
    )
    
    # 加载最佳模型
    checkpoint_path = 'checkpoints/VGG16_wav2vec2_layer4_exertion_detection_coral_v2_best.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        print(f"已加载模型: {checkpoint_path}")
    else:
        print(f"模型文件不存在: {checkpoint_path}")
        return
    
    # 创建评估器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    evaluator = AdvancedMetricsEvaluator(model, device)
    
    # 运行评估
    labels_file = 'data/processed/04_final/labels.csv'
    features_dir = 'data/processed/04_final/features'
    
    results = evaluator.run_all_metrics(labels_file, features_dir)
    
    # 打印总结
    print("\n" + "="*50)
    print("评估总结:")
    print("="*50)
    print(f"1. 标准准确率: {results['metric_1_percentage_confusion_matrix']['accuracy']:.4f}")
    print(f"2. 容差准确率 (≤1): {results['metric_2_tolerance_accuracy']['tolerance_accuracy']:.4f}")
    print(f"3. 多数投票准确率: {results['metric_3_majority_voting']['majority_accuracy']:.4f}")

if __name__ == "__main__":
    main()
