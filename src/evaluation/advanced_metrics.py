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
    
    def __init__(self, model, device='cuda', result_dir=None):
        self.model = model
        self.device = device
        self.result_dir = result_dir
        self.model.eval()
        
    def load_test_data(self, labels_file, features_dir):
        """加载测试数据 - 使用与训练器相同的数据加载方式"""
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
                
                # 提取特征 - 使用与DataLoader相同的处理方式
                mfcc_features = features['mfcc'] if 'mfcc' in features else np.zeros((750, 40))
                wav2vec2_features = features['wav2vec2'] if 'wav2vec2' in features else np.zeros((750, 768))
                
                # 转换为与DataLoader相同的格式
                # MFCC: (T, F) -> (F, T) -> (1, F, T) for Conv2d
                mfcc_features = mfcc_features.T  # (40, T)
                mfcc_features = mfcc_features[np.newaxis, :, :]  # (1, 40, T)
                
                # wav2vec2: (T, D) -> (T, D) for Transformer (已经是正确格式)
                # 不需要转置，因为原始格式就是 (T, D)
                
                # 转换为tensor - 与DataLoader完全一致
                mfcc_tensor = torch.FloatTensor(mfcc_features)  # (1, 40, T)
                wav2vec2_tensor = torch.FloatTensor(wav2vec2_features)  # (T, 768)
                
                # 注意：模型期望 [B, C, F, T] 格式，所以需要添加batch维度
                # 但这里我们先保持与DataLoader一致的格式，在预测时再处理
                
                test_features.append((mfcc_tensor, wav2vec2_tensor))
                test_labels.append(row['exertion_level'] - 1)  # 转换为0-4
                test_clip_ids.append(clip_id)
        
        return test_features, test_labels, test_clip_ids, test_df
    
    def predict_single(self, mfcc_tensor, wav2vec2_tensor):
        """单个样本预测 - 使用与训练器相同的预测方式"""
        with torch.no_grad():
            # DataLoader已经提供了batch维度，直接使用
            mfcc_tensor = mfcc_tensor.to(self.device)
            wav2vec2_tensor = wav2vec2_tensor.to(self.device)
            
            # 使用与训练器相同的模型调用方式
            logits, _ = self.model(mfcc_tensor, wav2vec2_tensor)
            
            # 使用CORAL预测方式（与训练器一致）
            try:
                from src.models.dual_channel_exertion import coral_predict
                pred = coral_predict(logits).cpu().numpy()[0]
            except ImportError:
                # 如果无法导入CORAL，使用标准argmax
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
        
        # 使用self.result_dir或默认路径
        if self.result_dir:
            save_path = os.path.join(self.result_dir, 'percentage_confusion_matrix.png')
        else:
            save_path = 'percentage_confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
        # 使用self.result_dir或默认路径
        if self.result_dir:
            save_path = os.path.join(self.result_dir, 'majority_voting_confusion_matrix.png')
        else:
            save_path = 'majority_voting_confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'majority_accuracy': float(majority_accuracy),
            'confusion_matrix': cm.tolist(),
            'majority_predictions': [int(p) for p in majority_predictions],
            'majority_labels': [int(l) for l in majority_labels]
        }
    
    def run_all_metrics(self, labels_file, features_dir):
        """运行所有指标 - 直接使用训练器的测试结果"""
        print("开始高级评估...")
        
        # 直接加载训练器的测试结果
        result_dir = "results/experiments/VGG16_wav2vec2_layer4_exertion_detection_coral_v2_20250810_013231"
        test_results_file = os.path.join(result_dir, "test_results.json")
        
        if os.path.exists(test_results_file):
            with open(test_results_file, 'r', encoding='utf-8') as f:
                test_results = json.load(f)
            
            print("使用训练器的测试结果...")
            test_labels = test_results['labels']
            test_predictions = test_results['predictions']
            test_probabilities = test_results['probabilities']
            
            print(f"加载了 {len(test_labels)} 个测试样本")
            print(f"训练器准确率: {test_results['metrics']['acc']:.4f}")
            
            # 指标1: 百分比混淆矩阵
            print("\n=== 指标1: 百分比混淆矩阵 ===")
            result1 = self.metric_1_percentage_confusion_matrix_from_results(test_labels, test_predictions)
            print(f"准确率: {result1['accuracy']:.4f}")
            
            # 指标2: 容差准确率
            print("\n=== 指标2: 容差准确率 ===")
            result2 = self.metric_2_tolerance_accuracy_from_results(test_labels, test_predictions)
            print(f"容差准确率 (≤1): {result2['tolerance_accuracy']:.4f}")
            for tolerance, acc in result2['tolerance_accuracies'].items():
                print(f"{tolerance}: {acc:.4f}")
            
            # 指标3: 多数投票（使用session信息）
            print("\n=== 指标3: 多数投票（使用session信息） ===")
            result3 = self.metric_3_session_majority_voting(test_labels, test_predictions, labels_file)
            print(f"多数投票准确率: {result3['majority_accuracy']:.4f}")
            
            # 保存结果
            all_results = {
                'metric_1_percentage_confusion_matrix': result1,
                'metric_2_tolerance_accuracy': result2,
                'metric_3_majority_voting': result3
            }
            
            # 保存结果到规范的结果目录
            if hasattr(self, 'result_dir') and self.result_dir:
                result_file = os.path.join(self.result_dir, 'advanced_metrics_results.json')
            else:
                result_file = 'advanced_metrics_results.json'
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n结果已保存到: {result_file}")
            
            return all_results
        else:
            print(f"训练器测试结果文件不存在: {test_results_file}")
            return None
    
    def metric_1_percentage_confusion_matrix_from_results(self, test_labels, test_predictions):
        """指标1: 百分比混淆矩阵 - 从训练器结果计算"""
        print("计算百分比混淆矩阵...")
        
        # 计算混淆矩阵
        cm = confusion_matrix(test_labels, test_predictions)
        
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
        
        # 使用self.result_dir或默认路径
        if self.result_dir:
            save_path = os.path.join(self.result_dir, 'percentage_confusion_matrix.png')
        else:
            save_path = 'percentage_confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算准确率
        accuracy = accuracy_score(test_labels, test_predictions)
        
        return {
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_percentage': cm_percentage.tolist(),
            'predictions': [int(p) for p in test_predictions],
            'labels': [int(l) for l in test_labels]
        }
    
    def metric_2_tolerance_accuracy_from_results(self, test_labels, test_predictions):
        """指标2: 容差准确率 - 从训练器结果计算"""
        print("计算容差准确率...")
        
        # 计算容差准确率
        tolerance_correct = 0
        for true_label, pred_label in zip(test_labels, test_predictions):
            if abs(true_label - pred_label) <= 1:
                tolerance_correct += 1
        
        tolerance_accuracy = tolerance_correct / len(test_labels)
        
        # 计算不同容差级别的准确率
        tolerance_levels = [0, 1, 2, 3, 4]
        tolerance_accuracies = {}
        
        for tolerance in tolerance_levels:
            correct = sum(1 for true, pred in zip(test_labels, test_predictions) 
                         if abs(true - pred) <= tolerance)
            tolerance_accuracies[f'tolerance_{tolerance}'] = correct / len(test_labels)
        
        return {
            'tolerance_accuracy': float(tolerance_accuracy),
            'tolerance_accuracies': {k: float(v) for k, v in tolerance_accuracies.items()},
            'predictions': [int(p) for p in test_predictions]
        }
    
    def metric_3_simplified_majority_voting_from_results(self, test_labels, test_predictions):
        """简化的多数投票 - 从训练器结果计算"""
        print("计算简化多数投票...")
        
        # 使用滑动窗口进行多数投票
        window_size = 3
        majority_predictions = []
        
        for i in range(len(test_predictions)):
            # 获取窗口内的预测
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(test_predictions), i + window_size // 2 + 1)
            window_predictions = test_predictions[start_idx:end_idx]
            
            # 多数投票
            majority_pred = max(set(window_predictions), key=window_predictions.count)
            majority_predictions.append(majority_pred)
        
        # 计算准确率
        majority_accuracy = accuracy_score(test_labels, majority_predictions)
        
        # 计算混淆矩阵
        cm = confusion_matrix(test_labels, majority_predictions)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(5), yticklabels=range(5))
        plt.title('Simplified Majority Voting Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 使用self.result_dir或默认路径
        if self.result_dir:
            save_path = os.path.join(self.result_dir, 'majority_voting_confusion_matrix.png')
        else:
            save_path = 'majority_voting_confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'majority_accuracy': float(majority_accuracy),
            'confusion_matrix': cm.tolist(),
            'majority_predictions': [int(p) for p in majority_predictions],
            'majority_labels': [int(l) for l in test_labels]
        }

    def metric_3_session_majority_voting(self, test_labels, test_predictions, labels_file):
        """指标3: 多数投票（使用session信息）"""
        print("计算session多数投票...")
        
        # 加载原始标签文件获取session信息
        import pandas as pd
        df = pd.read_csv(labels_file)
        test_df = df[df['split'] == 'test'].copy()
        test_df = test_df.reset_index(drop=True)
        
        print(f"测试集clip数量: {len(test_df)}")
        print(f"测试集session数量: {test_df['session_id'].nunique()}")
        
        # 确保数据顺序一致
        if len(test_df) != len(test_labels):
            print(f"警告: 数据长度不匹配 - 标签文件: {len(test_df)}, 预测结果: {len(test_labels)}")
            return self.metric_3_simplified_majority_voting_from_results(test_labels, test_predictions)
        
        # 为每个session创建clip序列
        from collections import defaultdict
        session_clips = defaultdict(list)
        
        for idx, (clip_id, session_id) in enumerate(zip(test_df['clip_id'], test_df['session_id'])):
            pred = test_predictions[idx]
            label = test_labels[idx]
            session_clips[session_id].append((clip_id, pred, label, idx))
        
        # 对每个session的clip按名称排序
        for session_id in session_clips:
            session_clips[session_id].sort(key=lambda x: x[0])
        
        print(f"处理了 {len(session_clips)} 个session")
        
        # 多数投票
        majority_predictions = []
        majority_labels = []
        
        for session_id, clips in session_clips.items():
            if len(clips) < 2:
                # 如果session只有一个clip，直接使用原始预测
                clip_id, pred, label, idx = clips[0]
                majority_predictions.append(pred)
                majority_labels.append(label)
                continue
                
            for i, (clip_id, pred, label, idx) in enumerate(clips):
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
                majority_labels.append(label)
        
        # 计算准确率
        majority_accuracy = accuracy_score(majority_labels, majority_predictions)
        
        # 计算混淆矩阵
        cm = confusion_matrix(majority_labels, majority_predictions)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(5), yticklabels=range(5))
        plt.title('Session-based Majority Voting Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 使用self.result_dir或默认路径
        if self.result_dir:
            save_path = os.path.join(self.result_dir, 'majority_voting_confusion_matrix.png')
        else:
            save_path = 'majority_voting_confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'majority_accuracy': float(majority_accuracy),
            'confusion_matrix': cm.tolist(),
            'majority_predictions': [int(p) for p in majority_predictions],
            'majority_labels': [int(l) for l in majority_labels],
            'session_count': len(session_clips),
            'method': 'session_based'
        }

def main():
    """主函数"""
    import sys
    import json
    sys.path.append('src')
    
    from models.vgg16_exertion import DualChannelExertionModel
    from utils.config_manager import ConfigManager
    
    # 查找最佳模型信息
    best_model_info_path = 'results/models/best_model_info.json'
    if os.path.exists(best_model_info_path):
        with open(best_model_info_path, 'r', encoding='utf-8') as f:
            best_model_info = json.load(f)
        checkpoint_path = best_model_info['model_path']
        result_dir = best_model_info['result_dir']
        print(f"找到最佳模型: {checkpoint_path}")
        print(f"结果目录: {result_dir}")
    else:
        # 如果没有最佳模型信息，尝试查找最新的checkpoint
        checkpoint_path = 'checkpoints/VGG16_wav2vec2_layer4_exertion_detection_coral_v2_best.pt'
        result_dir = 'results'
        print(f"使用默认模型路径: {checkpoint_path}")
    
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
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            print(f"已加载模型状态: {checkpoint_path}")
            print(f"  训练轮数: {checkpoint.get('epoch', 'N/A')}")
            print(f"  最佳指标: {checkpoint.get('metrics', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)
            print(f"已加载模型权重: {checkpoint_path}")
        print(f"已加载模型: {checkpoint_path}")
    else:
        print(f"模型文件不存在: {checkpoint_path}")
        return
    
    # 创建评估器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 检查模型状态
    model.eval()
    print(f"模型设备: {next(model.parameters()).device}")
    print(f"模型训练模式: {model.training}")
    
    # 测试模型预测
    print("测试模型预测...")
    with torch.no_grad():
        # 创建测试输入
        test_mfcc = torch.randn(1, 1, 40, 100).to(device)
        test_wav2vec2 = torch.randn(1, 100, 768).to(device)
        
        logits, _ = model(test_mfcc, test_wav2vec2)
        print(f"  测试输入形状: MFCC {test_mfcc.shape}, wav2vec2 {test_wav2vec2.shape}")
        print(f"  输出logits形状: {logits.shape}")
        print(f"  输出logits值: {logits.cpu().numpy()}")
        
        # 使用CORAL预测
        try:
            from models.dual_channel_exertion import coral_predict
            pred = coral_predict(logits).cpu().numpy()[0]
            print(f"  CORAL预测: {pred}")
        except Exception as e:
            print(f"  CORAL预测失败: {e}")
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            print(f"  Argmax预测: {pred}")
    
    evaluator = AdvancedMetricsEvaluator(model, device, result_dir)
    
    # 运行评估
    labels_file = 'data/processed/04_final/labels.csv'
    features_dir = 'data/processed/04_final/features'
    
    results = evaluator.run_all_metrics(labels_file, features_dir)
    
    # 保存结果到规范的结果目录
    if os.path.exists(result_dir):
        advanced_metrics_file = os.path.join(result_dir, 'advanced_metrics_results.json')
        with open(advanced_metrics_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"高级评估结果已保存到: {advanced_metrics_file}")
    
    # 打印总结
    print("\n" + "="*50)
    print("评估总结:")
    print("="*50)
    print(f"1. 标准准确率: {results['metric_1_percentage_confusion_matrix']['accuracy']:.4f}")
    print(f"2. 容差准确率 (≤1): {results['metric_2_tolerance_accuracy']['tolerance_accuracy']:.4f}")
    print(f"3. 多数投票准确率: {results['metric_3_majority_voting']['majority_accuracy']:.4f}")

if __name__ == "__main__":
    main()
