#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化评估器
专注于基本的分类指标，避免复杂的AUC优化
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from datetime import datetime

class SimpleExertionEvaluator:
    """简化运动强度检测评估器"""
    
    def __init__(self, result_dir, device='cuda'):
        self.result_dir = result_dir
        self.device = device
        self.eval_dir = os.path.join(result_dir, "evaluation")
        os.makedirs(self.eval_dir, exist_ok=True)
        
        print(f"简化评估器初始化完成，设备: {device}")
        print(f"评估结果保存目录: {self.eval_dir}")
    
    def evaluate_cross_validation(self, cv_results):
        """评估交叉验证结果"""
        print("开始简化交叉验证评估...")
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        # 收集所有fold的结果
        for fold_result in cv_results:
            val_history = fold_result['val_history']
            
            # 找到最佳epoch（基于验证准确率）
            best_acc = 0.0
            best_epoch_idx = 0
            
            for i, val_epoch in enumerate(val_history):
                if val_epoch['accuracy'] > best_acc:
                    best_acc = val_epoch['accuracy']
                    best_epoch_idx = i
            
            best_val_result = val_history[best_epoch_idx]
            
            all_labels.extend(best_val_result['labels'])
            all_preds.extend(best_val_result['preds'])
            all_probs.extend(best_val_result['probs'])
        
        # 转换为numpy数组
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # 计算基本指标
        metrics = self._calculate_basic_metrics(all_labels, all_preds, all_probs)
        
        # 生成可视化
        self._generate_basic_visualizations(all_labels, all_preds, all_probs)
        
        # 保存结果
        self._save_basic_results(metrics, all_labels, all_preds, all_probs)
        
        return {
            'metrics': metrics,
            'labels': all_labels.tolist(),
            'preds': all_preds.tolist(),
            'probs': all_probs.tolist()
        }
    
    def _calculate_basic_metrics(self, labels, preds, probs):
        """计算基本分类指标"""
        print("计算基本分类指标...")
        
        # 基本准确率
        accuracy = accuracy_score(labels, preds)
        
        # 每个类别的指标
        precision = precision_score(labels, preds, average=None, zero_division=0)
        recall = recall_score(labels, preds, average=None, zero_division=0)
        f1 = f1_score(labels, preds, average=None, zero_division=0)
        
        # 宏平均和加权平均
        precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        
        precision_weighted = precision_score(labels, preds, average='weighted', zero_division=0)
        recall_weighted = recall_score(labels, preds, average='weighted', zero_division=0)
        f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
        
        # 类别分布
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        label_distribution = {int(label): int(count) for label, count in zip(unique_labels, label_counts)}
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'label_distribution': label_distribution
        }
        
        print(f"准确率: {accuracy:.4f}")
        print(f"F1宏平均: {f1_macro:.4f}")
        print(f"F1加权平均: {f1_weighted:.4f}")
        
        return metrics
    
    def _generate_basic_visualizations(self, labels, preds, probs):
        """生成基本可视化"""
        print("生成基本可视化...")
        
        # 混淆矩阵
        self._plot_confusion_matrix(labels, preds)
        
        # 类别准确率
        self._plot_class_accuracy(labels, preds)
        
        # 预测分布
        self._plot_prediction_distribution(labels, preds)
    
    def _plot_confusion_matrix(self, labels, preds):
        """绘制混淆矩阵"""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[f'Class {i}' for i in range(5)],
                   yticklabels=[f'Class {i}' for i in range(5)])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        save_path = os.path.join(self.eval_dir, "confusion_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"混淆矩阵已保存到: {save_path}")
    
    def _plot_class_accuracy(self, labels, preds):
        """绘制每个类别的准确率"""
        unique_labels = np.unique(labels)
        class_accuracies = []
        
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 0:
                class_acc = accuracy_score(labels[mask], preds[mask])
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(unique_labels)), class_accuracies)
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Class')
        plt.xticks(range(len(unique_labels)), [f'Class {i}' for i in unique_labels])
        
        # 在柱状图上添加数值
        for bar, acc in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.eval_dir, "class_accuracy.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"类别准确率图已保存到: {save_path}")
    
    def _plot_prediction_distribution(self, labels, preds):
        """绘制预测分布"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 实际标签分布
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        ax1.bar(unique_labels, label_counts, alpha=0.7, label='Actual')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_title('Actual Label Distribution')
        ax1.legend()
        
        # 预测标签分布
        unique_preds, pred_counts = np.unique(preds, return_counts=True)
        ax2.bar(unique_preds, pred_counts, alpha=0.7, color='orange', label='Predicted')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.set_title('Predicted Label Distribution')
        ax2.legend()
        
        plt.tight_layout()
        
        save_path = os.path.join(self.eval_dir, "prediction_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"预测分布图已保存到: {save_path}")
    
    def _save_basic_results(self, metrics, labels, preds, probs):
        """保存基本结果"""
        results = {
            'metrics': metrics,
            'labels': labels.tolist(),
            'preds': preds.tolist(),
            'probs': probs.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        save_path = os.path.join(self.eval_dir, "basic_evaluation_results.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"基本评估结果已保存到: {save_path}")
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(labels, preds, 
                                  target_names=[f'Class {i}' for i in range(5)]))
