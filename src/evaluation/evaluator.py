import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ExertionEvaluator:
    """
    运动强度检测模型评估器
    支持模型评估、混淆矩阵可视化和AUC阈值优化
    """
    
    def __init__(self, result_dir, config):
        self.result_dir = result_dir
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建评估结果目录
        self.eval_dir = os.path.join(result_dir, "evaluation")
        os.makedirs(self.eval_dir, exist_ok=True)
        
        print(f"评估器初始化完成，设备: {self.device}")
        print(f"评估结果保存目录: {self.eval_dir}")
    
    def evaluate_model(self, model, test_loader, fold_idx=None):
        """
        评估模型性能
        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器
            fold_idx: fold索引（用于交叉验证）
        """
        print(f"开始评估模型{f' (Fold {fold_idx})' if fold_idx is not None else ''}")
        
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                # 获取数据
                mfb = batch.get('mfb', None)
                mfcc = batch.get('mfcc', None)
                wav2vec2 = batch.get('wav2vec2', None)
                labels = batch['exertion_level'].to(self.device)
                
                # 移动数据到设备
                if mfb is not None:
                    mfb = mfb.to(self.device)
                if mfcc is not None:
                    mfcc = mfcc.to(self.device)
                if wav2vec2 is not None:
                    wav2vec2 = wav2vec2.to(self.device)
                
                # 前向传播
                outputs = model(mfb=mfb, mfcc=mfcc, wav2vec2=wav2vec2)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                # 收集结果
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 转换为numpy数组
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 计算评估指标
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        # 保存结果
        if fold_idx is not None:
            result_path = os.path.join(self.eval_dir, f"fold_{fold_idx}_evaluation.json")
        else:
            result_path = os.path.join(self.eval_dir, "test_evaluation.json")
        
        self._save_evaluation_results(metrics, all_labels, all_preds, all_probs, result_path)
        
        # 生成可视化
        self._generate_visualizations(all_labels, all_preds, all_probs, fold_idx)
        
        return metrics, all_labels, all_preds, all_probs
    
    def _calculate_metrics(self, labels, preds, probs):
        """计算评估指标"""
        # 基础分类指标
        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro')
        f1_weighted = f1_score(labels, preds, average='weighted')
        precision_macro = precision_score(labels, preds, average='macro')
        recall_macro = recall_score(labels, preds, average='macro')
        
        # 每个类别的指标
        class_metrics = {}
        for i in range(self.config.get('num_classes', 5)):
            if i in labels:
                class_metrics[f'class_{i}'] = {
                    'precision': precision_score(labels, preds, average=None)[i],
                    'recall': recall_score(labels, preds, average=None)[i],
                    'f1': f1_score(labels, preds, average=None)[i]
                }
        
        # AUC指标（多分类）
        try:
            if probs.shape[1] == 2:  # 二分类
                auc = roc_auc_score(labels, probs[:, 1])
            else:  # 多分类
                auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        except:
            auc = None
        
        # 混淆矩阵
        cm = confusion_matrix(labels, preds)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'class_metrics': class_metrics,
            'classification_report': classification_report(labels, preds, output_dict=True)
        }
        
        return metrics
    
    def _save_evaluation_results(self, metrics, labels, preds, probs, result_path):
        """保存评估结果"""
        results = {
            'metrics': metrics,
            'predictions': preds.tolist(),
            'probabilities': probs.tolist(),
            'true_labels': labels.tolist()
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"评估结果已保存到: {result_path}")
    
    def _generate_visualizations(self, labels, preds, probs, fold_idx=None):
        """生成可视化图表"""
        # 混淆矩阵
        self._plot_confusion_matrix(labels, preds, fold_idx)
        
        # ROC曲线
        if probs.shape[1] > 2:
            self._plot_multiclass_roc(labels, probs, fold_idx)
        else:
            self._plot_binary_roc(labels, probs, fold_idx)
        
        # 预测分布
        self._plot_prediction_distribution(labels, preds, fold_idx)
        
        # 类别准确率
        self._plot_class_accuracy(labels, preds, fold_idx)
    
    def _plot_confusion_matrix(self, labels, preds, fold_idx=None):
        """绘制混淆矩阵"""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(self.config.get('num_classes', 5)),
                   yticklabels=range(self.config.get('num_classes', 5)))
        plt.title(f'混淆矩阵{f" (Fold {fold_idx})" if fold_idx is not None else ""}')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        if fold_idx is not None:
            save_path = os.path.join(self.eval_dir, f"fold_{fold_idx}_confusion_matrix.png")
        else:
            save_path = os.path.join(self.eval_dir, "test_confusion_matrix.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"混淆矩阵已保存到: {save_path}")
    
    def _plot_multiclass_roc(self, labels, probs, fold_idx=None):
        """绘制多分类ROC曲线"""
        n_classes = probs.shape[1]
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            # 计算每个类别的ROC
            fpr, tpr, _ = roc_curve((labels == i).astype(int), probs[:, i])
            auc = roc_auc_score((labels == i).astype(int), probs[:, i])
            
            plt.plot(fpr, tpr, label=f'类别 {i} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
        plt.xlabel('假正率 (FPR)')
        plt.ylabel('真正率 (TPR)')
        plt.title(f'多分类ROC曲线{f" (Fold {fold_idx})" if fold_idx is not None else ""}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if fold_idx is not None:
            save_path = os.path.join(self.eval_dir, f"fold_{fold_idx}_roc_curve.png")
        else:
            save_path = os.path.join(self.eval_dir, "test_roc_curve.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC曲线已保存到: {save_path}")
    
    def _plot_binary_roc(self, labels, probs, fold_idx=None):
        """绘制二分类ROC曲线"""
        fpr, tpr, thresholds = roc_curve(labels, probs[:, 1])
        auc = roc_auc_score(labels, probs[:, 1])
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
        plt.xlabel('假正率 (FPR)')
        plt.ylabel('真正率 (TPR)')
        plt.title(f'二分类ROC曲线{f" (Fold {fold_idx})" if fold_idx is not None else ""}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if fold_idx is not None:
            save_path = os.path.join(self.eval_dir, f"fold_{fold_idx}_roc_curve.png")
        else:
            save_path = os.path.join(self.eval_dir, "test_roc_curve.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC曲线已保存到: {save_path}")
    
    def _plot_prediction_distribution(self, labels, preds, fold_idx=None):
        """绘制预测分布"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 真实标签分布
        unique_labels, counts = np.unique(labels, return_counts=True)
        ax1.bar(unique_labels, counts, alpha=0.7, label='真实标签')
        ax1.set_xlabel('运动强度等级')
        ax1.set_ylabel('样本数量')
        ax1.set_title('真实标签分布')
        ax1.legend()
        
        # 预测标签分布
        unique_preds, pred_counts = np.unique(preds, return_counts=True)
        ax2.bar(unique_preds, pred_counts, alpha=0.7, color='orange', label='预测标签')
        ax2.set_xlabel('运动强度等级')
        ax2.set_ylabel('样本数量')
        ax2.set_title('预测标签分布')
        ax2.legend()
        
        plt.suptitle(f'标签分布对比{f" (Fold {fold_idx})" if fold_idx is not None else ""}')
        
        if fold_idx is not None:
            save_path = os.path.join(self.eval_dir, f"fold_{fold_idx}_prediction_distribution.png")
        else:
            save_path = os.path.join(self.eval_dir, "test_prediction_distribution.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"预测分布图已保存到: {save_path}")
    
    def _plot_class_accuracy(self, labels, preds, fold_idx=None):
        """绘制每个类别的准确率"""
        cm = confusion_matrix(labels, preds)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(class_accuracy)), class_accuracy, alpha=0.7)
        
        # 在柱状图上添加数值
        for i, (bar, acc) in enumerate(zip(bars, class_accuracy)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xlabel('运动强度等级')
        plt.ylabel('准确率')
        plt.title(f'各类别准确率{f" (Fold {fold_idx})" if fold_idx is not None else ""}')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        if fold_idx is not None:
            save_path = os.path.join(self.eval_dir, f"fold_{fold_idx}_class_accuracy.png")
        else:
            save_path = os.path.join(self.eval_dir, "test_class_accuracy.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"类别准确率图已保存到: {save_path}")
    
    def optimize_auc_threshold(self, labels, probs, fold_idx=None):
        """
        优化AUC阈值
        Args:
            labels: 真实标签
            probs: 预测概率
            fold_idx: fold索引
        """
        print(f"开始优化AUC阈值{f' (Fold {fold_idx})' if fold_idx is not None else ''}")
        
        n_classes = probs.shape[1]
        optimal_thresholds = {}
        threshold_metrics = {}
        
        for i in range(n_classes):
            # 二分类问题：当前类别 vs 其他类别
            binary_labels = (labels == i).astype(int)
            binary_probs = probs[:, i]
            
            # 计算不同阈值下的指标
            thresholds = np.arange(0.1, 0.9, 0.01)
            metrics_at_thresholds = []
            
            for threshold in thresholds:
                binary_preds = (binary_probs >= threshold).astype(int)
                
                # 计算指标
                accuracy = accuracy_score(binary_labels, binary_preds)
                precision = precision_score(binary_labels, binary_preds, zero_division=0)
                recall = recall_score(binary_labels, binary_preds, zero_division=0)
                f1 = f1_score(binary_labels, binary_preds, zero_division=0)
                
                metrics_at_thresholds.append({
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
            
            # 找到最优阈值（基于F1分数）
            best_idx = np.argmax([m['f1'] for m in metrics_at_thresholds])
            optimal_threshold = metrics_at_thresholds[best_idx]['threshold']
            
            optimal_thresholds[f'class_{i}'] = optimal_threshold
            threshold_metrics[f'class_{i}'] = metrics_at_thresholds
        
        # 绘制阈值优化图
        self._plot_threshold_optimization(threshold_metrics, fold_idx)
        
        # 保存结果
        threshold_results = {
            'optimal_thresholds': optimal_thresholds,
            'threshold_metrics': threshold_metrics
        }
        
        if fold_idx is not None:
            save_path = os.path.join(self.eval_dir, f"fold_{fold_idx}_threshold_optimization.json")
        else:
            save_path = os.path.join(self.eval_dir, "test_threshold_optimization.json")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(threshold_results, f, indent=2, ensure_ascii=False)
        
        print(f"阈值优化结果已保存到: {save_path}")
        print(f"最优阈值: {optimal_thresholds}")
        
        return optimal_thresholds, threshold_metrics
    
    def _plot_threshold_optimization(self, threshold_metrics, fold_idx=None):
        """绘制阈值优化图"""
        n_classes = len(threshold_metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (class_name, metrics) in enumerate(threshold_metrics.items()):
            if i >= 4:  # 最多显示4个类别
                break
                
            thresholds = [m['threshold'] for m in metrics]
            f1_scores = [m['f1'] for m in metrics]
            precision_scores = [m['precision'] for m in metrics]
            recall_scores = [m['recall'] for m in metrics]
            
            ax = axes[i]
            ax.plot(thresholds, f1_scores, label='F1分数', linewidth=2)
            ax.plot(thresholds, precision_scores, label='精确率', linewidth=2)
            ax.plot(thresholds, recall_scores, label='召回率', linewidth=2)
            
            # 标记最优阈值
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
            
            ax.axvline(x=best_threshold, color='red', linestyle='--', 
                      label=f'最优阈值: {best_threshold:.2f}')
            ax.scatter(best_threshold, best_f1, color='red', s=100, zorder=5)
            
            ax.set_xlabel('阈值')
            ax.set_ylabel('分数')
            ax.set_title(f'{class_name} 阈值优化')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'阈值优化结果{f" (Fold {fold_idx})" if fold_idx is not None else ""}')
        
        if fold_idx is not None:
            save_path = os.path.join(self.eval_dir, f"fold_{fold_idx}_threshold_optimization.png")
        else:
            save_path = os.path.join(self.eval_dir, "test_threshold_optimization.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"阈值优化图已保存到: {save_path}")
    
    def evaluate_cross_validation(self, cv_results):
        """评估交叉验证结果"""
        print("开始评估交叉验证结果")
        
        # 汇总所有fold的结果
        all_labels = []
        all_preds = []
        all_probs = []
        
        for fold_result in cv_results:
            # 获取最佳epoch的结果
            best_epoch_idx = np.argmax([epoch['accuracy'] for epoch in fold_result['val_history']])
            best_epoch = fold_result['val_history'][best_epoch_idx]
            
            all_labels.extend(best_epoch['labels'])
            all_preds.extend(best_epoch['preds'])
            all_probs.extend(best_epoch['probs'])
        
        # 转换为numpy数组
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # 计算整体指标
        overall_metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        # 生成整体可视化
        self._generate_visualizations(all_labels, all_preds, all_probs)
        
        # 优化阈值
        optimal_thresholds, threshold_metrics = self.optimize_auc_threshold(all_labels, all_probs)
        
        # 保存整体评估结果
        overall_results = {
            'overall_metrics': overall_metrics,
            'optimal_thresholds': optimal_thresholds,
            'fold_accuracies': [fold['best_acc'] for fold in cv_results],
            'avg_accuracy': np.mean([fold['best_acc'] for fold in cv_results]),
            'std_accuracy': np.std([fold['best_acc'] for fold in cv_results])
        }
        
        overall_path = os.path.join(self.eval_dir, "overall_cv_evaluation.json")
        with open(overall_path, 'w', encoding='utf-8') as f:
            json.dump(overall_results, f, indent=2, ensure_ascii=False)
        
        print(f"交叉验证整体评估结果已保存到: {overall_path}")
        print(f"平均准确率: {overall_results['avg_accuracy']:.4f} ± {overall_results['std_accuracy']:.4f}")
        
        return overall_results 