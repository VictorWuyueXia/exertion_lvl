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

# Set font for plots
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class ExertionEvaluator:
    """
    Exertion level detection model evaluator
    Supports model evaluation, confusion matrix visualization and AUC threshold optimization
    """
    
    def __init__(self, result_dir, config):
        self.result_dir = result_dir
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create evaluation result directory
        self.eval_dir = os.path.join(result_dir, "evaluation")
        os.makedirs(self.eval_dir, exist_ok=True)
        
        print(f"Evaluator initialized on device: {self.device}")
        print(f"Evaluation results directory: {self.eval_dir}")
    
    def evaluate_model(self, model, test_loader, fold_idx=None):
        """
        Evaluate model performance
        Args:
            model: Trained model
            test_loader: Test data loader
            fold_idx: Fold index (for cross-validation)
        """
        print(f"Starting model evaluation{f' (Fold {fold_idx})' if fold_idx is not None else ''}")
        
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Get data
                mfb = batch.get('mfb', None)
                mfcc = batch.get('mfcc', None)
                wav2vec2 = batch.get('wav2vec2', None)
                labels = batch['exertion_level'].to(self.device)
                
                # Move data to device
                if mfb is not None:
                    mfb = mfb.to(self.device)
                if mfcc is not None:
                    mfcc = mfcc.to(self.device)
                if wav2vec2 is not None:
                    wav2vec2 = wav2vec2.to(self.device)
                
                # Forward pass
                outputs = model(mfb=mfb, mfcc=mfcc, wav2vec2=wav2vec2)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                # Collect results
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate evaluation metrics
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        # Save results
        if fold_idx is not None:
            result_path = os.path.join(self.eval_dir, f"fold_{fold_idx}_evaluation.json")
        else:
            result_path = os.path.join(self.eval_dir, "test_evaluation.json")
        
        self._save_evaluation_results(metrics, all_labels, all_preds, all_probs, result_path)
        
        # Generate visualizations
        self._generate_visualizations(all_labels, all_preds, all_probs, fold_idx)
        
        return metrics, all_labels, all_preds, all_probs
    
    def _calculate_metrics(self, labels, preds, probs):
        """Calculate evaluation metrics"""
        # Convert inputs to numpy arrays if needed
        if isinstance(labels, (list, tuple)):
            labels = np.array(labels)
        if isinstance(preds, (list, tuple)):
            preds = np.array(preds)
        if isinstance(probs, (list, tuple)):
            probs = np.array(probs)
        
        # Basic metrics
        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro')
        f1_weighted = f1_score(labels, preds, average='weighted')
        precision_macro = precision_score(labels, preds, average='macro')
        recall_macro = recall_score(labels, preds, average='macro')
        
        # Per-class metrics
        f1_per_class = f1_score(labels, preds, average=None)
        precision_per_class = precision_score(labels, preds, average=None)
        recall_per_class = recall_score(labels, preds, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        # ROC AUC (one-vs-rest)
        try:
            auc = roc_auc_score(labels, probs, multi_class='ovr')
        except:
            auc = 0.0
        
        # Per-class AUC
        auc_per_class = []
        if len(probs.shape) > 1:  # Check if probs is 2D
            for i in range(probs.shape[1]):
                try:
                    class_auc = roc_auc_score((labels == i).astype(int), probs[:, i])
                    auc_per_class.append(class_auc)
                except:
                    auc_per_class.append(0.0)
        else:
            # If probs is 1D, create dummy AUC values
            auc_per_class = [0.0] * len(np.unique(labels))
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_per_class': f1_per_class.tolist(),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'auc': auc,
            'auc_per_class': auc_per_class,
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def _save_evaluation_results(self, metrics, labels, preds, probs, result_path):
        """Save evaluation results to file"""
        results = {
            'metrics': metrics,
            'predictions': preds.tolist(),
            'probabilities': probs.tolist(),
            'labels': labels.tolist()
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation results saved to: {result_path}")
    
    def _generate_visualizations(self, labels, preds, probs, fold_idx=None):
        """Generate evaluation visualizations"""
        suffix = f"_fold_{fold_idx}" if fold_idx is not None else ""
        
        # Confusion matrix
        self._plot_confusion_matrix(labels, preds, fold_idx)
        
        # ROC curves
        self._plot_multiclass_roc(labels, probs, fold_idx)
        
        # Prediction distribution
        self._plot_prediction_distribution(labels, preds, fold_idx)
        
        # Class accuracy
        self._plot_class_accuracy(labels, preds, fold_idx)
    
    def _plot_confusion_matrix(self, labels, preds, fold_idx=None):
        """Plot confusion matrix"""
        suffix = f"_fold_{fold_idx}" if fold_idx is not None else ""
        
        cm = confusion_matrix(labels, preds)
        
        # Check if confusion matrix is empty
        if cm.size == 0:
            print(f"Warning: Empty confusion matrix for fold {fold_idx}")
            return
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(5), yticklabels=range(5))
        plt.title(f'Confusion Matrix{suffix}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        save_path = os.path.join(self.eval_dir, f"confusion_matrix{suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_multiclass_roc(self, labels, probs, fold_idx=None):
        """Plot multiclass ROC curves"""
        suffix = f"_fold_{fold_idx}" if fold_idx is not None else ""
        
        # Check if probs is valid
        if len(probs.shape) < 2 or probs.shape[1] == 0:
            print(f"Warning: Invalid probability array for ROC plot in fold {fold_idx}")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i in range(probs.shape[1]):
            try:
                fpr, tpr, _ = roc_curve((labels == i).astype(int), probs[:, i])
                auc_score = roc_auc_score((labels == i).astype(int), probs[:, i])
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.3f})')
            except:
                print(f"Warning: Could not plot ROC curve for class {i}")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Multiclass ROC Curves{suffix}')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.eval_dir, f"roc_curves{suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_distribution(self, labels, preds, fold_idx=None):
        """Plot prediction distribution"""
        suffix = f"_fold_{fold_idx}" if fold_idx is not None else ""
        
        plt.figure(figsize=(12, 5))
        
        # Actual vs predicted distribution
        plt.subplot(1, 2, 1)
        plt.hist(labels, bins=5, alpha=0.7, label='Actual', color='blue')
        plt.hist(preds, bins=5, alpha=0.7, label='Predicted', color='red')
        plt.xlabel('Exertion Level')
        plt.ylabel('Count')
        plt.title(f'Actual vs Predicted Distribution{suffix}')
        plt.legend()
        
        # Prediction accuracy by class
        plt.subplot(1, 2, 2)
        class_accuracies = []
        for i in range(5):
            mask = labels == i
            if mask.sum() > 0:
                acc = (preds[mask] == labels[mask]).mean()
                class_accuracies.append(acc)
            else:
                class_accuracies.append(0.0)
        
        plt.bar(range(5), class_accuracies)
        plt.xlabel('Exertion Level')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy by Class{suffix}')
        plt.ylim(0, 1)
        
        save_path = os.path.join(self.eval_dir, f"prediction_distribution{suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_accuracy(self, labels, preds, fold_idx=None):
        """Plot class-wise accuracy"""
        suffix = f"_fold_{fold_idx}" if fold_idx is not None else ""
        
        # Calculate metrics for each class
        classes = range(5)
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        
        for i in classes:
            mask = labels == i
            if mask.sum() > 0:
                acc = (preds[mask] == labels[mask]).mean()
                # 对于多分类，使用micro平均
                f1 = f1_score(labels[mask], preds[mask], average='micro')
                prec = precision_score(labels[mask], preds[mask], average='micro', zero_division=0)
                rec = recall_score(labels[mask], preds[mask], average='micro', zero_division=0)
            else:
                acc = f1 = prec = rec = 0.0
            
            accuracies.append(acc)
            f1_scores.append(f1)
            precisions.append(prec)
            recalls.append(rec)
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(classes))
        width = 0.2
        
        plt.bar(x - width*1.5, accuracies, width, label='Accuracy', alpha=0.8)
        plt.bar(x - width*0.5, f1_scores, width, label='F1 Score', alpha=0.8)
        plt.bar(x + width*0.5, precisions, width, label='Precision', alpha=0.8)
        plt.bar(x + width*1.5, recalls, width, label='Recall', alpha=0.8)
        
        plt.xlabel('Exertion Level')
        plt.ylabel('Score')
        plt.title(f'Class-wise Performance Metrics{suffix}')
        plt.xticks(x, classes)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        save_path = os.path.join(self.eval_dir, f"class_accuracy{suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_cross_validation(self, cv_results):
        """Evaluate cross-validation results"""
        print("Evaluating cross-validation results")
        
        # Collect metrics from all folds
        fold_metrics = []
        all_labels = []
        all_preds = []
        all_probs = []
        
        for fold_result in cv_results:
            fold_idx = fold_result['fold_idx']
            
            # Load fold evaluation results
            eval_path = os.path.join(self.eval_dir, f"fold_{fold_idx+1}_evaluation.json")
            
            if os.path.exists(eval_path):
                with open(eval_path, 'r', encoding='utf-8') as f:
                    fold_data = json.load(f)
                
                # Calculate metrics from saved data
                fold_metrics_data = self._calculate_metrics(
                    np.array(fold_data['labels']), 
                    np.array(fold_data['predictions']), 
                    np.array(fold_data['probabilities'])
                )
                fold_metrics.append(fold_metrics_data)
                all_labels.extend(fold_data['labels'])
                all_preds.extend(fold_data['predictions'])
                all_probs.extend(fold_data['probabilities'])
        
        # Calculate overall metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        overall_metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        # Calculate fold statistics
        fold_accuracies = [m['accuracy'] for m in fold_metrics]
        fold_f1_scores = [m['f1_macro'] for m in fold_metrics]
        
        overall_results = {
            'avg_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies),
            'avg_f1': np.mean(fold_f1_scores),
            'std_f1': np.std(fold_f1_scores),
            'fold_accuracies': fold_accuracies,
            'fold_f1_scores': fold_f1_scores,
            'overall_metrics': overall_metrics
        }
        
        # Save overall results
        overall_path = os.path.join(self.eval_dir, "overall_evaluation.json")
        with open(overall_path, 'w', encoding='utf-8') as f:
            json.dump(overall_results, f, indent=2, ensure_ascii=False)
        
        # Generate overall visualizations
        self._generate_visualizations(all_labels, all_preds, all_probs)
        
        print("Cross-validation evaluation completed")
        return overall_results 