import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# 添加WandB管理器
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.wandb_manager import WandBManager

class ExertionTrainer:
    """
    运动强度检测模型训练器
    支持5折交叉验证、混合精度训练和RTX4070优化
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建结果目录
        self.result_dir = self._create_result_dir()
        
        # 训练历史
        self.train_history = []
        self.val_history = []
        
        # 最佳模型
        self.best_model = None
        self.best_score = 0.0
        
        # 初始化WandB管理器
        self.wandb_manager = WandBManager(config)
        
        # 设置随机种子
        torch.manual_seed(config.get('seed', 42))
        np.random.seed(config.get('seed', 42))
        
        print(f"训练器初始化完成，设备: {self.device}")
        print(f"结果保存目录: {self.result_dir}")
    
    def _create_result_dir(self):
        """创建结果保存目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"VGG16_wav2vec2_layer{self.config.get('wav2vec2_layer', 4)}"
        result_dir = os.path.join("result", f"{model_name}_{timestamp}")
        
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(os.path.join(result_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(result_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(result_dir, "logs"), exist_ok=True)
        
        return result_dir
    
    def _save_config(self):
        """保存训练配置"""
        config_path = os.path.join(self.result_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def _get_optimizer(self, model):
        """获取优化器"""
        optimizer_name = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name.lower() == 'adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    def _get_scheduler(self, optimizer):
        """获取学习率调度器"""
        scheduler_name = self.config.get('scheduler', 'cosine')
        epochs = self.config.get('epochs', 100)
        
        if scheduler_name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_name.lower() == 'step':
            step_size = self.config.get('step_size', 30)
            gamma = self.config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name.lower() == 'plateau':
            patience = self.config.get('patience', 10)
            factor = self.config.get('factor', 0.5)
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
        else:
            return None
    
    def _get_criterion(self):
        """获取损失函数"""
        criterion_name = self.config.get('criterion', 'cross_entropy')
        
        if criterion_name.lower() == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif criterion_name.lower() == 'focal':
            alpha = self.config.get('focal_alpha', 1.0)
            gamma = self.config.get('focal_gamma', 2.0)
            return FocalLoss(alpha=alpha, gamma=gamma)
        else:
            raise ValueError(f"不支持的损失函数: {criterion_name}")
    
    def train_epoch(self, model, train_loader, optimizer, criterion, scaler=None):
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc="训练中")
        for batch_idx, batch in enumerate(pbar):
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
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(mfb=mfb, mfcc=mfcc, wav2vec2=wav2vec2)
                    loss = criterion(outputs, labels)
                
                # 反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(mfb=mfb, mfcc=mfcc, wav2vec2=wav2vec2)
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
            
            # 统计
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, model, val_loader, criterion):
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="验证中")
            for batch_idx, batch in enumerate(pbar):
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
                loss = criterion(outputs, labels)
                
                # 统计
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy, all_probs, all_preds, all_labels
    
    def train_fold(self, fold_idx, train_loader, val_loader, model, config):
        """训练一个fold"""
        print(f"\n开始训练 Fold {fold_idx + 1}")
        
        # 初始化
        optimizer = self._get_optimizer(model)
        scheduler = self._get_scheduler(optimizer)
        criterion = self._get_criterion()
        
        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler() if self.config.get('use_amp', True) else None
        
        # 记录模型信息到WandB
        self.wandb_manager.log_model_info(model)
        
        # 训练历史
        fold_train_history = []
        fold_val_history = []
        best_val_acc = 0.0
        patience_counter = 0
        patience = self.config.get('patience', 20)
        
        for epoch in range(config['epochs']):
            print(f"\nEpoch {epoch + 1}/{config['epochs']}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, scaler
            )
            
            # 验证
            val_loss, val_acc, val_probs, val_preds, val_labels = self.validate_epoch(
                model, val_loader, criterion
            )
            
            # 学习率调度
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # 记录指标到WandB
            global_step = epoch + fold_idx * config['epochs']
            self.wandb_manager.log_training_metrics({
                'loss': train_loss,
                'accuracy': train_acc
            }, step=global_step)
            
            self.wandb_manager.log_validation_metrics({
                'loss': val_loss,
                'accuracy': val_acc
            }, step=global_step)
            
            self.wandb_manager.log_learning_rate(current_lr, step=global_step)
            
            # 记录梯度信息
            self.wandb_manager.log_gradients(model, step=global_step)
            
            # 记录预测结果（每10个epoch记录一次）
            if epoch % 10 == 0:
                self.wandb_manager.log_predictions(val_labels, val_preds, epoch)
            
            # 记录历史
            fold_train_history.append({
                'epoch': epoch,
                'loss': train_loss,
                'accuracy': train_acc
            })
            
            fold_val_history.append({
                'epoch': epoch,
                'loss': val_loss,
                'accuracy': val_acc,
                'probs': val_probs,
                'preds': val_preds,
                'labels': val_labels
            })
            
            print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"学习率: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # 保存模型
                model_path = os.path.join(self.result_dir, "models", f"fold_{fold_idx + 1}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': config
                }, model_path)
                
                # 保存到WandB
                self.wandb_manager.save_model(model, model_path, {
                    'fold': fold_idx + 1,
                    'epoch': epoch,
                    'val_acc': val_acc
                })
                
                print(f"保存最佳模型，验证准确率: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                print(f"早停触发，{patience}个epoch没有改善")
                break
        
        return fold_train_history, fold_val_history, best_val_acc
    
    def cross_validation_train(self, dataset, config):
        """5折交叉验证训练"""
        print("开始5折交叉验证训练")
        
        # 准备数据
        all_sessions = dataset.metadata_df['session'].tolist()
        all_labels = []
        
        for session in all_sessions:
            label_row = dataset.labels_df[dataset.labels_df['segment_id'] == session]
            if not label_row.empty:
                all_labels.append(label_row.iloc[0]['exertion_level'])
            else:
                all_labels.append(0)  # 默认标签
        
        # 5折交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.get('seed', 42))
        
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_sessions, all_labels)):
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx + 1}/5")
            print(f"{'='*50}")
            
            # 创建数据加载器
            train_sessions = [all_sessions[i] for i in train_idx]
            val_sessions = [all_sessions[i] for i in val_idx]
            
            train_loader = self._create_dataloader(dataset, train_sessions, config, shuffle=True)
            val_loader = self._create_dataloader(dataset, val_sessions, config, shuffle=False)
            
            # 创建模型
            model = self._create_model(config)
            model = model.to(self.device)
            
            # 训练
            train_history, val_history, best_acc = self.train_fold(
                fold_idx, train_loader, val_loader, model, config
            )
            
            fold_results.append({
                'fold': fold_idx + 1,
                'train_history': train_history,
                'val_history': val_history,
                'best_acc': best_acc
            })
            
            print(f"Fold {fold_idx + 1} 完成，最佳验证准确率: {best_acc:.4f}")
        
        # 保存结果
        self._save_cv_results(fold_results)
        
        # 记录最终结果到WandB
        self._log_final_results(fold_results)
        
        return fold_results
    
    def finish(self):
        """结束训练器，清理资源"""
        if hasattr(self, 'wandb_manager'):
            self.wandb_manager.finish()
        print("训练器已结束")
    
    def _create_dataloader(self, dataset, session_ids, config, shuffle=True):
        """创建数据加载器"""
        from src.data.loader import get_dataloader_from_sessions
        
        return get_dataloader_from_sessions(
            session_ids=session_ids,
            meta_df=dataset.metadata_df,
            feature_dir=config['feature_dir'],
            labels_df=dataset.labels_df,
            use_acoustic=config.get('use_acoustic', True),
            use_mfb=config.get('use_mfb', True),
            use_embed=config.get('use_wav2vec2', True),
            selected_wav2vec2_layers=config.get('wav2vec2_layers', [4]),
            batch_size=config.get('batch_size', 8),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
    
    def _create_model(self, config):
        """创建模型"""
        from src.models.vgg16_exertion import create_model
        return create_model(config)
    
    def _save_cv_results(self, fold_results):
        """保存交叉验证结果"""
        # 保存训练历史
        results_path = os.path.join(self.result_dir, "cv_results.json")
        
        # 转换numpy数组为列表以便JSON序列化
        serializable_results = []
        for fold_result in fold_results:
            serializable_fold = {
                'fold': fold_result['fold'],
                'best_acc': fold_result['best_acc'],
                'train_history': fold_result['train_history'],
                'val_history': []
            }
            
            for val_epoch in fold_result['val_history']:
                serializable_val_epoch = {
                    'epoch': val_epoch['epoch'],
                    'loss': val_epoch['loss'],
                    'accuracy': val_epoch['accuracy'],
                    'probs': [prob.tolist() for prob in val_epoch['probs']],
                    'preds': val_epoch['preds'],
                    'labels': val_epoch['labels']
                }
                serializable_fold['val_history'].append(serializable_val_epoch)
            
            serializable_results.append(serializable_fold)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 计算平均结果
        avg_acc = np.mean([fold['best_acc'] for fold in fold_results])
        std_acc = np.std([fold['best_acc'] for fold in fold_results])
        
        print(f"\n交叉验证结果:")
        print(f"平均准确率: {avg_acc:.4f} ± {std_acc:.4f}")
        
        # 保存汇总结果
        summary = {
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc,
            'fold_accuracies': [fold['best_acc'] for fold in fold_results]
        }
        
        summary_path = os.path.join(self.result_dir, "cv_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _log_final_results(self, fold_results):
        """记录最终结果到WandB"""
        if not hasattr(self, 'wandb_manager') or not self.wandb_manager.enabled:
            return
        
        try:
            # 计算统计信息
            accuracies = [fold['best_acc'] for fold in fold_results]
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            
            # 记录最终测试指标
            self.wandb_manager.log_test_metrics({
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'best_accuracy': max(accuracies),
                'worst_accuracy': min(accuracies)
            })
            
            # 记录每个fold的结果
            for fold_result in fold_results:
                fold_idx = fold_result['fold']
                best_acc = fold_result['best_acc']
                
                self.wandb_manager.log_test_metrics({
                    f'fold_{fold_idx}_accuracy': best_acc
                })
            
            print(f"最终结果已记录到WandB")
            
        except Exception as e:
            print(f"记录最终结果到WandB时出错: {e}")


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() 