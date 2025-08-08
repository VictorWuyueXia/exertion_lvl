import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        # 确保学习率和权重衰减是数值类型
        if isinstance(lr, str):
            try:
                lr = float(lr)
            except ValueError:
                lr = 1e-4
        if isinstance(weight_decay, str):
            try:
                weight_decay = float(weight_decay)
            except ValueError:
                weight_decay = 1e-4
        
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
            return nn.CrossEntropyLoss()  # 使用标准交叉熵损失，不忽略任何类别
        elif criterion_name.lower() == 'focal':
            alpha = self.config.get('focal_alpha', 1.0)
            gamma = self.config.get('focal_gamma', 2.0)
            return FocalLoss(alpha=alpha, gamma=gamma)
        else:
            raise ValueError(f"不支持的损失函数: {criterion_name}")
    
    def train_epoch(self, model, train_loader, optimizer, criterion, scaler=None, epoch=None):
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        epoch_desc = f"训练中 (Epoch {epoch+1})" if epoch is not None else "训练中"
        pbar = tqdm(train_loader, desc=epoch_desc, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for batch_idx, batch in enumerate(pbar):
            # 检查批次是否有效
            if batch is None:
                print("警告: 收到无效批次，跳过")
                continue
                
            # 获取数据
            mfb = batch.get('mfb', None)
            mfcc = batch.get('acoustic', None)  # 数据加载器返回的是'acoustic'
            wav2vec2 = batch.get('embeds', None)  # 数据加载器返回的是'embeds'
            labels = batch.get('exertion_levels', None)
            
            # 调试信息
            if batch_idx == 0:
                print(f"训练器调试 - 批次0:")
                print(f"  MFCC shape: {mfcc.shape if mfcc is not None else None}")
                print(f"  wav2vec2 shape: {wav2vec2.shape if wav2vec2 is not None else None}")
                print(f"  batch keys: {list(batch.keys())}")
                print(f"  use_mfcc: {self.config.get('use_mfcc', 'Not found')}")
                print(f"  use_wav2vec2: {self.config.get('use_wav2vec2', 'Not found')}")
            
            # 检查标签是否为空
            if labels is None:
                print("警告: 批次中没有有效的标签，跳过此批次")
                continue
                
            labels = labels.to(self.device)
            
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
                    outputs = model(mfcc=mfcc, wav2vec2=wav2vec2, mfb=mfb)
                    loss = criterion(outputs, labels)
                
                # 反向传播
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(mfcc=mfcc, wav2vec2=wav2vec2, mfb=mfb)
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            # 统计
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            current_acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, model, val_loader, criterion, epoch=None):
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            epoch_desc = f"验证中 (Epoch {epoch+1})" if epoch is not None else "验证中"
            pbar = tqdm(val_loader, desc=epoch_desc, 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            for batch_idx, batch in enumerate(pbar):
                # 获取数据
                mfb = batch.get('mfb', None)
                mfcc = batch.get('acoustic', None)  # 数据加载器返回的是'acoustic'
                wav2vec2 = batch.get('embeds', None)  # 数据加载器返回的是'embeds'
                labels = batch.get('exertion_levels', None)
                
                # 检查标签是否为空
                if labels is None:
                    print("警告: 验证批次中没有有效的标签，跳过此批次")
                    continue
                    
                labels = labels.to(self.device)
                
                # 移动数据到设备
                if mfb is not None:
                    mfb = mfb.to(self.device)
                if mfcc is not None:
                    mfcc = mfcc.to(self.device)
                if wav2vec2 is not None:
                    wav2vec2 = wav2vec2.to(self.device)
                
                # 前向传播
                outputs = model(mfcc=mfcc, wav2vec2=wav2vec2, mfb=mfb)
                loss = criterion(outputs, labels)
                
                # 统计
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        # 计算指标（包含所有类别）
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
        best_epoch = 0
        patience_counter = 0
        patience = self.config.get('patience', 20)
        
        for epoch in range(config.get('training', {}).get('epochs', 15)):
            print(f"\n{'='*60}")
            n_folds = config.get('training', {}).get('n_folds', 1)
            print(f"Epoch {epoch + 1}/{config.get('training', {}).get('epochs', 15)} - Fold {fold_idx + 1}/{n_folds}")
            print(f"{'='*60}")
            
            # 记录开始时间
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, scaler, epoch
            )
            
            # 验证
            val_loss, val_acc, val_probs, val_preds, val_labels = self.validate_epoch(
                model, val_loader, criterion, epoch
            )
            
            # 学习率调度
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # 记录指标到WandB
            global_step = epoch + fold_idx * config.get('training', {}).get('epochs', 15)
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
            
            # 计算epoch耗时
            epoch_time = time.time() - epoch_start_time
            
            # 计算GPU内存使用
            gpu_memory_used = 0
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
            
            print(f"\nEpoch {epoch + 1} 结果:")
            print(f"   训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"   验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"   学习率: {current_lr:.6f}")
            print(f"   耗时: {epoch_time:.1f}秒")
            print(f"   GPU内存: {gpu_memory_used:.2f}GB")
            
            # 显示改进情况
            if epoch > 0:
                train_improvement = train_acc - fold_train_history[-1]['accuracy']
                val_improvement = val_acc - fold_val_history[-1]['accuracy']
                print(f"   训练改进: {train_improvement:+.4f}")
                print(f"   验证改进: {val_improvement:+.4f}")
            
            # 显示最佳记录
            if val_acc > best_val_acc:
                print(f"   新的最佳验证准确率: {val_acc:.4f} (之前: {best_val_acc:.4f})")
            else:
                print(f"   当前最佳: {best_val_acc:.4f} (还需 {patience - patience_counter} 个epoch)")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                
                # 保存最佳模型
                best_model_path = os.path.join(self.result_dir, "models", f"fold_{fold_idx + 1}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': config
                }, best_model_path)
                
                # 保存到WandB
                self.wandb_manager.save_model(model, best_model_path, {
                    'fold': fold_idx + 1,
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'model_type': 'best'
                })
                
                print(f"保存最佳模型，验证准确率: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # 保存最终模型（如果与最佳模型不同）
            if epoch == config.get('training', {}).get('epochs', 15) - 1 or patience_counter >= patience:
                final_epoch = epoch
                if final_epoch != best_epoch:
                    final_model_path = os.path.join(self.result_dir, "models", f"fold_{fold_idx + 1}_final.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'config': config
                    }, final_model_path)
                    
                    # 保存到WandB
                    self.wandb_manager.save_model(model, final_model_path, {
                        'fold': fold_idx + 1,
                        'epoch': epoch,
                        'val_acc': val_acc,
                        'model_type': 'final'
                    })
                    
                    print(f"保存最终模型，验证准确率: {val_acc:.4f}")
            
            # 早停
            if patience_counter >= patience:
                print(f"早停触发，{patience}个epoch没有改善")
                break
        
        return fold_train_history, fold_val_history, best_val_acc, best_epoch, final_epoch
    
    def _get_config_value(self, config, key, default=None):
        """从嵌套配置中获取值"""
        # 处理嵌套配置结构
        if 'data' in config and key in ['feature_dir', 'use_mfcc', 'use_mfb', 'use_wav2vec2', 'wav2vec2_layers']:
            if key == 'feature_dir':
                return config['data'].get('feature_dir', default)
            elif key == 'use_mfcc':
                return config['data'].get('features', {}).get('use_mfcc', default)
            elif key == 'use_mfb':
                return config['data'].get('features', {}).get('use_mfb', default)
            elif key == 'use_wav2vec2':
                return config['data'].get('features', {}).get('use_wav2vec2', default)
            elif key == 'wav2vec2_layers':
                return config['data'].get('wav2vec2_layers', default)
        return config.get(key, default)
    
    def cross_validation_train(self, dataset, config):
        """交叉验证训练（支持单Fold）"""
        n_folds = config.get('training', {}).get('n_folds')
        if n_folds is None:
            raise ValueError("配置文件中必须指定 'training.n_folds' 参数")
            
        if n_folds == 1:
            print("开始单Fold训练")
        else:
            print(f"开始{n_folds}折交叉验证训练")
        print(f"总epoch数: {config.get('training', {}).get('epochs', 15)}")
        print(f"Batch大小: {config.get('training', {}).get('batch_size', 8)}")
        print(f"学习率: {config.get('training', {}).get('learning_rate', 1e-4)}")
        print(f"设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("="*60)
        
        # 数据平衡
        balance_data = config.get('data', {}).get('balance_data', True)
        print(f"数据平衡配置: {balance_data}")
        if balance_data:
            print("\n应用数据平衡...")
            from src.data.balancer import balance_audio_dataset
            
            # 从配置文件中获取平衡参数
            balance_config = config.get('data', {}).get('balance_config', {})
            target_samples = balance_config.get('target_samples_per_class', 800)
            max_oversampling = balance_config.get('max_oversampling_ratio', 1.5)
            
            balanced_metadata_df = balance_audio_dataset(
                dataset.metadata_df,
                dataset.labels_df,
                self._get_config_value(config, 'feature_dir'),
                target_samples_per_class=target_samples,
                random_seed=config.get('system', {}).get('seed', 42),
                ignore_classes=[],  # 不忽略任何类别
                max_oversampling_ratio=max_oversampling
            )
            
            # 创建平衡后的数据集
            from src.data.loader import AudioFeatureDataset
            balanced_dataset = AudioFeatureDataset(
                metadata_df=balanced_metadata_df,
                feature_dir=self._get_config_value(config, 'feature_dir'),
                labels_df=dataset.labels_df,
                use_acoustic=self._get_config_value(config, 'use_mfcc', True),
                use_mfb=self._get_config_value(config, 'use_mfb', False),
                use_embed=self._get_config_value(config, 'use_wav2vec2', True),
                selected_wav2vec2_layers=self._get_config_value(config, 'wav2vec2_layers', [4])
            )
            
            dataset = balanced_dataset
        
        # 准备数据
        all_sessions = dataset.metadata_df['session'].tolist()
        all_labels = []
        
        for session in all_sessions:
            # 修复标签匹配逻辑
            base_session_id = session.split('_stride_')[0] if '_stride_' in session else session
            label_row = dataset.labels_df[dataset.labels_df['Session Name'] == base_session_id]
            if not label_row.empty:
                all_labels.append(label_row.iloc[0]['Exertion'] - 1)  # 转换为0-4
            else:
                all_labels.append(0)  # 默认标签
        
        # 交叉验证设置
        n_folds = config.get('training', {}).get('n_folds')
        if n_folds is None:
            raise ValueError("配置文件中必须指定 'training.n_folds' 参数")
            
        if n_folds == 1:
            # 单Fold训练：使用基于clip的分层分割
            from src.data.loader import split_train_test_val_stratified
            
            # 从配置文件中获取分割参数
            val_size = config.get('data', {}).get('split', {}).get('val_size', 0.1)
            test_size = config.get('data', {}).get('split', {}).get('test_size', 0.1)
            random_state = config.get('system', {}).get('seed', 42)
            
            print(f"单Fold训练配置（基于clip的分层分割）:")
            print(f"  验证集大小: {val_size}")
            print(f"  测试集大小: {test_size}")
            print(f"  训练集大小: {1-val_size-test_size}")
            print(f"  随机种子: {random_state}")
            
            # 检查是否使用分层分割
            use_stratified_split = config.get('data', {}).get('split', {}).get('use_stratified_split', False)
            
            if use_stratified_split:
                print("使用基于clip的分层分割确保所有类别都有代表...")
                train_sessions, val_sessions, test_sessions = split_train_test_val_stratified(
                    dataset.metadata_df, 
                    dataset.labels_df,
                    test_size=test_size, 
                    val_size=val_size, 
                    random_state=random_state
                )
            else:
                print("使用基于clip的随机分割...")
                # 直接基于clip进行随机分割
                from sklearn.model_selection import train_test_split
                import numpy as np
                
                # 获取所有clips和对应的标签
                clips = dataset.metadata_df['session'].tolist()
                labels = []
                for session in clips:
                    base_session_id = session.split('_stride_')[0] if '_stride_' in session else session
                    label_row = dataset.labels_df[dataset.labels_df['Session Name'] == base_session_id]
                    if not label_row.empty:
                        labels.append(label_row.iloc[0]['Exertion'] - 1)
                    else:
                        labels.append(0)
                
                # 分层分割
                train_clips, temp_clips, train_labels, temp_labels = train_test_split(
                    clips, labels, test_size=test_size + val_size, 
                    random_state=random_state, stratify=labels
                )
                
                # 从临时集合中分割出验证集和测试集
                val_clips, test_clips, val_labels, test_labels = train_test_split(
                    temp_clips, temp_labels, test_size=test_size/(test_size + val_size),
                    random_state=random_state, stratify=temp_labels
                )
                
                train_sessions = train_clips
                val_sessions = val_clips
                test_sessions = test_clips
            
            # 转换为索引
            session_to_idx = {session: idx for idx, session in enumerate(all_sessions)}
            train_idx = [session_to_idx[session] for session in train_sessions if session in session_to_idx]
            val_idx = [session_to_idx[session] for session in val_sessions if session in session_to_idx]
            test_idx = [session_to_idx[session] for session in test_sessions if session in session_to_idx]
            
            fold_splits = [(train_idx, val_idx)]
            test_indices = test_idx  # 保存测试集索引
        else:
            # 多Fold交叉验证（也需要基于参与者）
            random_state = config.get('system', {}).get('seed', 42)
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            fold_splits = list(skf.split(all_sessions, all_labels))
            test_indices = None
        
        # 数据泄露检查
        print("\n开始数据泄露检查...")
        from src.utils.data_leakage_checker import check_training_pipeline
        
        if n_folds == 1:
            train_indices = train_idx
            val_indices = val_idx
            # test_indices已经在上面定义了
        else:
            # 多fold情况下，使用第一个fold进行检查
            train_indices = fold_splits[0][0]
            val_indices = fold_splits[0][1]
            test_indices = None
        
        leakage_result = check_training_pipeline(dataset, config, train_indices, val_indices, test_indices)
        
        if not leakage_result['passed']:
            print("❌ 数据泄露检查失败，训练终止")
            return []
        
        print("✅ 数据泄露检查通过，继续训练")
        
        # 数据标准化（在train_test_split之后）
        if config.get('data', {}).get('normalize_data', True):
            print("\n应用数据标准化...")
            from src.data.normalizer_memory_efficient import normalize_dataset_features_memory_efficient
            
            # 标准化参数保存目录
            normalize_save_dir = os.path.join(self.result_dir, "normalization")
            
            # 内存友好的特征标准化
            normalized_train, normalized_val, normalized_test = normalize_dataset_features_memory_efficient(
                dataset, train_indices, val_indices, test_indices, normalize_save_dir, batch_size=50
            )
            
            print("数据标准化完成")
        else:
            normalized_train = normalized_val = normalized_test = None
        
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            print(f"\n{'='*50}")
            if n_folds == 1:
                print("单Fold训练")
            else:
                print(f"Fold {fold_idx + 1}/{n_folds}")
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
            train_history, val_history, best_acc, best_epoch, final_epoch = self.train_fold(
                fold_idx, train_loader, val_loader, model, config
            )
            
            fold_results.append({
                'fold': fold_idx + 1,
                'train_history': train_history,
                'val_history': val_history,
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'final_epoch': final_epoch
            })
            
            if n_folds == 1:
                print(f"单Fold训练完成，最佳验证准确率: {best_acc:.4f}")
            else:
                print(f"Fold {fold_idx + 1} 完成，最佳验证准确率: {best_acc:.4f}")
        
        # 保存结果
        self._save_cv_results(fold_results)
        
        # 记录最终结果到WandB
        self._log_final_results(fold_results)
        
        # 测试最佳和最终模型
        test_results = self.test_best_and_final_models(fold_results, dataset, config)
        
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
            feature_dir=self._get_config_value(config, 'feature_dir'),
            labels_df=dataset.labels_df,
            use_acoustic=self._get_config_value(config, 'use_mfcc', True),
            use_mfb=self._get_config_value(config, 'use_mfb', False),
            use_embed=self._get_config_value(config, 'use_wav2vec2', True),
            selected_wav2vec2_layers=self._get_config_value(config, 'wav2vec2_layers', [4]),
            batch_size=config.get('training', {}).get('batch_size', 32),
            shuffle=shuffle,
            num_workers=config.get('training', {}).get('num_workers', 4),
            pin_memory=config.get('training', {}).get('pin_memory', False),
            persistent_workers=config.get('training', {}).get('persistent_workers', False),
            prefetch_factor=config.get('training', {}).get('prefetch_factor', 2)
        )
    
    def _create_model(self, config):
        """创建模型"""
        from src.models.vgg16_exertion import create_model
        
        # 如果config中没有模型配置，使用默认值
        if 'mfcc_dim' not in config:
            model_config = {
                'mfcc_dim': 40,  # 实际MFCC维度
                'wav2vec2_dim': 768,
                'mfb_dim': 40,   # MFB特征维度
                'num_classes': 5,  # 1-5级，共5个类别
                'dropout_rate': 0.5,
                'use_mfcc': self._get_config_value(config, 'use_mfcc', True),
                'use_wav2vec2': self._get_config_value(config, 'use_wav2vec2', True),
                'use_mfb': self._get_config_value(config, 'use_mfb', False),
                'optimize_for_rtx4070': True,
            }
        else:
            model_config = config
        

        return create_model(model_config)
    
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
                    'epoch': int(val_epoch['epoch']),
                    'loss': float(val_epoch['loss']),
                    'accuracy': float(val_epoch['accuracy']),
                    'probs': [prob.tolist() for prob in val_epoch['probs']],
                    'preds': [int(p) for p in val_epoch['preds']],
                    'labels': [int(l) for l in val_epoch['labels']]
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
    
    def test_best_and_final_models(self, fold_results, dataset, config):
        """测试验证最佳epoch和最终epoch的模型"""
        print("\n开始测试最佳和最终模型...")
        
        test_results = {}
        
        for fold_result in fold_results:
            fold_idx = fold_result['fold']
            best_epoch = fold_result['best_epoch']
            final_epoch = fold_result['final_epoch']
            
            print(f"\nFold {fold_idx}:")
            print(f"  最佳epoch: {best_epoch + 1}")
            print(f"  最终epoch: {final_epoch + 1}")
            
            # 测试最佳模型
            best_model_path = os.path.join(self.result_dir, "models", f"fold_{fold_idx}_best.pth")
            if os.path.exists(best_model_path):
                best_acc = self._test_single_model(best_model_path, dataset, config, f"最佳模型 (Epoch {best_epoch + 1})")
                test_results[f'fold_{fold_idx}_best'] = best_acc
            else:
                print(f"  最佳模型文件不存在: {best_model_path}")
            
            # 测试最终模型（如果与最佳模型不同）
            if best_epoch != final_epoch:
                final_model_path = os.path.join(self.result_dir, "models", f"fold_{fold_idx}_final.pth")
                if os.path.exists(final_model_path):
                    final_acc = self._test_single_model(final_model_path, dataset, config, f"最终模型 (Epoch {final_epoch + 1})")
                    test_results[f'fold_{fold_idx}_final'] = final_acc
                    
                    # 比较结果
                    improvement = best_acc - final_acc
                    print(f"  最佳模型 vs 最终模型: {improvement:+.4f}")
                else:
                    print(f"  最终模型文件不存在: {final_model_path}")
            else:
                print(f"  最佳模型和最终模型相同，跳过最终模型测试")
        
        # 保存测试结果
        test_results_path = os.path.join(self.result_dir, "model_test_results.json")
        with open(test_results_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n模型测试结果已保存到: {test_results_path}")
        return test_results
    
    def _test_single_model(self, model_path, dataset, config, model_name):
        """测试单个模型"""
        try:
            # 加载模型
            checkpoint = torch.load(model_path, map_location=self.device)
            model = self._create_model(config)
            
            # 兼容不同的保存格式
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # 直接加载状态字典
                model.load_state_dict(checkpoint)
                
            model.to(self.device)
            model.eval()
            
            # 创建测试数据加载器（使用验证集）
            all_sessions = dataset.metadata_df['session'].tolist()
            test_sessions = all_sessions[:len(all_sessions)//5]  # 使用20%的数据作为测试集
            
            test_loader = self._create_dataloader(dataset, test_sessions, config, shuffle=False)
            
            # 测试
            criterion = self._get_criterion()
            test_loss, test_acc, _, _, _ = self.validate_epoch(model, test_loader, criterion)
            
            print(f"  {model_name} - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
            
            return test_acc
            
        except Exception as e:
            print(f"  测试{model_name}时出错: {e}")
            return 0.0





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