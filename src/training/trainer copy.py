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

# Add WandB manager
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.wandb_manager import WandBManager

class ExertionTrainer:
    """
    Exertion level detection model trainer
    Supports 5-fold cross validation, mixed precision training and GPU optimization
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create result directory
        self.result_dir = self._create_result_dir()
        
        # Training history
        self.train_history = []
        self.val_history = []
        
        # Best model
        self.best_model = None
        self.best_score = 0.0
        
        # Initialize WandB manager
        self.wandb_manager = WandBManager(config)
        
        # Set random seed
        torch.manual_seed(config.get('seed', 42))
        np.random.seed(config.get('seed', 42))
        
        print(f"Results directory: {self.result_dir}")
    
    def _create_result_dir(self):
        """Create result directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"VGG16_wav2vec2_layer{self.config.get('wav2vec2_layer', 4)}"
        result_dir = os.path.join("result", f"{model_name}_{timestamp}")
        
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(os.path.join(result_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(result_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(result_dir, "logs"), exist_ok=True)
        
        return result_dir
    
    def _save_config(self):
        """Save training configuration"""
        config_path = os.path.join(self.result_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def _get_optimizer(self, model):
        """Get optimizer"""
        optimizer_name = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        # Ensure learning rate and weight decay are numeric
        if isinstance(lr, str):
            lr = float(lr)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
        
        if optimizer_name.lower() == 'adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _get_scheduler(self, optimizer):
        """Get learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'step')
        
        if scheduler_name.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.get('step_size', 5),
                gamma=self.config.get('gamma', 0.5)
            )
        elif scheduler_name.lower() == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.config.get('gamma', 0.9)
            )
        elif scheduler_name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.config.get('epochs', 15),
                eta_min=1e-6
            )
        elif scheduler_name.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return None
    
    def _get_criterion(self):
        """Get loss function"""
        criterion_name = self.config.get('criterion', 'cross_entropy')
        
        if criterion_name.lower() == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif criterion_name.lower() == 'focal':
            return FocalLoss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion_name}")
    
    def train_epoch(self, model, train_loader, optimizer, criterion, scaler=None, epoch=None):
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        # 为每个epoch添加进度条
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Get data (batch is a tuple of (features, labels))
            features, labels = batch
            
            # Check if labels are valid
            if labels is None:
                print("Warning: Invalid batch with no labels, skipping")
                continue
                
            labels = labels.to(self.device)
            features = features.to(self.device)
            
            # Split features into mfcc and wav2vec2 if needed
            # For now, we'll pass the combined features directly to the model
            mfcc = None
            wav2vec2 = None
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy_score(all_labels, all_preds):.4f}'
            })
            

        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, model, val_loader, criterion, epoch=None):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        # 为验证添加进度条
        from tqdm import tqdm
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} (Val)", leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Get data (batch is a tuple of (features, labels))
                features, labels = batch
                
                # Check if labels are valid
                if labels is None:
                    print("Warning: Invalid batch with no labels, skipping")
                    continue
                    
                labels = labels.to(self.device)
                features = features.to(self.device)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy_score(all_labels, all_preds):.4f}'
                })
                

        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy, np.array(all_probs), np.array(all_preds), np.array(all_labels)
    
    def train_fold(self, fold_idx, train_loader, val_loader, model, config):
        """Train one fold"""
        print(f"Starting Fold {fold_idx + 1}")
        
        # Initialize
        optimizer = self._get_optimizer(model)
        scheduler = self._get_scheduler(optimizer)
        criterion = self._get_criterion()
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.config.get('use_amp', True) else None
        
        # Log model info to WandB
        self.wandb_manager.log_model_info(model)
        
        # Training history
        fold_train_history = []
        fold_val_history = []
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        patience = self.config.get('patience', 20)
        
        epochs = config.get('training', {}).get('epochs', 15)
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, scaler, epoch
            )
            
            # Validation
            val_loss, val_acc, val_probs, val_preds, val_labels = self.validate_epoch(
                model, val_loader, criterion, epoch
            )
            
            # Learning rate scheduling
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Log metrics to WandB after each epoch
            global_step = epoch + fold_idx * epochs
            
            # Log training metrics
            self.wandb_manager.log_training_metrics({
                'loss': train_loss,
                'accuracy': train_acc
            }, step=global_step)
            
            # Log validation metrics
            self.wandb_manager.log_validation_metrics({
                'loss': val_loss,
                'accuracy': val_acc
            }, step=global_step)
            
            # Log learning rate
            self.wandb_manager.log_learning_rate(current_lr, step=global_step)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
            
            # Record history
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
            
            # Show best record
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                best_model_path = os.path.join(self.result_dir, "models", f"best_fold_{fold_idx+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': config
                }, best_model_path)
                
                print(f"Epoch {epoch+1}/{epochs}: New best model saved (Val Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        # Save final model
        final_model_path = os.path.join(self.result_dir, "models", f"final_fold_{fold_idx+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': config
        }, final_model_path)
        
        # Save fold evaluation results
        fold_eval_path = os.path.join(self.result_dir, "evaluation", f"fold_{fold_idx+1}_evaluation.json")
        fold_eval_data = {
            'fold_idx': fold_idx,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'final_val_acc': val_acc,
            'labels': [int(label) for label in fold_val_history[-1]['labels']],
            'predictions': [int(pred) for pred in fold_val_history[-1]['preds']],
            'probabilities': fold_val_history[-1]['probs'].tolist() if hasattr(fold_val_history[-1]['probs'], 'tolist') else fold_val_history[-1]['probs']
        }
        
        os.makedirs(os.path.dirname(fold_eval_path), exist_ok=True)
        with open(fold_eval_path, 'w', encoding='utf-8') as f:
            json.dump(fold_eval_data, f, indent=2, ensure_ascii=False)
        
        # Return fold results
        return {
            'fold_idx': fold_idx,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'final_val_acc': val_acc,
            'train_history': fold_train_history,
            'val_history': fold_val_history,
            'best_model_path': best_model_path,
            'final_model_path': final_model_path
        }
    
    def _get_config_value(self, config, key, default=None):
        """Get configuration value with nested structure support"""
        if key in config:
            return config[key]
        
        # Check nested structures
        for section in ['data', 'model', 'training', 'system']:
            if section in config and key in config[section]:
                return config[section][key]
        
        return default
    
    def cross_validation_train(self, dataset, config):
        """Perform cross-validation training"""
        print("Starting cross-validation training")
        
        # Get configuration
        n_folds = self._get_config_value(config, 'n_folds', 5)
        batch_size = self._get_config_value(config, 'batch_size', 8)
        num_workers = self._get_config_value(config, 'num_workers', 4)
        pin_memory = self._get_config_value(config, 'pin_memory', True)
        
        # Create single fold training (no cross-validation)
        if n_folds == 1:
            # Use pre-defined train/validation split from dataset
            train_indices = [i for i, sample in enumerate(dataset.available_samples) if sample['split'] == 'train']
            val_indices = [i for i, sample in enumerate(dataset.available_samples) if sample['split'] == 'val']
            
            fold_results = []
            
            print(f"\n{'='*60}")
            print(f"Single Fold Training")
            print(f"{'='*60}")
            
            print(f"Train: {len(train_indices)} samples, Val: {len(val_indices)} samples")
            
            # Create data loaders
            train_loader = self._create_dataloader(dataset, train_indices, config, shuffle=True)
            val_loader = self._create_dataloader(dataset, val_indices, config, shuffle=False)
            
            # Create model
            model = self._create_model(config)
            model = model.to(self.device)
            
            # Train single fold
            fold_result = self.train_fold(0, train_loader, val_loader, model, config)
            fold_results.append(fold_result)
            
            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # Use cross-validation
            labels = [sample['exertion_level'] for sample in dataset.available_samples]
            indices = np.arange(len(dataset))
            
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.get('seed', 42))
            
            fold_results = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
                print(f"\n{'='*60}")
                print(f"Fold {fold_idx + 1}/{n_folds}")
                print(f"{'='*60}")
                
                print(f"Train samples: {len(train_idx)}")
                print(f"Val samples: {len(val_idx)}")
                
                # Create data loaders
                train_loader = self._create_dataloader(dataset, train_idx, config, shuffle=True)
                val_loader = self._create_dataloader(dataset, val_idx, config, shuffle=False)
                
                # Create model
                model = self._create_model(config)
                model = model.to(self.device)
                
                # Train fold
                fold_result = self.train_fold(fold_idx, train_loader, val_loader, model, config)
                fold_results.append(fold_result)
                
                # Clean up
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Save cross-validation results
        self._save_cv_results(fold_results)
        
        # Log final results
        self._log_final_results(fold_results)
        
        return fold_results
    
    def finish(self):
        """Finish training and cleanup"""
        self.wandb_manager.finish()
        print("Training completed")
    
    def _create_dataloader(self, dataset, indices, config, shuffle=True):
        """Create data loader for specific indices"""
        batch_size = self._get_config_value(config, 'batch_size', 8)
        num_workers = self._get_config_value(config, 'num_workers', 4)
        pin_memory = self._get_config_value(config, 'pin_memory', True)
        
        # Create subset dataset
        from torch.utils.data import Subset
        filtered_dataset = Subset(dataset, indices)
        
        return DataLoader(
            filtered_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=getattr(filtered_dataset, 'collate_fn', None)
        )
    
    def _create_model(self, config):
        """Create model from configuration"""
        from src.models.vgg16_exertion import create_model
        return create_model(config)
    
    def _save_cv_results(self, fold_results):
        """Save cross-validation results"""
        results_path = os.path.join(self.result_dir, "cv_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in fold_results:
            serializable_result = {
                'fold_idx': result['fold_idx'],
                'best_epoch': result['best_epoch'],
                'best_val_acc': float(result['best_val_acc']),
                'final_val_acc': float(result['final_val_acc']),
                'best_model_path': result['best_model_path'],
                'final_model_path': result['final_model_path']
            }
            serializable_results.append(serializable_result)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"Cross-validation results saved to: {results_path}")
    
    def _log_final_results(self, fold_results):
        """Log final training results"""
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        
        # Calculate statistics
        val_accuracies = [result['best_val_acc'] for result in fold_results]
        avg_acc = np.mean(val_accuracies)
        std_acc = np.std(val_accuracies)
        
        print(f"Average validation accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"Individual fold accuracies: {[f'{acc:.4f}' for acc in val_accuracies]}")
        print(f"Best fold: {np.argmax(val_accuracies) + 1} ({max(val_accuracies):.4f})")
        print(f"Worst fold: {np.argmin(val_accuracies) + 1} ({min(val_accuracies):.4f})")
        
        # Log to WandB
        self.wandb_manager.log_cv_results({
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc,
            'fold_accuracies': val_accuracies,
            'best_fold': np.argmax(val_accuracies) + 1,
            'worst_fold': np.argmin(val_accuracies) + 1
        })
        
        print(f"{'='*60}")
    
    def test_best_and_final_models(self, fold_results, dataset, config):
        """Test best and final models from all folds"""
        print("\nTesting best and final models from all folds")
        
        test_results = {}
        
        for result in fold_results:
            fold_idx = result['fold_idx']
            
            # Test best model
            best_result = self._test_single_model(
                result['best_model_path'], dataset, config, f"best_fold_{fold_idx+1}"
            )
            test_results[f"best_fold_{fold_idx+1}"] = best_result
            
            # Test final model
            final_result = self._test_single_model(
                result['final_model_path'], dataset, config, f"final_fold_{fold_idx+1}"
            )
            test_results[f"final_fold_{fold_idx+1}"] = final_result
        
        # Save test results
        test_results_path = os.path.join(self.result_dir, "test_results.json")
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        print(f"Test results saved to: {test_results_path}")
        return test_results
    
    def _test_single_model(self, model_path, dataset, config, model_name):
        """Test a single model"""
        print(f"Testing {model_name}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        model = self._create_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Create test data loader
        test_loader = self._create_dataloader(dataset, dataset.metadata_df['session_id'].unique(), config, shuffle=False)
        
        # Test
        criterion = self._get_criterion()
        test_loss, test_acc, test_probs, test_preds, test_labels = self.validate_epoch(
            model, test_loader, criterion
        )
        
        # Calculate additional metrics
        from sklearn.metrics import f1_score, roc_auc_score
        
        f1_macro = f1_score(test_labels, test_preds, average='macro')
        
        # ROC AUC (one-vs-rest)
        try:
            auc = roc_auc_score(test_labels, test_probs, multi_class='ovr')
        except:
            auc = 0.0
        
        result = {
            'model_name': model_name,
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'f1_macro': float(f1_macro),
            'auc': float(auc),
            'predictions': test_preds.tolist(),
            'probabilities': test_probs.tolist(),
            'labels': test_labels.tolist()
        }
        
        print(f"  {model_name} - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {f1_macro:.4f}, AUC: {auc:.4f}")
        
        return result

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean() 