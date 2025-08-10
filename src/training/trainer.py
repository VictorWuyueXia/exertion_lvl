# -*- coding: utf-8 -*-
"""
Trainer for DualChannelExertionModel (MFCC + wav2vec2 mids, CORAL ordinal)
- Compatible with multiple legacy batch formats:
  * dict: {'mfcc':[B,C,F,T], 'w2v':[B,T,Dw], 'y':[B], 'mask':[B,T](optional)}
  * tuple: (mfcc, w2v, y)
  * tuple: (mfcc, w2v, mask, y)
  * tuple: ({'mfcc','w2v','mask?'}, y)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ---- import your model ----
# Put dual_channel_exertion.py on PYTHONPATH, or adjust this import.
try:
    from dual_channel_exertion import (
        DualChannelExertionModel,
        coral_predict, coral_predict_proba, coral_loss
    )
except Exception:
    from src.models.dual_channel_exertion import (
        DualChannelExertionModel,
        coral_predict, coral_predict_proba, coral_loss
    )

# ---------------- Metrics ----------------
def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """
    Compute QWK (Quadratic Weighted Kappa) between two integer rating vectors.
    """
    assert y_true.shape == y_pred.shape
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    w = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            w[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist) / max(1.0, np.sum(act_hist))

    num = np.sum(w * O)
    den = np.sum(w * E) if np.sum(w * E) > 0 else 1.0
    return 1.0 - (num / den)

# -------------- Config dataclass --------------
@dataclass
class TrainConfig:
    # Data
    train_batches_per_epoch: Optional[int] = None
    val_batches_per_epoch: Optional[int] = None
    num_workers: int = 4
    pin_memory: bool = True

    # Model
    mfcc_dim: int = 40
    wav2vec2_dim: int = 768
    hidden_dim: int = 256
    num_classes: int = 5
    in_ch: int = 1

    # Optim
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.999)

    # Scheduler
    scheduler: Optional[str] = "cosine"
    step_size: int = 5
    gamma: float = 0.5
    warmup_epochs: int = 0
    max_epochs: int = 40
    min_lr_mult: float = 0.05

    # Batch/AMP
    batch_size: int = 16
    amp: bool = True
    grad_clip: float = 1.0

    # Loss (CORAL)
    criterion: str = "coral"
    coral_gamma: Optional[float] = None  # e.g., 1.5 for ordinal focal; None disables
    coral_alpha: float = 1.0             # pos weight factor, 1.0 disables
    per_threshold_weight: Optional[list] = None  # length K-1 or None

    # Early stopping / checkpoint
    monitor: str = "qwk"  # qwk | uar | macro_f1 | val_loss
    patience: int = 7
    ckpt_dir: str = "./checkpoints"
    exp_name: str = "exertion_vggish_w2v2_coral"

    # Logging
    log_every: int = 20

class Trainer:
    def __init__(self, config: Dict[str, Any], device: Optional[str] = None):
        self.cfg = TrainConfig(**config)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)

        # build model
        self.model = DualChannelExertionModel(
            mfcc_dim=self.cfg.mfcc_dim,
            wav2vec2_dim=self.cfg.wav2vec2_dim,
            hidden_dim=self.cfg.hidden_dim,
            num_classes=self.cfg.num_classes,
            in_ch=self.cfg.in_ch,
        ).to(self.device)

        # optimizer
        if self.cfg.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr,
                                         betas=self.cfg.betas, weight_decay=self.cfg.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.optimizer}")

        # scheduler
        self.scheduler = None
        if self.cfg.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.max_epochs,
                eta_min=self.cfg.lr * self.cfg.min_lr_mult
            )
        elif self.cfg.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.step_size,
                gamma=self.cfg.gamma
            )

        # AMP (new API)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.amp and self.device.type == "cuda")

        # CORAL weights
        self.per_thr = None
        if self.cfg.per_threshold_weight is not None:
            self.per_thr = torch.tensor(self.cfg.per_threshold_weight, dtype=torch.float32, device=self.device)

        self.best_metric = -np.inf
        self.best_state = None
        self.history = []

    # --------------------- public API ---------------------
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        for epoch in range(self.cfg.max_epochs):
            tr_loss, tr_acc = self._train_epoch(train_loader, epoch)
            va_loss, va_acc, va_probs, va_preds, va_labels = self._validate_epoch(val_loader, epoch)
            va_qwk = quadratic_weighted_kappa(va_labels, va_preds, self.cfg.num_classes)
            va_f1 = f1_score(va_labels, va_preds, average='macro')

            # confusion matrix print (quick diagnostic)
            cm = confusion_matrix(va_labels, va_preds, labels=list(range(self.cfg.num_classes)))
            print("Val confusion matrix:\n", cm)

            # monitor
            metric_map = {
                "qwk": va_qwk,
                "macro_f1": va_f1,
                "uar": self._uar(va_labels, va_preds),
                "val_loss": -va_loss  # we want higher-is-better
            }
            metric = metric_map.get(self.cfg.monitor, va_qwk)

            if metric > self.best_metric:
                self.best_metric = metric
                self.best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                self._save_ckpt("best.pt", epoch, va_qwk, va_f1, va_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            self.history.append({
                "epoch": epoch + 1,
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": va_loss, "val_acc": va_acc,
                "val_qwk": va_qwk, "val_f1_macro": va_f1
            })
            print(f"[Epoch {epoch+1}/{self.cfg.max_epochs}] "
                  f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
                  f"acc={va_acc:.4f} qwk={va_qwk:.4f} f1_macro={va_f1:.4f}")

            if self._early_stop():
                print("Early stopping triggered.")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", leave=False):
                mfcc, w2v, labels, mask = self._unpack_batch(batch)
                mfcc = mfcc.to(self.device, non_blocking=True)
                w2v  = w2v.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                mask = mask.to(self.device) if mask is not None else None

                with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                    logits, _ = self.model(mfcc, w2v, mask)
                    loss = coral_loss(logits, labels, K=self.cfg.num_classes,
                                      gamma=self.cfg.coral_gamma, per_threshold_weight=self.per_thr, alpha=1.0)

                total_loss += loss.item()
                preds = coral_predict(logits).cpu().numpy()
                probs = coral_predict_proba(logits, K=self.cfg.num_classes).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)

        all_preds = np.asarray(all_preds, dtype=int)
        all_labels = np.asarray(all_labels, dtype=int)
        all_probs = np.asarray(all_probs, dtype=float)

        acc = accuracy_score(all_labels, all_preds)
        f1m = f1_score(all_labels, all_preds, average='macro')
        qwk = quadratic_weighted_kappa(all_labels, all_preds, self.cfg.num_classes)
        
        # 生成可视化
        self._generate_test_visualizations(all_labels, all_preds, all_probs)
        
        return {"loss": total_loss / max(1, len(test_loader)), "acc": acc, "f1_macro": f1m, "qwk": qwk}
    
    def _generate_test_visualizations(self, labels, preds, probs):
        """生成测试集可视化"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        # 创建结果目录
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        
        # 1. 混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(self.cfg.num_classes), 
                   yticklabels=range(self.cfg.num_classes))
        plt.title('Test Set Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(self.cfg.ckpt_dir, 'test_confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 类别准确率
        plt.figure(figsize=(10, 6))
        class_accuracies = []
        for i in range(self.cfg.num_classes):
            mask = labels == i
            if mask.sum() > 0:
                acc = (preds[mask] == labels[mask]).mean()
                class_accuracies.append(acc)
            else:
                class_accuracies.append(0.0)
        
        bars = plt.bar(range(self.cfg.num_classes), class_accuracies)
        plt.xlabel('Exertion Level')
        plt.ylabel('Accuracy')
        plt.title('Test Set Accuracy by Class')
        plt.xticks(range(self.cfg.num_classes))
        
        # 在柱状图上添加数值
        for bar, acc in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.cfg.ckpt_dir, 'test_class_accuracy.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 预测分布
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
        plt.savefig(os.path.join(self.cfg.ckpt_dir, 'test_prediction_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"测试集可视化已保存到: {self.cfg.ckpt_dir}")

    # --------------------- internals ---------------------
    def _unpack_batch(self, batch):
        # New format: dict
        if isinstance(batch, dict):
            mfcc = batch['mfcc']
            w2v  = batch['w2v']
            y    = batch['y']
            mask = batch.get('mask', None)
            return mfcc, w2v, y, mask

        # Legacy formats
        if isinstance(batch, (tuple, list)):
            # ({...}, y)
            if len(batch) == 2 and isinstance(batch[0], dict):
                x, y = batch
                mfcc = x['mfcc']; w2v = x['w2v']; mask = x.get('mask', None)
                return mfcc, w2v, y, mask
            # (mfcc, w2v, y)
            if len(batch) == 3:
                mfcc, w2v, y = batch
                mask = None
                return mfcc, w2v, y, mask
            # (mfcc, w2v, mask, y)
            if len(batch) == 4:
                mfcc, w2v, mask, y = batch
                return mfcc, w2v, y, mask

        raise RuntimeError(
            "Unsupported batch format. Expected dict {'mfcc','w2v','y','mask?'} "
            "or tuple (mfcc,w2v,y)/(mfcc,w2v,mask,y) or ({...},y)."
        )

    def _train_epoch(self, loader: DataLoader, epoch: int):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        steps = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} (Train)", leave=False)
        for batch in pbar:
            mfcc, w2v, labels, mask = self._unpack_batch(batch)
            mfcc = mfcc.to(self.device, non_blocking=True)
            w2v  = w2v.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            mask = mask.to(self.device) if mask is not None else None

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                logits, _ = self.model(mfcc, w2v, mask)
                loss = coral_loss(logits, labels, K=self.cfg.num_classes,
                                  gamma=self.cfg.coral_gamma, per_threshold_weight=self.per_thr, alpha=1.0)

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            preds = coral_predict(logits)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            steps += 1
            if self.cfg.train_batches_per_epoch and steps >= self.cfg.train_batches_per_epoch:
                break

            if steps % max(1, self.cfg.log_every) == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{accuracy_score(all_labels, all_preds):.4f}"})

        return total_loss / max(1, steps), accuracy_score(all_labels, all_preds)

    def _validate_epoch(self, loader: DataLoader, epoch: int):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        steps = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Epoch {epoch+1} (Val)", leave=False):
                mfcc, w2v, labels, mask = self._unpack_batch(batch)
                mfcc = mfcc.to(self.device, non_blocking=True)
                w2v  = w2v.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                mask = mask.to(self.device) if mask is not None else None

                with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                    logits, _ = self.model(mfcc, w2v, mask)
                    loss = coral_loss(logits, labels, K=self.cfg.num_classes,
                                      gamma=self.cfg.coral_gamma, per_threshold_weight=self.per_thr, alpha=1.0)

                total_loss += loss.item()
                preds = coral_predict(logits).cpu().numpy()
                probs = coral_predict_proba(logits, K=self.cfg.num_classes).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)

                steps += 1
                if self.cfg.val_batches_per_epoch and steps >= self.cfg.val_batches_per_epoch:
                    break

        all_preds = np.asarray(all_preds, dtype=int)
        all_labels = np.asarray(all_labels, dtype=int)
        all_probs = np.asarray(all_probs, dtype=float)

        acc = accuracy_score(all_labels, all_preds)
        f1m = f1_score(all_labels, all_preds, average='macro')
        return total_loss / max(1, steps), acc, all_probs, all_preds, all_labels

    def _uar(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import recall_score
        return recall_score(y_true, y_pred, average='macro')

    def _save_ckpt(self, filename: str, epoch: int, qwk: float, f1m: float, val_loss: float):
        path = os.path.join(self.cfg.ckpt_dir, f"{self.cfg.exp_name}_{filename}")
        torch.save({
            "model_state": self.best_state if self.best_state is not None else self.model.state_dict(),
            "epoch": epoch,
            "config": self.cfg.__dict__,
            "metrics": {"qwk": qwk, "f1_macro": f1m, "val_loss": val_loss}
        }, path)

    def _early_stop(self) -> bool:
        # stop if no improvement (of the monitored metric) for 'patience' epochs
        if len(self.history) <= self.cfg.patience:
            return False
        monitor_key = {"qwk":"val_qwk","macro_f1":"val_f1_macro","uar":"val_f1_macro","val_loss":"val_loss"}.get(self.cfg.monitor, "val_qwk")
        recent = [h[monitor_key] for h in self.history[-(self.cfg.patience+1):]]
        return max(recent) == recent[0]  # best was at the oldest point -> no improvement
