# src/train.py
# -*- coding: utf-8 -*-
"""
Training runner wired to your real dataloaders + the new CORAL Trainer.
No dummy data. No cross-validation. Just train/val/test.
"""

import os, sys, argparse, yaml, json
from datetime import datetime
import torch

# make project root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# your config & data pipeline
from src.utils.config_manager import load_config_from_args, ConfigManager
from src.data_preparation.loader import create_data_loaders

# our new trainer (dual-channel + CORAL)
try:
    from src.training.trainer import Trainer  # <- place trainer.py here
except ImportError:
    from trainer import Trainer                # fallback if you keep it at project root

def _to_trainer_cfg(cm: ConfigManager) -> dict:
    """Map your config keys into the Trainer's flat config dict."""
    get = cm.get
    cfg = {
        # --- data/loader knobs ---
        "num_workers":      get("training.num_workers", 4),
        "pin_memory":       get("training.pin_memory", True),
        "batch_size":       get("training.batch_size", 16),

        # --- model ---
        "mfcc_dim":         get("model.mfcc_dim", 40),
        "wav2vec2_dim":     get("model.wav2vec2_dim", 768),
        "hidden_dim":       get("model.hidden_dim", 256),
        "num_classes":      get("model.num_classes", 5),
        "in_ch":            get("model.in_ch", 1),

        # --- optim ---
        "optimizer":        get("training.optimizer", "adamw"),
        "lr":               get("training.learning_rate", 3e-4),
        "weight_decay":     get("training.weight_decay", 1e-2),
        "betas":            tuple(get("training.betas", [0.9, 0.999])),

        # --- scheduler / epochs ---
        "scheduler":        get("training.scheduler", "cosine"),
        "warmup_epochs":    get("training.warmup_epochs", 0),
        "max_epochs":       get("training.max_epochs", 40),
        "min_lr_mult":      get("training.min_lr_mult", 0.05),

        # --- amp / grad ---
        "amp":              get("system.gpu.enable_amp", True),
        "grad_clip":        get("training.grad_clip", 1.0),

        # --- loss: CORAL ---
        "criterion":            "coral",
        "coral_gamma":          get("training.coral_gamma", None),
        "coral_alpha":          get("training.coral_alpha", 1.0),
        "per_threshold_weight": get("training.per_threshold_weight", None),

        # --- early stopping / logging ---
        "monitor":          get("training.monitor", "qwk"),  # qwk|uar|macro_f1|val_loss
        "patience":         get("training.patience", 7),
        "ckpt_dir":         get("output.ckpt_dir", "./checkpoints"),
        "exp_name":         get("experiment.name", "exertion_vggish_w2v2_coral"),
        "log_every":        get("output.logging.log_every", 20),

        # optional caps
        "train_batches_per_epoch": get("training.train_batches_per_epoch", None),
        "val_batches_per_epoch":   get("training.val_batches_per_epoch", None),
    }
    return cfg

def main():
    print("="*60)
    print("CONFIG & ENV")
    print("="*60)

    # load your config (keeps CLI flags etc.)
    cm: ConfigManager = load_config_from_args()
    print(f"Config path: {cm.config_path}")

    # device info (just logging)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # build loaders from your pipeline (NO dummies)
    # create_data_loaders should read splits from your config and return 3 DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(cm)

    # build Trainer config from your ConfigManager
    trainer_cfg = _to_trainer_cfg(cm)
    print("="*60)
    print("TRAINER CFG (flattened):")
    print(json.dumps(trainer_cfg, indent=2))
    print("="*60)

    # init trainer (this constructs the dual-channel model internally)
    trainer = Trainer(trainer_cfg)

    # (optional) show param count
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Model params: {total_params/1e6:.2f}M (trainable: {trainable_params/1e6:.2f}M)")

    # train → val
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start training")
    trainer.fit(train_loader, val_loader)

    # test best
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing best model")
    metrics = trainer.test(test_loader)
    print("Test metrics:", metrics)

if __name__ == "__main__":
    main()
