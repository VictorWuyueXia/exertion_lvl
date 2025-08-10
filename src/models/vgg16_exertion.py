# -*- coding: utf-8 -*-
"""
双通道用力水平检测模型（精简版）
MFCC (轻量VGG-ish) + w2v2中层特征 (Transformer) -> 门控晚融合 -> 注意力池化 -> CORAL有序分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------- 轻量 VGG-ish（更简） -----------------------
class VGG16ishMFCC(nn.Module):
    """
    轻量：2 个 ConvBlock（Conv3x3x2+BN+ReLU），仅在频率轴池化
    Input : [B, C, F, T]  -> Output: [B, T, H]
    """
    def __init__(self, in_ch=1, hid=256, drop=0.1):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )
        self.b1 = block(in_ch, 48);  self.p1 = nn.MaxPool2d((2,1))  # F↓2, T保留
        self.b2 = block(48,   96);   self.p2 = nn.MaxPool2d((2,1))  # F↓2, T保留
        self.drop = nn.Dropout2d(drop)
        self.proj = nn.Linear(96, hid)

    def forward(self, x):            # x: [B, C, F, T]
        h = self.b1(x); h = self.p1(h)
        h = self.b2(h); h = self.p2(h)       # [B, 96, F', T]
        h = self.drop(h)
        h = h.mean(dim=2)                    # 频轴全局平均 -> [B, 96, T]
        h = h.permute(0, 2, 1)               # -> [B, T, 96]
        return self.proj(h)                  # -> [B, T, H]

# ----------------------- w2v2 时间编码器 -----------------------
class W2VTemporal(nn.Module):
    """Input: [B,T,D_w] -> Output: [B,T,H]"""
    def __init__(self, in_dim=768, hid=256, n_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hid)
        layer = nn.TransformerEncoderLayer(d_model=hid, nhead=nhead,
                                           dim_feedforward=hid*2, dropout=dropout,
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
    def forward(self, x, pad_mask=None):
        z = self.in_proj(x)
        return self.enc(z, src_key_padding_mask=pad_mask)

# ----------------------- 注意力池化 -----------------------
class AttnPool(nn.Module):
    """[B,T,H] -> [B,H], [B,T]"""
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.w = nn.Linear(dim, 1)
    def forward(self, H, pad_mask=None):
        H = self.drop(H)
        a = self.w(H).squeeze(-1)                 # [B,T]
        if pad_mask is not None:
            # Use a smaller value compatible with float16
            a = a.masked_fill(pad_mask, -1e4)
        a = torch.softmax(a, dim=1)
        z = (H * a.unsqueeze(-1)).sum(dim=1)      # [B,H]
        return z, a

# ----------------------- CORAL 头 & 工具 -----------------------
class CoralHead(nn.Module):
    def __init__(self, in_dim, K=5, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout),
                                 nn.Linear(in_dim, K-1))
        self.K = K
    def forward(self, z): return self.net(z)      # [B,K-1]

def coral_targets(y, K=5):
    thr = torch.arange(K-1, device=y.device).unsqueeze(0)   # [1,K-1]
    return (y.unsqueeze(1) > thr).float()                   # [B,K-1]

def coral_loss(logits, y, K=5, gamma=None, per_threshold_weight=None, alpha=1.0):
    """
    logits: [B,K-1], y: [B]∈{0..K-1}
    - 修正点：pos_weight 必须是长度 K-1 的向量
    """
    tgt = coral_targets(y, K)                                # [B,K-1]
    if gamma is None:
        pos_weight = None
        if alpha != 1.0:
            pos_weight = torch.full((K-1,), alpha, device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, tgt,
                                                  weight=per_threshold_weight,
                                                  pos_weight=pos_weight)
    # Ordinal Focal
    p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
    bce = -(tgt*torch.log(p) + (1-tgt)*torch.log(1-p))
    pt  = tgt*p + (1-tgt)*(1-p)
    mod = (1-pt).pow(gamma)
    if alpha != 1.0:
        mod = mod * (alpha * tgt + (1-tgt))
    if per_threshold_weight is not None:
        bce = bce * per_threshold_weight
    return (mod * bce).mean()

def coral_predict(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)

def coral_predict_proba(logits, K=5):
    probs = torch.sigmoid(logits)                 # [B,K-1]
    B = probs.size(0)
    class_probs = torch.zeros(B, K, device=logits.device, dtype=probs.dtype)
    class_probs[:, 0]   = 1 - probs[:, 0]
    class_probs[:, -1]  = probs[:, -2]
    for i in range(1, K-1):
        class_probs[:, i] = probs[:, i-1] - probs[:, i]
    return class_probs.clamp(0, 1)

# ----------------------- 整体模型 -----------------------
class DualChannelExertionModel(nn.Module):
    """
    轻量 MFCC 塔 + 小型 Transformer(w2v2) -> 门控晚融合 -> 注意力池化 -> CORAL
    保持 I/O：mfcc:[B,C,F,T], w2v:[B,T,Dw] -> logits:[B,K-1], attn:[B,T]
    """
    def __init__(self,
                 mfcc_dim=40,
                 wav2vec2_dim=768,
                 hidden_dim=256,
                 num_classes=5,
                 in_ch=1):
        super().__init__()
        self.mfcc_tower = VGG16ishMFCC(in_ch=in_ch, hid=hidden_dim, drop=0.15)  # 适中的dropout
        self.w2v_tower  = W2VTemporal(in_dim=wav2vec2_dim, hid=hidden_dim,
                                      n_layers=2, nhead=4, dropout=0.15)  # 适中的dropout
        # 轻量归一化，稳定门控
        self.pre_norm_mfcc = nn.LayerNorm(hidden_dim)
        self.pre_norm_w2v  = nn.LayerNorm(hidden_dim)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.pool = AttnPool(hidden_dim, dropout=0.15)  # 适中的dropout
        self.head = CoralHead(in_dim=hidden_dim, K=num_classes, dropout=0.15)  # 适中的dropout

        self.num_classes = num_classes
        self.hidden_dim  = hidden_dim
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, mfcc, w2v2, mask=None):
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()  # 确保是布尔
        # [B,T,H]
        h_mfcc = self.mfcc_tower(mfcc)
        h_w2v  = self.w2v_tower(w2v2, pad_mask=mask)
        # 预归一化再融合
        h_mfcc = self.pre_norm_mfcc(h_mfcc)
        h_w2v  = self.pre_norm_w2v(h_w2v)

        g = self.gate(torch.cat([h_mfcc, h_w2v], dim=-1))  # [B,T,1]
        H = g*h_mfcc + (1-g)*h_w2v                         # [B,T,H]

        z, attn = self.pool(H, pad_mask=mask)              # [B,H], [B,T]
        logits  = self.head(z)                             # [B,K-1]
        return logits, attn

    def step(self, batch, gamma=None, per_threshold_weight=None, alpha=1.0):
        logits, attn = self.forward(batch['mfcc'], batch['w2v'], batch.get('mask', None))
        loss = coral_loss(logits, batch['y'], K=self.num_classes,
                          gamma=gamma, per_threshold_weight=per_threshold_weight, alpha=alpha)
        pred = coral_predict(logits)
        return loss, pred, attn

    def predict_proba(self, logits): return coral_predict_proba(logits, self.num_classes)
    def get_feature_dim(self): return self.hidden_dim

# ----------------------- 工具 & 工厂 -----------------------
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': total, 'trainable_params': train,
            'total_params_millions': total/1e6, 'trainable_params_millions': train/1e6}

def validate_config(config):
    for k in ['mfcc_dim', 'wav2vec2_dim', 'num_classes']:
        if k not in config: raise ValueError(f"Missing required config key: {k}")
    if config.get('hidden_dim', 256) <= 0: raise ValueError("hidden_dim must be positive")
    if config.get('num_classes', 5) < 2:   raise ValueError("num_classes must be >= 2")

def create_model(config):
    validate_config(config)
    return DualChannelExertionModel(
        mfcc_dim     = config.get('mfcc_dim', 40),
        wav2vec2_dim = config.get('wav2vec2_dim', 768),
        hidden_dim   = config.get('hidden_dim', 256),
        num_classes  = config.get('num_classes', 5),
        in_ch        = config.get('in_ch', 1)
    )
