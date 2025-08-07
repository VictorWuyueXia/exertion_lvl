#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机种子设置工具
用于确保数据加载器的可重现性
"""

import torch
import numpy as np
import random
import os

def seed_worker(worker_id):
    """为数据加载器的工作进程设置随机种子"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed=42):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
