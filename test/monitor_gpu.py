#!/usr/bin/env python3
"""
简单的GPU监控脚本
"""

import torch
import time
import psutil
import os

def monitor_gpu():
    """监控GPU使用情况"""
    print("=== GPU 监控 ===")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 获取GPU利用率
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"GPU 利用率: {utilization.gpu}%")
            print(f"GPU 内存利用率: {utilization.memory}%")
        except ImportError:
            print("pynvml 未安装，无法获取GPU利用率")
        
        # 获取内存使用情况
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3     # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        print(f"GPU 内存使用:")
        print(f"  已分配: {allocated:.2f} GB")
        print(f"  已缓存: {cached:.2f} GB")
        print(f"  总内存: {total:.2f} GB")
        print(f"  可用内存: {total - cached:.2f} GB")
    
    # 获取CPU和系统信息
    print(f"\n=== 系统信息 ===")
    print(f"CPU 使用率: {psutil.cpu_percent()}%")
    print(f"内存使用率: {psutil.virtual_memory().percent}%")
    print(f"当前进程数: {len(psutil.pids())}")

if __name__ == "__main__":
    monitor_gpu() 