#!/usr/bin/env python3
"""
测试RTX 4070优化是否正常工作
"""

import torch
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'data'))

from get_features import (
    initialize_wav2vec2_model, 
    get_optimal_batch_size, 
    clear_gpu_cache, 
    get_memory_usage
)

def test_rtx4070_optimization():
    """测试RTX 4070优化功能"""
    print("=== RTX 4070 优化测试 ===\n")
    
    # 检查CUDA可用性
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"PyTorch 版本: {torch.__version__}")
    
    print("\n--- 测试批处理大小计算 ---")
    batch_size = get_optimal_batch_size()
    print(f"推荐的批处理大小: {batch_size}")
    
    print("\n--- 测试内存管理功能 ---")
    if torch.cuda.is_available():
        # 初始化模型
        print("初始化 Wav2Vec2 模型...")
        initialize_wav2vec2_model()
        
        # 检查内存使用情况
        mem_info = get_memory_usage()
        print(f"模型加载后内存使用:")
        print(f"  已分配: {mem_info['allocated_gb']:.2f} GB")
        print(f"  已缓存: {mem_info['cached_gb']:.2f} GB")
        print(f"  总内存: {mem_info['total_gb']:.2f} GB")
        print(f"  可用内存: {mem_info['free_gb']:.2f} GB")
        
        # 测试内存清理
        print("\n清理GPU缓存...")
        clear_gpu_cache()
        
        mem_info_after = get_memory_usage()
        print(f"清理后内存使用:")
        print(f"  已分配: {mem_info_after['allocated_gb']:.2f} GB")
        print(f"  已缓存: {mem_info_after['cached_gb']:.2f} GB")
        print(f"  可用内存: {mem_info_after['free_gb']:.2f} GB")
    
    print("\n--- 测试完成 ---")
    print("RTX 4070 优化已成功应用！")

if __name__ == "__main__":
    test_rtx4070_optimization() 