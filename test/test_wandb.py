#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weights & Biases 测试脚本
验证WandB配置和监控功能
"""

import os
import sys
import time
import torch
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.wandb_manager import WandBManager

def test_wandb_manager():
    """测试WandB管理器"""
    print("测试WandB管理器...")
    
    # 创建测试配置
    test_config = {
        'wandb': {
            'enabled': True,
            'project': 'exertion-level-detection-test',
            'name': 'test_run_{timestamp}',
            'tags': ['test'],
            'description': 'WandB功能测试',
            'monitoring': {
                'gpu_interval': 5,  # 5秒间隔
                'system_interval': 10,  # 10秒间隔
                'gpu_usage': True,
                'memory_usage': True,
                'cpu_usage': True
            }
        },
        'training': {
            'epochs': 2,
            'batch_size': 4,
            'learning_rate': 1e-4
        },
        'system': {
            'seed': 42
        }
    }
    
    try:
        # 创建WandB管理器
        wandb_manager = WandBManager(test_config)
        
        if not wandb_manager.enabled:
            print("WandB已禁用，跳过测试")
            return True
        
        print("WandB管理器创建成功")
        
        # 测试指标记录
        print("测试指标记录...")
        for i in range(5):
            # 模拟训练指标
            train_metrics = {
                'loss': 1.0 - i * 0.1,
                'accuracy': 0.5 + i * 0.1
            }
            
            # 模拟验证指标
            val_metrics = {
                'loss': 1.1 - i * 0.08,
                'accuracy': 0.48 + i * 0.08
            }
            
            wandb_manager.log_training_metrics(train_metrics, step=i)
            wandb_manager.log_validation_metrics(val_metrics, step=i)
            wandb_manager.log_learning_rate(1e-4 * (0.9 ** i), step=i)
            
            print(f"Step {i}: 训练损失={train_metrics['loss']:.3f}, 验证准确率={val_metrics['accuracy']:.3f}")
            time.sleep(2)  # 等待2秒
        
        # 测试模型信息记录
        if torch.cuda.is_available():
            print("测试模型信息记录...")
            # 创建一个简单的测试模型
            test_model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 2)
            )
            wandb_manager.log_model_info(test_model)
        
        # 测试预测结果记录
        print("测试预测结果记录...")
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        wandb_manager.log_predictions(y_true, y_pred, epoch=1)
        
        # 等待一段时间让监控线程运行
        print("等待监控数据收集...")
        time.sleep(10)
        
        # 结束WandB会话
        wandb_manager.finish()
        
        print("WandB测试完成！")
        return True
        
    except Exception as e:
        print(f"WandB测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_monitoring():
    """测试GPU监控"""
    print("\n测试GPU监控...")
    
    if not torch.cuda.is_available():
        print("未检测到GPU，跳过GPU监控测试")
        return True
    
    try:
        # 获取GPU信息
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU: {gpu_name}")
        print(f"GPU内存: {gpu_memory:.1f}GB")
        
        # 测试GPU使用率
        print("测试GPU使用率...")
        for i in range(3):
            # 创建一些张量来占用GPU内存
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            
            # 获取GPU使用情况
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            
            print(f"GPU内存使用: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
            time.sleep(1)
        
        # 清理GPU内存
        del x, y, z
        torch.cuda.empty_cache()
        
        print("GPU监控测试完成")
        return True
        
    except Exception as e:
        print(f"GPU监控测试失败: {e}")
        return False

def main():
    """主函数"""
    print("="*50)
    print("Weights & Biases 功能测试")
    print("="*50)
    
    # 测试WandB管理器
    success1 = test_wandb_manager()
    
    # 测试GPU监控
    success2 = test_gpu_monitoring()
    
    print("\n" + "="*50)
    if success1 and success2:
        print("所有测试通过！WandB配置正确")
        print("现在可以开始训练，所有指标将自动记录到WandB")
    else:
        print("部分测试失败，请检查配置")
    print("="*50)

if __name__ == "__main__":
    main()
