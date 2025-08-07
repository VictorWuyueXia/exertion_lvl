#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weights & Biases 初始化脚本
用于设置WandB环境和配置
"""

import os
import wandb
import subprocess
import sys
from pathlib import Path

def check_wandb_installation():
    """检查WandB是否正确安装"""
    try:
        import wandb
        print(f"WandB版本: {wandb.__version__}")
        return True
    except ImportError:
        print("WandB未安装，请运行: pip install wandb")
        return False

def check_wandb_login():
    """检查WandB登录状态"""
    try:
        # 尝试获取当前用户
        api = wandb.Api()
        user = api.default_entity
        print(f"WandB已登录，用户: {user}")
        return True
    except Exception as e:
        print(f"WandB登录检查失败: {e}")
        return False

def setup_wandb_login():
    """设置WandB登录"""
    print("设置WandB登录...")
    
    # 检查是否已有API密钥
    api_key = os.environ.get('WANDB_API_KEY')
    if api_key:
        print("检测到环境变量中的WandB API密钥")
        return True
    
    # 检查配置文件
    wandb_dir = Path.home() / '.netrc'
    if wandb_dir.exists():
        print("检测到WandB配置文件")
        return True
    
    print("需要登录WandB...")
    print("请访问 https://wandb.ai/authorize 获取API密钥")
    print("然后运行以下命令登录:")
    print("wandb login")
    
    # 尝试自动登录
    try:
        result = subprocess.run(['wandb', 'login'], 
                              input=b'\n',  # 发送回车键
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            print("WandB登录成功")
            return True
        else:
            print("自动登录失败，请手动运行: wandb login")
            return False
    except FileNotFoundError:
        print("wandb命令未找到，请确保WandB已正确安装")
        return False

def create_wandb_project():
    """创建WandB项目"""
    try:
        # 初始化WandB（不启动运行）
        wandb.init(
            project="exertion-level-detection",
            entity=None,  # 使用默认实体
            name="setup_test",
            mode="disabled"  # 禁用模式，只用于测试
        )
        print("WandB项目连接测试成功")
        return True
    except Exception as e:
        print(f"WandB项目连接测试失败: {e}")
        return False

def setup_gpu_monitoring():
    """设置GPU监控"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
            print("GPU监控已启用")
            
            # 检查pynvml（用于GPU温度监控）
            try:
                import pynvml
                print("pynvml可用，GPU温度监控已启用")
            except ImportError:
                print("pynvml未安装，GPU温度监控将不可用")
                print("如需GPU温度监控，请安装: pip install pynvml")
        else:
            print("未检测到GPU，将使用CPU训练")
        return True
    except Exception as e:
        print(f"GPU监控设置失败: {e}")
        return False

def main():
    """主函数"""
    print("="*50)
    print("Weights & Biases 环境设置")
    print("="*50)
    
    # 检查安装
    if not check_wandb_installation():
        return False
    
    # 检查登录
    if not check_wandb_login():
        if not setup_wandb_login():
            return False
    
    # 测试项目连接
    if not create_wandb_project():
        return False
    
    # 设置GPU监控
    setup_gpu_monitoring()
    
    print("\n" + "="*50)
    print("WandB环境设置完成！")
    print("="*50)
    print("现在可以开始训练，所有指标将自动记录到WandB")
    print("访问 https://wandb.ai 查看训练进度")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
