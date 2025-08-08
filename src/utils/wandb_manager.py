#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weights & Biases 管理器
用于监控训练过程和GPU使用情况
"""

import os
import time
import psutil
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Optional, List
import threading
from collections import defaultdict

class WandBManager:
    """Weights & Biases 管理器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化WandB管理器
        
        Args:
            config: WandB配置字典
        """
        self.config = config
        self.wandb_config = config.get('wandb', {})
        self.enabled = self.wandb_config.get('enabled', True)
        
        if not self.enabled:
            print("WandB已禁用")
            return
        
        # 初始化WandB
        self._init_wandb()
        
        # 监控相关
        self.monitoring_thread = None
        self.stop_monitoring = False
        self.metrics_history = defaultdict(list)
        
        # 启动系统监控
        if self.wandb_config.get('monitoring', {}).get('gpu_usage', True):
            self._start_system_monitoring()
    
    def _init_wandb(self):
        """初始化WandB"""
        try:
            # 生成实验名称
            name_template = self.wandb_config.get('name', 'exertion_detection_{timestamp}')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = name_template.replace('{timestamp}', timestamp)
            
            # 初始化WandB
            wandb.init(
                project=self.wandb_config.get('project', 'exertion-level-detection'),
                entity=self.wandb_config.get('entity'),
                name=experiment_name,
                tags=self.wandb_config.get('tags', []),
                notes=self.wandb_config.get('description', ''),
                config=self.config,
                reinit=True
            )
            
            print(f"WandB初始化成功: {experiment_name}")
            
        except Exception as e:
            print(f"WandB初始化失败: {e}")
            self.enabled = False
    
    def _start_system_monitoring(self):
        """启动系统监控线程"""
        if not self.enabled:
            return
        
        self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitoring_thread.start()
        print("系统监控线程已启动")
    
    def _monitor_system(self):
        """系统监控线程"""
        gpu_interval = self.wandb_config.get('monitoring', {}).get('gpu_interval', 30)
        system_interval = self.wandb_config.get('monitoring', {}).get('system_interval', 60)
        
        last_gpu_log = 0
        last_system_log = 0
        
        while not self.stop_monitoring:
            current_time = time.time()
            
            # GPU监控
            if current_time - last_gpu_log >= gpu_interval:
                self._log_gpu_metrics()
                last_gpu_log = current_time
            
            # 系统监控
            if current_time - last_system_log >= system_interval:
                self._log_system_metrics()
                last_system_log = current_time
            
            time.sleep(1)
    
    def _print_metrics_to_terminal(self, metric_type: str, metrics: Dict[str, float], step: Optional[int] = None):
        """在终端打印格式化的指标信息"""
        if not self.enabled:
            return
        
        # 格式化指标显示
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        step_info = f" (Step: {step})" if step is not None else ""
        
        # 使用颜色编码（如果支持）
        try:
            from colorama import init, Fore, Style
            init()
            
            if metric_type == "训练":
                color = Fore.GREEN
            elif metric_type == "验证":
                color = Fore.BLUE
            else:
                color = Fore.YELLOW
                
            print(f"{color}[WandB {metric_type}]{Style.RESET_ALL} {metric_str}{step_info}")
            
        except ImportError:
            # 如果没有colorama，使用普通输出
            print(f"[WandB {metric_type}] {metric_str}{step_info}")
    
    def _log_gpu_metrics(self):
        """记录GPU指标"""
        if not torch.cuda.is_available():
            return
        
        try:
            # GPU使用率（兼容性处理）
            try:
                gpu_utilization = torch.cuda.utilization(0)
            except:
                gpu_utilization = 0  # 如果不可用，设为0
            
            # GPU内存使用
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3    # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            # GPU温度（如果可用）
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                gpu_temp = None
            
            metrics = {
                'gpu/utilization': gpu_utilization,
                'gpu/memory_allocated_gb': gpu_memory_allocated,
                'gpu/memory_reserved_gb': gpu_memory_reserved,
                'gpu/memory_total_gb': gpu_memory_total,
                'gpu/memory_usage_percent': (gpu_memory_allocated / gpu_memory_total) * 100
            }
            
            if gpu_temp is not None:
                metrics['gpu/temperature_celsius'] = gpu_temp
            
            wandb.log(metrics, step=wandb.run.step if wandb.run else None)
            
            # 终端显示GPU信息（每5次记录一次，避免过于频繁）
            if hasattr(self, '_gpu_log_count'):
                self._gpu_log_count += 1
            else:
                self._gpu_log_count = 1
            
            if self._gpu_log_count % 5 == 0:
                gpu_info = f"GPU使用率: {gpu_utilization}%, 内存: {gpu_memory_allocated:.2f}GB/{gpu_memory_total:.2f}GB"
                if gpu_temp is not None:
                    gpu_info += f", 温度: {gpu_temp}°C"
                print(f"[WandB GPU监控] {gpu_info}")
            
        except Exception as e:
            # 忽略Broken pipe错误，这通常是因为WandB连接中断
            if "Broken pipe" not in str(e):
                print(f"GPU监控错误: {e}")
    
    def _log_system_metrics(self):
        """记录系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / 1024**3
            memory_total_gb = memory.total / 1024**3
            
            # 磁盘使用
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used_gb = disk.used / 1024**3
            disk_total_gb = disk.total / 1024**3
            
            metrics = {
                'system/cpu_percent': cpu_percent,
                'system/memory_percent': memory_percent,
                'system/memory_used_gb': memory_used_gb,
                'system/memory_total_gb': memory_total_gb,
                'system/disk_percent': disk_percent,
                'system/disk_used_gb': disk_used_gb,
                'system/disk_total_gb': disk_total_gb
            }
            
            wandb.log(metrics, step=wandb.run.step if wandb.run else None)
            
        except Exception as e:
            # 忽略Broken pipe错误，这通常是因为WandB连接中断
            if "Broken pipe" not in str(e):
                print(f"系统监控错误: {e}")
    
    def log_training_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """记录训练指标"""
        if not self.enabled:
            return
        
        try:
            # 添加前缀
            prefixed_metrics = {f'train/{k}': v for k, v in metrics.items()}
            wandb.log(prefixed_metrics, step=step)
            
            # 保存到历史记录
            for k, v in metrics.items():
                self.metrics_history[f'train_{k}'].append(v)
            
            # 增强终端显示
            self._print_metrics_to_terminal("训练", metrics, step)
                
        except Exception as e:
            print(f"训练指标记录错误: {e}")
    
    def log_validation_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """记录验证指标"""
        if not self.enabled:
            return
        
        try:
            # 添加前缀
            prefixed_metrics = {f'val/{k}': v for k, v in metrics.items()}
            wandb.log(prefixed_metrics, step=step)
            
            # 保存到历史记录
            for k, v in metrics.items():
                self.metrics_history[f'val_{k}'].append(v)
            
            # 增强终端显示
            self._print_metrics_to_terminal("验证", metrics, step)
                
        except Exception as e:
            print(f"验证指标记录错误: {e}")
    
    def log_test_metrics(self, metrics: Dict[str, float]):
        """记录测试指标"""
        if not self.enabled:
            return
        
        try:
            # 添加前缀
            prefixed_metrics = {f'test/{k}': v for k, v in metrics.items()}
            wandb.log(prefixed_metrics)
            
        except Exception as e:
            print(f"测试指标记录错误: {e}")
    
    def log_model_info(self, model: torch.nn.Module):
        """记录模型信息"""
        if not self.enabled:
            return
        
        try:
            # 计算模型参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 记录模型信息
            wandb.log({
                'model/total_parameters': total_params,
                'model/trainable_parameters': trainable_params,
                'model/model_size_mb': total_params * 4 / 1024**2  # 假设float32
            })
            
        except Exception as e:
            print(f"模型信息记录错误: {e}")
    
    def log_learning_rate(self, lr: float, step: Optional[int] = None):
        """记录学习率"""
        if not self.enabled:
            return
        
        try:
            wandb.log({'train/learning_rate': lr}, step=step)
        except Exception as e:
            print(f"学习率记录错误: {e}")
    
    def log_gradients(self, model: torch.nn.Module, step: Optional[int] = None):
        """记录梯度信息"""
        if not self.enabled:
            return
        
        try:
            grad_norm = 0.0
            param_norm = 0.0
            
            for p in model.parameters():
                if p.grad is not None:
                    param_norm += p.data.norm(2).item() ** 2
                    grad_norm += p.grad.data.norm(2).item() ** 2
            
            param_norm = param_norm ** 0.5
            grad_norm = grad_norm ** 0.5
            
            wandb.log({
                'train/gradient_norm': grad_norm,
                'train/parameter_norm': param_norm,
                'train/gradient_parameter_ratio': grad_norm / (param_norm + 1e-8)
            }, step=step)
            
        except Exception as e:
            print(f"梯度信息记录错误: {e}")
    
    def log_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, epoch: int):
        """记录预测结果"""
        if not self.enabled:
            return
        
        try:
            # 创建混淆矩阵
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # 绘制混淆矩阵
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Epoch {epoch}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # 保存到WandB
            wandb.log({f'confusion_matrix_epoch_{epoch}': wandb.Image(plt)})
            plt.close()
            
        except Exception as e:
            print(f"预测结果记录错误: {e}")
    
    def save_model(self, model: torch.nn.Module, path: str, metadata: Optional[Dict] = None):
        """保存模型到WandB"""
        if not self.enabled:
            return
        
        try:
            # 保存模型文件
            torch.save(model.state_dict(), path)
            
            # 上传到WandB
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                description="训练好的运动强度检测模型"
            )
            artifact.add_file(path)
            
            if metadata:
                artifact.metadata.update(metadata)
            
            wandb.log_artifact(artifact)
            print(f"模型已保存到WandB: {path}")
            
        except Exception as e:
            print(f"模型保存错误: {e}")
    
    def log_config(self, config: Dict[str, Any]):
        """记录配置信息"""
        if not self.enabled:
            return
        
        try:
            wandb.config.update(config)
        except Exception as e:
            print(f"配置记录错误: {e}")
    
    def finish(self):
        """结束WandB会话"""
        if not self.enabled:
            return
        
        try:
            self.stop_monitoring = True
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            
            wandb.finish()
            print("WandB会话已结束")
            
        except Exception as e:
            print(f"WandB结束错误: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.finish()
