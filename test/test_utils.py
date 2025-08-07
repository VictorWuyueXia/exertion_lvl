#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具模块单元测试
测试配置管理、随机种子设置、WandB管理等功能
"""

import sys
import os
import tempfile
import shutil
import unittest
import torch
import numpy as np
import yaml
import json
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_manager import ConfigManager
from src.utils.set_seed import set_seed, seed_worker
from src.utils.wandb_manager import WandBManager

class TestConfigManager(unittest.TestCase):
    """配置管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")
        
        # 创建测试配置文件
        self.test_config = {
            'data': {
                'feature_dir': 'data/features',
                'label_file': 'data/labels.csv',
                'features': {
                    'use_acoustic': True,
                    'use_mfb': True,
                    'use_wav2vec2': True,
                    'wav2vec2_layers': [4]
                }
            },
            'model': {
                'mfb_dim': 40,
                'mfcc_dim': 13,
                'wav2vec2_dim': 768,
                'num_classes': 5,
                'dropout_rate': 0.5,
                'use_mfcc': True,
                'use_wav2vec2': True,
                'optimize_for_rtx4070': False
            },
            'training': {
                'batch_size': 8,
                'epochs': 100,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'optimizer': 'adamw',
                'scheduler': 'cosine',
                'criterion': 'cross_entropy',
                'use_amp': True,
                'patience': 20
            },
            'system': {
                'seed': 42,
                'num_workers': 4,
                'pin_memory': True
            },
            'wandb': {
                'enabled': True,
                'project': 'exertion-level-detection',
                'name': 'test_run',
                'tags': ['test'],
                'description': '测试运行'
            }
        }
        
        # 保存配置文件
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.test_config, f, default_flow_style=False, allow_unicode=True)
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def test_config_manager_initialization(self):
        """测试配置管理器初始化"""
        print("测试配置管理器初始化...")
        
        config_manager = ConfigManager(self.config_file)
        
        self.assertIsInstance(config_manager, ConfigManager)
        self.assertEqual(config_manager.config_file, self.config_file)
        self.assertEqual(config_manager.config, self.test_config)
        
        print("配置管理器初始化测试通过")
    
    def test_get_method(self):
        """测试get方法"""
        print("测试get方法...")
        
        config_manager = ConfigManager(self.config_file)
        
        # 测试获取简单值
        batch_size = config_manager.get('training.batch_size')
        self.assertEqual(batch_size, 8)
        
        # 测试获取嵌套值
        use_acoustic = config_manager.get('data.features.use_acoustic')
        self.assertTrue(use_acoustic)
        
        # 测试获取默认值
        default_value = config_manager.get('nonexistent.key', 'default')
        self.assertEqual(default_value, 'default')
        
        # 测试获取不存在的值
        nonexistent = config_manager.get('nonexistent.key')
        self.assertIsNone(nonexistent)
        
        print("get方法测试通过")
    
    def test_set_method(self):
        """测试set方法"""
        print("测试set方法...")
        
        config_manager = ConfigManager(self.config_file)
        
        # 测试设置简单值
        config_manager.set('training.batch_size', 16)
        new_batch_size = config_manager.get('training.batch_size')
        self.assertEqual(new_batch_size, 16)
        
        # 测试设置嵌套值
        config_manager.set('data.features.use_acoustic', False)
        new_use_acoustic = config_manager.get('data.features.use_acoustic')
        self.assertFalse(new_use_acoustic)
        
        # 测试设置新值
        config_manager.set('new.key', 'new_value')
        new_value = config_manager.get('new.key')
        self.assertEqual(new_value, 'new_value')
        
        print("set方法测试通过")
    
    def test_get_model_config(self):
        """测试获取模型配置"""
        print("测试获取模型配置...")
        
        config_manager = ConfigManager(self.config_file)
        
        model_config = config_manager.get_model_config()
        
        self.assertIsInstance(model_config, dict)
        self.assertIn('mfb_dim', model_config)
        self.assertIn('mfcc_dim', model_config)
        self.assertIn('wav2vec2_dim', model_config)
        self.assertIn('num_classes', model_config)
        self.assertIn('dropout_rate', model_config)
        
        self.assertEqual(model_config['mfb_dim'], 40)
        self.assertEqual(model_config['mfcc_dim'], 13)
        self.assertEqual(model_config['wav2vec2_dim'], 768)
        self.assertEqual(model_config['num_classes'], 5)
        
        print("模型配置获取测试通过")
    
    def test_get_training_config(self):
        """测试获取训练配置"""
        print("测试获取训练配置...")
        
        config_manager = ConfigManager(self.config_file)
        
        training_config = config_manager.get_training_config()
        
        self.assertIsInstance(training_config, dict)
        self.assertIn('batch_size', training_config)
        self.assertIn('epochs', training_config)
        self.assertIn('learning_rate', training_config)
        self.assertIn('optimizer', training_config)
        self.assertIn('scheduler', training_config)
        
        self.assertEqual(training_config['batch_size'], 8)
        self.assertEqual(training_config['epochs'], 100)
        self.assertEqual(training_config['learning_rate'], 1e-4)
        self.assertEqual(training_config['optimizer'], 'adamw')
        
        print("训练配置获取测试通过")
    
    def test_get_data_config(self):
        """测试获取数据配置"""
        print("测试获取数据配置...")
        
        config_manager = ConfigManager(self.config_file)
        
        data_config = config_manager.get_data_config()
        
        self.assertIsInstance(data_config, dict)
        self.assertIn('feature_dir', data_config)
        self.assertIn('label_file', data_config)
        self.assertIn('features', data_config)
        
        self.assertEqual(data_config['feature_dir'], 'data/features')
        self.assertEqual(data_config['label_file'], 'data/labels.csv')
        self.assertTrue(data_config['features']['use_acoustic'])
        
        print("数据配置获取测试通过")
    
    def test_save_config(self):
        """测试保存配置"""
        print("测试保存配置...")
        
        config_manager = ConfigManager(self.config_file)
        
        # 修改配置
        config_manager.set('training.batch_size', 32)
        config_manager.set('new.section', {'key': 'value'})
        
        # 保存到新文件
        new_config_file = os.path.join(self.temp_dir, "new_config.yaml")
        config_manager.save_config(new_config_file)
        
        # 验证文件存在
        self.assertTrue(os.path.exists(new_config_file))
        
        # 重新加载验证
        new_config_manager = ConfigManager(new_config_file)
        new_batch_size = new_config_manager.get('training.batch_size')
        new_value = new_config_manager.get('new.section.key')
        
        self.assertEqual(new_batch_size, 32)
        self.assertEqual(new_value, 'value')
        
        print("配置保存测试通过")
    
    def test_validate_config(self):
        """测试配置验证"""
        print("测试配置验证...")
        
        config_manager = ConfigManager(self.config_file)
        
        # 测试有效配置
        is_valid = config_manager.validate_config()
        self.assertTrue(is_valid)
        
        # 测试无效配置（缺少必需字段）
        config_manager.set('training.batch_size', None)
        is_valid = config_manager.validate_config()
        self.assertFalse(is_valid)
        
        print("配置验证测试通过")
    
    def test_config_merging(self):
        """测试配置合并"""
        print("测试配置合并...")
        
        config_manager = ConfigManager(self.config_file)
        
        # 创建覆盖配置
        override_config = {
            'training': {
                'batch_size': 16,
                'epochs': 50
            },
            'system': {
                'seed': 123
            }
        }
        
        # 合并配置
        merged_config = config_manager.merge_config(override_config)
        
        # 验证合并结果
        self.assertEqual(merged_config['training']['batch_size'], 16)
        self.assertEqual(merged_config['training']['epochs'], 50)
        self.assertEqual(merged_config['system']['seed'], 123)
        
        # 验证原始配置未改变
        original_batch_size = config_manager.get('training.batch_size')
        self.assertEqual(original_batch_size, 8)
        
        print("配置合并测试通过")

class TestSetSeed(unittest.TestCase):
    """随机种子设置测试类"""
    
    def test_set_seed(self):
        """测试设置随机种子"""
        print("测试设置随机种子...")
        
        # 设置种子
        set_seed(42)
        
        # 生成随机数
        torch_rand = torch.rand(5)
        np_rand = np.random.rand(5)
        
        # 重新设置种子
        set_seed(42)
        
        # 再次生成随机数
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        # 验证结果一致
        torch.testing.assert_close(torch_rand, torch_rand2)
        np.testing.assert_array_almost_equal(np_rand, np_rand2)
        
        print("随机种子设置测试通过")
    
    def test_seed_worker(self):
        """测试数据加载器种子工作器"""
        print("测试数据加载器种子工作器...")
        
        # 测试种子工作器函数
        worker_id = 0
        seed_worker(worker_id)
        
        # 验证没有异常
        self.assertTrue(True)
        
        print("种子工作器测试通过")
    
    def test_deterministic_behavior(self):
        """测试确定性行为"""
        print("测试确定性行为...")
        
        # 设置种子
        set_seed(123)
        
        # 创建模型参数
        model = torch.nn.Linear(10, 5)
        initial_weights = model.weight.data.clone()
        
        # 重新设置种子
        set_seed(123)
        
        # 重新创建模型
        model2 = torch.nn.Linear(10, 5)
        initial_weights2 = model2.weight.data.clone()
        
        # 验证权重一致
        torch.testing.assert_close(initial_weights, initial_weights2)
        
        print("确定性行为测试通过")

class TestWandBManager(unittest.TestCase):
    """WandB管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'wandb': {
                'enabled': True,
                'project': 'exertion-level-detection-test',
                'name': 'test_run',
                'tags': ['test'],
                'description': '测试运行',
                'monitoring': {
                    'gpu_interval': 5,
                    'system_interval': 10,
                    'gpu_usage': True,
                    'memory_usage': True,
                    'cpu_usage': True
                }
            },
            'training': {
                'epochs': 10,
                'batch_size': 4,
                'learning_rate': 1e-4
            },
            'system': {
                'seed': 42
            }
        }
    
    @patch('src.utils.wandb_manager.wandb')
    def test_wandb_manager_initialization(self, mock_wandb):
        """测试WandB管理器初始化"""
        print("测试WandB管理器初始化...")
        
        # 模拟wandb.init
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        wandb_manager = WandBManager(self.config)
        
        self.assertIsInstance(wandb_manager, WandBManager)
        self.assertTrue(wandb_manager.enabled)
        self.assertEqual(wandb_manager.project, 'exertion-level-detection-test')
        self.assertEqual(wandb_manager.name, 'test_run')
        
        # 验证wandb.init被调用
        mock_wandb.init.assert_called_once()
        
        print("WandB管理器初始化测试通过")
    
    @patch('src.utils.wandb_manager.wandb')
    def test_wandb_disabled(self, mock_wandb):
        """测试WandB禁用"""
        print("测试WandB禁用...")
        
        config_disabled = self.config.copy()
        config_disabled['wandb']['enabled'] = False
        
        wandb_manager = WandBManager(config_disabled)
        
        self.assertFalse(wandb_manager.enabled)
        
        # 验证wandb.init没有被调用
        mock_wandb.init.assert_not_called()
        
        print("WandB禁用测试通过")
    
    @patch('src.utils.wandb_manager.wandb')
    def test_log_training_metrics(self, mock_wandb):
        """测试记录训练指标"""
        print("测试记录训练指标...")
        
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        wandb_manager = WandBManager(self.config)
        
        # 记录训练指标
        train_metrics = {
            'loss': 0.5,
            'accuracy': 0.85,
            'learning_rate': 1e-4
        }
        
        wandb_manager.log_training_metrics(train_metrics, step=1)
        
        # 验证wandb.log被调用
        mock_run.log.assert_called()
        
        print("训练指标记录测试通过")
    
    @patch('src.utils.wandb_manager.wandb')
    def test_log_validation_metrics(self, mock_wandb):
        """测试记录验证指标"""
        print("测试记录验证指标...")
        
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        wandb_manager = WandBManager(self.config)
        
        # 记录验证指标
        val_metrics = {
            'val_loss': 0.6,
            'val_accuracy': 0.82,
            'val_f1': 0.81
        }
        
        wandb_manager.log_validation_metrics(val_metrics, step=1)
        
        # 验证wandb.log被调用
        mock_run.log.assert_called()
        
        print("验证指标记录测试通过")
    
    @patch('src.utils.wandb_manager.wandb')
    def test_log_model_info(self, mock_wandb):
        """测试记录模型信息"""
        print("测试记录模型信息...")
        
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        wandb_manager = WandBManager(self.config)
        
        # 创建测试模型
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # 记录模型信息
        wandb_manager.log_model_info(model)
        
        # 验证wandb.log被调用
        mock_run.log.assert_called()
        
        print("模型信息记录测试通过")
    
    @patch('src.utils.wandb_manager.wandb')
    def test_log_predictions(self, mock_wandb):
        """测试记录预测结果"""
        print("测试记录预测结果...")
        
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        wandb_manager = WandBManager(self.config)
        
        # 创建测试数据
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        # 记录预测结果
        wandb_manager.log_predictions(y_true, y_pred, epoch=1)
        
        # 验证wandb.log被调用
        mock_run.log.assert_called()
        
        print("预测结果记录测试通过")
    
    @patch('src.utils.wandb_manager.wandb')
    def test_finish(self, mock_wandb):
        """测试结束WandB运行"""
        print("测试结束WandB运行...")
        
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        wandb_manager = WandBManager(self.config)
        
        # 结束运行
        wandb_manager.finish()
        
        # 验证wandb.finish被调用
        mock_run.finish.assert_called_once()
        
        print("WandB结束测试通过")

class TestUtilsIntegration(unittest.TestCase):
    """工具集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "integration_config.yaml")
        
        # 创建集成测试配置
        self.integration_config = {
            'data': {
                'feature_dir': 'data/features',
                'label_file': 'data/labels.csv',
                'features': {
                    'use_acoustic': True,
                    'use_mfb': True,
                    'use_wav2vec2': True,
                    'wav2vec2_layers': [4]
                }
            },
            'model': {
                'mfb_dim': 40,
                'mfcc_dim': 13,
                'wav2vec2_dim': 768,
                'num_classes': 5,
                'dropout_rate': 0.5,
                'use_mfcc': True,
                'use_wav2vec2': True,
                'optimize_for_rtx4070': False
            },
            'training': {
                'batch_size': 8,
                'epochs': 10,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'optimizer': 'adamw',
                'scheduler': 'cosine',
                'criterion': 'cross_entropy',
                'use_amp': True,
                'patience': 5
            },
            'system': {
                'seed': 42,
                'num_workers': 2,
                'pin_memory': True
            },
            'wandb': {
                'enabled': False,  # 禁用WandB进行测试
                'project': 'exertion-level-detection-test',
                'name': 'integration_test',
                'tags': ['integration', 'test'],
                'description': '集成测试'
            }
        }
        
        # 保存配置文件
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.integration_config, f, default_flow_style=False, allow_unicode=True)
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def test_full_utils_integration(self):
        """测试完整工具集成"""
        print("测试完整工具集成...")
        
        # 1. 测试配置管理
        config_manager = ConfigManager(self.config_file)
        self.assertIsInstance(config_manager, ConfigManager)
        
        # 2. 测试随机种子设置
        set_seed(42)
        initial_rand = torch.rand(5)
        
        # 3. 测试WandB管理器（禁用状态）
        wandb_manager = WandBManager(self.integration_config)
        self.assertFalse(wandb_manager.enabled)
        
        # 4. 测试配置获取
        model_config = config_manager.get_model_config()
        training_config = config_manager.get_training_config()
        data_config = config_manager.get_data_config()
        
        self.assertIsInstance(model_config, dict)
        self.assertIsInstance(training_config, dict)
        self.assertIsInstance(data_config, dict)
        
        # 5. 测试配置修改和保存
        config_manager.set('training.batch_size', 16)
        new_config_file = os.path.join(self.temp_dir, "modified_config.yaml")
        config_manager.save_config(new_config_file)
        
        # 6. 验证修改
        new_config_manager = ConfigManager(new_config_file)
        new_batch_size = new_config_manager.get('training.batch_size')
        self.assertEqual(new_batch_size, 16)
        
        # 7. 测试确定性
        set_seed(42)
        final_rand = torch.rand(5)
        torch.testing.assert_close(initial_rand, final_rand)
        
        print("完整工具集成测试通过")
    
    def test_error_handling(self):
        """测试错误处理"""
        print("测试错误处理...")
        
        # 测试不存在的配置文件
        with self.assertRaises(FileNotFoundError):
            ConfigManager("nonexistent_config.yaml")
        
        # 测试无效的配置文件
        invalid_config_file = os.path.join(self.temp_dir, "invalid_config.yaml")
        with open(invalid_config_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with self.assertRaises(Exception):
            ConfigManager(invalid_config_file)
        
        print("错误处理测试通过")

def run_utils_tests():
    """运行工具测试"""
    print("开始运行工具模块单元测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.makeSuite(TestConfigManager))
    test_suite.addTest(unittest.makeSuite(TestSetSeed))
    test_suite.addTest(unittest.makeSuite(TestWandBManager))
    test_suite.addTest(unittest.makeSuite(TestUtilsIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果
    print(f"\n测试结果:")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败测试数: {len(result.failures)}")
    print(f"错误测试数: {len(result.errors)}")
    
    if result.failures:
        print("\n失败测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    success = run_utils_tests()
    if success:
        print("所有工具测试通过!")
    else:
        print("部分工具测试失败!")
