#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练模块单元测试
测试训练器、优化器、损失函数、交叉验证等功能
"""

import sys
import os
import tempfile
import shutil
import unittest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import ExertionTrainer, FocalLoss
from src.models.vgg16_exertion import create_model
from src.data.loader import AudioFeatureDataset

class TestExertionTrainer(unittest.TestCase):
    """运动强度训练器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 测试配置
        self.config = {
            'mfb_dim': 40,
            'mfcc_dim': 13,
            'wav2vec2_dim': 768,
            'num_classes': 5,
            'dropout_rate': 0.5,
            'use_mfcc': True,
            'use_wav2vec2': True,
            'optimize_for_rtx4070': False,
            
            # 训练配置
            'batch_size': 4,
            'epochs': 5,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'criterion': 'cross_entropy',
            'use_amp': False,
            'patience': 3,
            
            # 系统配置
            'seed': 42,
            'num_workers': 0,
            'pin_memory': False,
            
            # 交叉验证配置
            'n_folds': 3,
            'test_size': 0.2,
        }
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试数据
        self._create_test_data()
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """创建测试数据"""
        # 创建特征目录
        feature_dir = os.path.join(self.temp_dir, "features")
        os.makedirs(feature_dir, exist_ok=True)
        
        # 创建测试会话
        for i in range(6):
            session_dir = os.path.join(feature_dir, f"session_{i}")
            os.makedirs(session_dir, exist_ok=True)
            
            # 创建MFCC特征
            mfcc_data = np.random.randn(100, self.config['mfcc_dim'])
            np.save(os.path.join(session_dir, "mfcc.npy"), mfcc_data)
            
            # 创建wav2vec2特征
            wav2vec2_data = np.random.randn(100, self.config['wav2vec2_dim'])
            np.save(os.path.join(session_dir, "wav2vec2.npy"), wav2vec2_data)
        
        # 创建标签数据
        labels_data = []
        for i in range(6):
            labels_data.append({
                'session_id': f'session_{i}',
                'participant_id': f'P{i:03d}',
                'exertion_level': i % 5,
                'speed': ['slow', 'medium', 'fast'][i % 3],
                'task': ['walking', 'running', 'jumping'][i % 3]
            })
        
        self.labels_df = pd.DataFrame(labels_data)
        self.labels_df.to_csv(os.path.join(self.temp_dir, "labels.csv"), index=False)
    
    def test_trainer_initialization(self):
        """测试训练器初始化"""
        print("测试训练器初始化...")
        
        trainer = ExertionTrainer(self.config)
        
        self.assertIsInstance(trainer, ExertionTrainer)
        self.assertEqual(trainer.device, self.device)
        self.assertIsInstance(trainer.train_history, list)
        self.assertIsInstance(trainer.val_history, list)
        
        print("训练器初始化测试通过")
    
    def test_optimizer_creation(self):
        """测试优化器创建"""
        print("测试优化器创建...")
        
        trainer = ExertionTrainer(self.config)
        model = create_model(self.config)
        
        # 测试AdamW优化器
        optimizer = trainer._get_optimizer(model)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        
        # 测试Adam优化器
        config_adam = self.config.copy()
        config_adam['optimizer'] = 'adam'
        trainer_adam = ExertionTrainer(config_adam)
        optimizer_adam = trainer_adam._get_optimizer(model)
        self.assertIsInstance(optimizer_adam, torch.optim.Adam)
        
        # 测试SGD优化器
        config_sgd = self.config.copy()
        config_sgd['optimizer'] = 'sgd'
        trainer_sgd = ExertionTrainer(config_sgd)
        optimizer_sgd = trainer_sgd._get_optimizer(model)
        self.assertIsInstance(optimizer_sgd, torch.optim.SGD)
        
        print("优化器创建测试通过")
    
    def test_scheduler_creation(self):
        """测试学习率调度器创建"""
        print("测试学习率调度器创建...")
        
        trainer = ExertionTrainer(self.config)
        model = create_model(self.config)
        optimizer = trainer._get_optimizer(model)
        
        # 测试Cosine调度器
        scheduler = trainer._get_scheduler(optimizer)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        
        # 测试Step调度器
        config_step = self.config.copy()
        config_step['scheduler'] = 'step'
        trainer_step = ExertionTrainer(config_step)
        scheduler_step = trainer_step._get_scheduler(optimizer)
        self.assertIsInstance(scheduler_step, torch.optim.lr_scheduler.StepLR)
        
        print("学习率调度器创建测试通过")
    
    def test_criterion_creation(self):
        """测试损失函数创建"""
        print("测试损失函数创建...")
        
        trainer = ExertionTrainer(self.config)
        
        # 测试交叉熵损失
        criterion = trainer._get_criterion()
        self.assertIsInstance(criterion, nn.CrossEntropyLoss)
        
        # 测试Focal Loss
        config_focal = self.config.copy()
        config_focal['criterion'] = 'focal'
        trainer_focal = ExertionTrainer(config_focal)
        criterion_focal = trainer_focal._get_criterion()
        self.assertIsInstance(criterion_focal, FocalLoss)
        
        print("损失函数创建测试通过")
    
    def test_focal_loss(self):
        """测试Focal Loss"""
        print("测试Focal Loss...")
        
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        
        # 创建测试数据
        batch_size = 4
        num_classes = 5
        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # 计算损失
        loss = focal_loss(inputs, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
        
        print(f"Focal Loss: {loss.item():.4f}")
    
    def test_train_epoch(self):
        """测试训练一个epoch"""
        print("测试训练一个epoch...")
        
        trainer = ExertionTrainer(self.config)
        model = create_model(self.config).to(self.device)
        optimizer = trainer._get_optimizer(model)
        criterion = trainer._get_criterion()
        
        # 创建测试数据加载器
        from torch.utils.data import DataLoader, TensorDataset
        
        batch_size = 4
        seq_len = 100
        num_samples = 16
        
        # 创建模拟数据
        mfcc_data = torch.randn(num_samples, self.config['mfcc_dim'], seq_len)
        wav2vec2_data = torch.randn(num_samples, self.config['wav2vec2_dim'], seq_len)
        labels = torch.randint(0, self.config['num_classes'], (num_samples,))
        
        dataset = TensorDataset(mfcc_data, wav2vec2_data, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 训练一个epoch
        model.train()
        train_loss, train_acc = trainer.train_epoch(
            model, dataloader, optimizer, criterion, epoch=1
        )
        
        self.assertIsInstance(train_loss, float)
        self.assertIsInstance(train_acc, float)
        self.assertGreater(train_loss, 0)
        self.assertGreaterEqual(train_acc, 0)
        self.assertLessEqual(train_acc, 1)
        
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
    
    def test_validate_epoch(self):
        """测试验证一个epoch"""
        print("测试验证一个epoch...")
        
        trainer = ExertionTrainer(self.config)
        model = create_model(self.config).to(self.device)
        criterion = trainer._get_criterion()
        
        # 创建测试数据加载器
        from torch.utils.data import DataLoader, TensorDataset
        
        batch_size = 4
        seq_len = 100
        num_samples = 12
        
        # 创建模拟数据
        mfcc_data = torch.randn(num_samples, self.config['mfcc_dim'], seq_len)
        wav2vec2_data = torch.randn(num_samples, self.config['wav2vec2_dim'], seq_len)
        labels = torch.randint(0, self.config['num_classes'], (num_samples,))
        
        dataset = TensorDataset(mfcc_data, wav2vec2_data, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 验证一个epoch
        model.eval()
        val_loss, val_acc, val_metrics = trainer.validate_epoch(
            model, dataloader, criterion, epoch=1
        )
        
        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(val_acc, float)
        self.assertIsInstance(val_metrics, dict)
        self.assertGreater(val_loss, 0)
        self.assertGreaterEqual(val_acc, 0)
        self.assertLessEqual(val_acc, 1)
        
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        print(f"验证指标: {val_metrics}")
    
    def test_model_creation(self):
        """测试模型创建"""
        print("测试模型创建...")
        
        trainer = ExertionTrainer(self.config)
        model = trainer._create_model(self.config)
        
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.input_dim, 
                        self.config['mfcc_dim'] + self.config['wav2vec2_dim'])
        
        print("模型创建测试通过")
    
    def test_dataloader_creation(self):
        """测试数据加载器创建"""
        print("测试数据加载器创建...")
        
        trainer = ExertionTrainer(self.config)
        
        # 创建模拟数据集
        from src.data.loader import get_session_metadata
        
        feature_dir = os.path.join(self.temp_dir, "features")
        metadata_df = get_session_metadata(feature_dir)
        
        dataset = AudioFeatureDataset(
            metadata_df=metadata_df,
            feature_dir=feature_dir,
            labels_df=self.labels_df,
            use_acoustic=True,
            use_mfb=False,
            use_embed=True,
            selected_wav2vec2_layers=(4,)
        )
        
        # 创建数据加载器
        session_ids = metadata_df['session'].tolist()
        dataloader = trainer._create_dataloader(
            dataset, session_ids, self.config, shuffle=True
        )
        
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)
        
        # 测试数据加载
        for batch in dataloader:
            self.assertIn('mfcc', batch)
            self.assertIn('wav2vec2', batch)
            self.assertIn('exertion_level', batch)
            break
        
        print("数据加载器创建测试通过")
    
    @patch('src.training.trainer.ExertionTrainer._create_dataloader')
    @patch('src.training.trainer.ExertionTrainer._create_model')
    def test_train_fold(self, mock_create_model, mock_create_dataloader):
        """测试训练单个fold"""
        print("测试训练单个fold...")
        
        # 模拟数据加载器
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_create_dataloader.side_effect = [mock_train_loader, mock_val_loader]
        
        # 模拟模型
        mock_model = create_model(self.config).to(self.device)
        mock_create_model.return_value = mock_model
        
        # 模拟训练数据
        batch_size = 4
        seq_len = 100
        num_samples = 16
        
        mfcc_data = torch.randn(num_samples, self.config['mfcc_dim'], seq_len)
        wav2vec2_data = torch.randn(num_samples, self.config['wav2vec2_dim'], seq_len)
        labels = torch.randint(0, self.config['num_classes'], (num_samples,))
        
        # 模拟数据加载器的迭代
        def mock_iter():
            for i in range(0, num_samples, batch_size):
                yield {
                    'mfcc': mfcc_data[i:i+batch_size],
                    'wav2vec2': wav2vec2_data[i:i+batch_size],
                    'exertion_level': labels[i:i+batch_size]
                }
        
        mock_train_loader.__iter__ = mock_iter
        mock_val_loader.__iter__ = mock_iter
        
        trainer = ExertionTrainer(self.config)
        
        # 训练一个fold
        fold_results = trainer.train_fold(0, mock_train_loader, mock_val_loader, mock_model, self.config)
        
        self.assertIsInstance(fold_results, dict)
        self.assertIn('train_losses', fold_results)
        self.assertIn('val_losses', fold_results)
        self.assertIn('train_accs', fold_results)
        self.assertIn('val_accs', fold_results)
        self.assertIn('best_model_path', fold_results)
        
        print("单个fold训练测试通过")
    
    def test_cross_validation_split(self):
        """测试交叉验证数据分割"""
        print("测试交叉验证数据分割...")
        
        from src.data.loader import get_session_metadata, greedy_grouped_split
        
        feature_dir = os.path.join(self.temp_dir, "features")
        metadata_df = get_session_metadata(feature_dir)
        
        # 测试数据分割
        folds = greedy_grouped_split(metadata_df, n_folds=3, segment_duration=15.0)
        
        self.assertIsInstance(folds, list)
        self.assertEqual(len(folds), 3)
        
        # 检查每个fold
        for i, fold in enumerate(folds):
            self.assertIsInstance(fold, dict)
            self.assertIn('train', fold)
            self.assertIn('val', fold)
            self.assertGreater(len(fold['train']), 0)
            self.assertGreater(len(fold['val']), 0)
            
            print(f"Fold {i}: 训练集 {len(fold['train'])} 个样本, 验证集 {len(fold['val'])} 个样本")
        
        print("交叉验证数据分割测试通过")

class TestTrainingIntegration(unittest.TestCase):
    """训练集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp_dir = tempfile.mkdtemp()
        
        self.config = {
            'mfb_dim': 40,
            'mfcc_dim': 13,
            'wav2vec2_dim': 768,
            'num_classes': 5,
            'dropout_rate': 0.5,
            'use_mfcc': True,
            'use_wav2vec2': True,
            'optimize_for_rtx4070': False,
            
            # 训练配置
            'batch_size': 2,
            'epochs': 2,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'criterion': 'cross_entropy',
            'use_amp': False,
            'patience': 2,
            
            # 系统配置
            'seed': 42,
            'num_workers': 0,
            'pin_memory': False,
            
            # 交叉验证配置
            'n_folds': 2,
            'test_size': 0.2,
        }
        
        self._create_test_data()
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """创建测试数据"""
        # 创建特征目录
        feature_dir = os.path.join(self.temp_dir, "features")
        os.makedirs(feature_dir, exist_ok=True)
        
        # 创建测试会话
        for i in range(8):
            session_dir = os.path.join(feature_dir, f"session_{i}")
            os.makedirs(session_dir, exist_ok=True)
            
            # 创建MFCC特征
            mfcc_data = np.random.randn(100, self.config['mfcc_dim'])
            np.save(os.path.join(session_dir, "mfcc.npy"), mfcc_data)
            
            # 创建wav2vec2特征
            wav2vec2_data = np.random.randn(100, self.config['wav2vec2_dim'])
            np.save(os.path.join(session_dir, "wav2vec2.npy"), wav2vec2_data)
        
        # 创建标签数据
        labels_data = []
        for i in range(8):
            labels_data.append({
                'session_id': f'session_{i}',
                'participant_id': f'P{i:03d}',
                'exertion_level': i % 5,
                'speed': ['slow', 'medium', 'fast'][i % 3],
                'task': ['walking', 'running', 'jumping'][i % 3]
            })
        
        self.labels_df = pd.DataFrame(labels_data)
        self.labels_df.to_csv(os.path.join(self.temp_dir, "labels.csv"), index=False)
    
    @patch('src.training.trainer.ExertionTrainer._create_dataloader')
    @patch('src.training.trainer.ExertionTrainer._create_model')
    def test_mini_training_cycle(self, mock_create_model, mock_create_dataloader):
        """测试迷你训练周期"""
        print("测试迷你训练周期...")
        
        # 模拟数据加载器
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_create_dataloader.return_value = mock_train_loader
        
        # 模拟模型
        mock_model = create_model(self.config).to(self.device)
        mock_create_model.return_value = mock_model
        
        # 创建模拟数据
        batch_size = 2
        seq_len = 50
        num_samples = 8
        
        mfcc_data = torch.randn(num_samples, self.config['mfcc_dim'], seq_len)
        wav2vec2_data = torch.randn(num_samples, self.config['wav2vec2_dim'], seq_len)
        labels = torch.randint(0, self.config['num_classes'], (num_samples,))
        
        # 模拟数据加载器的迭代
        def mock_iter():
            for i in range(0, num_samples, batch_size):
                yield {
                    'mfcc': mfcc_data[i:i+batch_size],
                    'wav2vec2': wav2vec2_data[i:i+batch_size],
                    'exertion_level': labels[i:i+batch_size]
                }
        
        mock_train_loader.__iter__ = mock_iter
        mock_val_loader.__iter__ = mock_iter
        
        trainer = ExertionTrainer(self.config)
        
        # 训练一个fold
        fold_results = trainer.train_fold(0, mock_train_loader, mock_val_loader, mock_model, self.config)
        
        self.assertIsInstance(fold_results, dict)
        self.assertIn('train_losses', fold_results)
        self.assertIn('val_losses', fold_results)
        
        print("迷你训练周期测试通过")
    
    def test_training_configuration_validation(self):
        """测试训练配置验证"""
        print("测试训练配置验证...")
        
        # 测试有效配置
        valid_config = self.config.copy()
        trainer = ExertionTrainer(valid_config)
        self.assertIsInstance(trainer, ExertionTrainer)
        
        # 测试无效配置（缺少必需参数）
        invalid_config = self.config.copy()
        del invalid_config['batch_size']
        
        with self.assertRaises(KeyError):
            trainer = ExertionTrainer(invalid_config)
        
        print("训练配置验证测试通过")

def run_training_tests():
    """运行训练测试"""
    print("开始运行训练模块单元测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.makeSuite(TestExertionTrainer))
    test_suite.addTest(unittest.makeSuite(TestTrainingIntegration))
    
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
    success = run_training_tests()
    if success:
        print("所有训练测试通过!")
    else:
        print("部分训练测试失败!")
