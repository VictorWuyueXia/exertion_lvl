#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估模块单元测试
测试评估器、指标计算、可视化等功能
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

from src.evaluation.evaluator import ExertionEvaluator
from src.models.vgg16_exertion import create_model

class TestExertionEvaluator(unittest.TestCase):
    """运动强度评估器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.result_dir = os.path.join(self.temp_dir, "result")
        os.makedirs(self.result_dir, exist_ok=True)
        
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
        }
        
        # 创建测试数据
        self._create_test_data()
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """创建测试数据"""
        # 创建测试标签和预测
        np.random.seed(42)
        self.num_samples = 100
        self.num_classes = 5
        
        # 创建真实标签
        self.true_labels = np.random.randint(0, self.num_classes, self.num_samples)
        
        # 创建预测概率（模拟模型输出）
        self.pred_probs = np.random.rand(self.num_samples, self.num_classes)
        # 归一化概率
        self.pred_probs = self.pred_probs / self.pred_probs.sum(axis=1, keepdims=True)
        
        # 创建预测标签
        self.pred_labels = np.argmax(self.pred_probs, axis=1)
    
    def test_evaluator_initialization(self):
        """测试评估器初始化"""
        print("测试评估器初始化...")
        
        evaluator = ExertionEvaluator(self.result_dir, self.config)
        
        self.assertIsInstance(evaluator, ExertionEvaluator)
        self.assertEqual(evaluator.device, self.device)
        self.assertEqual(evaluator.result_dir, self.result_dir)
        self.assertEqual(evaluator.config, self.config)
        
        # 检查评估目录是否创建
        self.assertTrue(os.path.exists(evaluator.eval_dir))
        
        print("评估器初始化测试通过")
    
    def test_metrics_calculation(self):
        """测试指标计算"""
        print("测试指标计算...")
        
        evaluator = ExertionEvaluator(self.result_dir, self.config)
        
        # 计算指标
        metrics = evaluator._calculate_metrics(
            self.true_labels, 
            self.pred_labels, 
            self.pred_probs
        )
        
        # 检查指标类型和范围
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('confusion_matrix', metrics)
        
        # 检查指标范围
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
        self.assertGreaterEqual(metrics['precision'], 0)
        self.assertLessEqual(metrics['precision'], 1)
        self.assertGreaterEqual(metrics['recall'], 0)
        self.assertLessEqual(metrics['recall'], 1)
        self.assertGreaterEqual(metrics['f1_score'], 0)
        self.assertLessEqual(metrics['f1_score'], 1)
        
        # 检查混淆矩阵
        self.assertEqual(metrics['confusion_matrix'].shape, (self.num_classes, self.num_classes))
        
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1_score']:.4f}")
    
    def test_perfect_predictions(self):
        """测试完美预测的指标"""
        print("测试完美预测的指标...")
        
        evaluator = ExertionEvaluator(self.result_dir, self.config)
        
        # 创建完美预测
        perfect_labels = np.arange(self.num_classes)
        perfect_probs = np.zeros((self.num_classes, self.num_classes))
        np.fill_diagonal(perfect_probs, 1.0)
        
        # 计算指标
        metrics = evaluator._calculate_metrics(
            perfect_labels, 
            perfect_labels, 
            perfect_probs
        )
        
        # 完美预测应该有100%的准确率
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1_score'], 1.0)
        
        print("完美预测指标测试通过")
    
    def test_random_predictions(self):
        """测试随机预测的指标"""
        print("测试随机预测的指标...")
        
        evaluator = ExertionEvaluator(self.result_dir, self.config)
        
        # 创建随机预测
        random_labels = np.random.randint(0, self.num_classes, self.num_samples)
        random_probs = np.random.rand(self.num_samples, self.num_classes)
        random_probs = random_probs / random_probs.sum(axis=1, keepdims=True)
        
        # 计算指标
        metrics = evaluator._calculate_metrics(
            self.true_labels, 
            random_labels, 
            random_probs
        )
        
        # 随机预测的准确率应该接近1/num_classes
        expected_accuracy = 1.0 / self.num_classes
        self.assertLess(metrics['accuracy'], 0.5)  # 随机预测准确率应该较低
        
        print(f"随机预测准确率: {metrics['accuracy']:.4f} (期望约 {expected_accuracy:.4f})")
    
    def test_model_evaluation(self):
        """测试模型评估"""
        print("测试模型评估...")
        
        evaluator = ExertionEvaluator(self.result_dir, self.config)
        
        # 创建测试模型
        model = create_model(self.config).to(self.device)
        
        # 创建测试数据加载器
        from torch.utils.data import DataLoader, TensorDataset
        
        batch_size = 4
        seq_len = 100
        num_samples = 20
        
        # 创建模拟数据
        mfcc_data = torch.randn(num_samples, self.config['mfcc_dim'], seq_len)
        wav2vec2_data = torch.randn(num_samples, self.config['wav2vec2_dim'], seq_len)
        labels = torch.randint(0, self.config['num_classes'], (num_samples,))
        
        dataset = TensorDataset(mfcc_data, wav2vec2_data, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 评估模型
        metrics, all_labels, all_preds, all_probs = evaluator.evaluate_model(
            model, dataloader, fold_idx=0
        )
        
        # 检查返回结果
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(all_labels, np.ndarray)
        self.assertIsInstance(all_preds, np.ndarray)
        self.assertIsInstance(all_probs, np.ndarray)
        
        self.assertEqual(len(all_labels), num_samples)
        self.assertEqual(len(all_preds), num_samples)
        self.assertEqual(all_probs.shape, (num_samples, self.config['num_classes']))
        
        print(f"模型评估完成，准确率: {metrics['accuracy']:.4f}")
    
    def test_visualization_generation(self):
        """测试可视化生成"""
        print("测试可视化生成...")
        
        evaluator = ExertionEvaluator(self.result_dir, self.config)
        
        # 生成可视化
        evaluator._generate_visualizations(
            self.true_labels, 
            self.pred_labels, 
            self.pred_probs, 
            fold_idx=0
        )
        
        # 检查生成的文件
        expected_files = [
            'confusion_matrix_fold_0.png',
            'multiclass_roc_fold_0.png',
            'prediction_distribution_fold_0.png',
            'class_accuracy_fold_0.png'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(evaluator.eval_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"文件不存在: {filename}")
            self.assertGreater(os.path.getsize(filepath), 0, f"文件为空: {filename}")
        
        print("可视化生成测试通过")
    
    def test_confusion_matrix_plot(self):
        """测试混淆矩阵绘制"""
        print("测试混淆矩阵绘制...")
        
        evaluator = ExertionEvaluator(self.result_dir, self.config)
        
        # 绘制混淆矩阵
        evaluator._plot_confusion_matrix(self.true_labels, self.pred_labels, fold_idx=1)
        
        # 检查文件
        filepath = os.path.join(evaluator.eval_dir, 'confusion_matrix_fold_1.png')
        self.assertTrue(os.path.exists(filepath))
        self.assertGreater(os.path.getsize(filepath), 0)
        
        print("混淆矩阵绘制测试通过")
    
    def test_roc_curve_plot(self):
        """测试ROC曲线绘制"""
        print("测试ROC曲线绘制...")
        
        evaluator = ExertionEvaluator(self.result_dir, self.config)
        
        # 绘制多分类ROC曲线
        evaluator._plot_multiclass_roc(self.true_labels, self.pred_probs, fold_idx=2)
        
        # 检查文件
        filepath = os.path.join(evaluator.eval_dir, 'multiclass_roc_fold_2.png')
        self.assertTrue(os.path.exists(filepath))
        self.assertGreater(os.path.getsize(filepath), 0)
        
        print("ROC曲线绘制测试通过")
    
    def test_prediction_distribution_plot(self):
        """测试预测分布绘制"""
        print("测试预测分布绘制...")
        
        evaluator = ExertionEvaluator(self.result_dir, self.config)
        
        # 绘制预测分布
        evaluator._plot_prediction_distribution(self.true_labels, self.pred_labels, fold_idx=3)
        
        # 检查文件
        filepath = os.path.join(evaluator.eval_dir, 'prediction_distribution_fold_3.png')
        self.assertTrue(os.path.exists(filepath))
        self.assertGreater(os.path.getsize(filepath), 0)
        
        print("预测分布绘制测试通过")
    
    def test_class_accuracy_plot(self):
        """测试类别准确率绘制"""
        print("测试类别准确率绘制...")
        
        evaluator = ExertionEvaluator(self.result_dir, self.config)
        
        # 绘制类别准确率
        evaluator._plot_class_accuracy(self.true_labels, self.pred_labels, fold_idx=4)
        
        # 检查文件
        filepath = os.path.join(evaluator.eval_dir, 'class_accuracy_fold_4.png')
        self.assertTrue(os.path.exists(filepath))
        self.assertGreater(os.path.getsize(filepath), 0)
        
        print("类别准确率绘制测试通过")
    
    def test_auc_threshold_optimization(self):
        """测试AUC阈值优化"""
        print("测试AUC阈值优化...")
        
        evaluator = ExertionEvaluator(self.result_dir, self.config)
        
        # 优化AUC阈值
        threshold_metrics = evaluator.optimize_auc_threshold(
            self.true_labels, 
            self.pred_probs, 
            fold_idx=5
        )
        
        # 检查结果
        self.assertIsInstance(threshold_metrics, dict)
        self.assertIn('optimal_threshold', threshold_metrics)
        self.assertIn('optimal_auc', threshold_metrics)
        self.assertIn('thresholds', threshold_metrics)
        self.assertIn('aucs', threshold_metrics)
        
        # 检查阈值范围
        optimal_threshold = threshold_metrics['optimal_threshold']
        self.assertGreaterEqual(optimal_threshold, 0)
        self.assertLessEqual(optimal_threshold, 1)
        
        # 检查AUC值
        optimal_auc = threshold_metrics['optimal_auc']
        self.assertGreaterEqual(optimal_auc, 0)
        self.assertLessEqual(optimal_auc, 1)
        
        print(f"最优阈值: {optimal_threshold:.4f}")
        print(f"最优AUC: {optimal_auc:.4f}")
    
    def test_evaluation_results_saving(self):
        """测试评估结果保存"""
        print("测试评估结果保存...")
        
        evaluator = ExertionEvaluator(self.result_dir, self.config)
        
        # 计算指标
        metrics = evaluator._calculate_metrics(
            self.true_labels, 
            self.pred_labels, 
            self.pred_probs
        )
        
        # 保存结果
        result_path = os.path.join(evaluator.eval_dir, "test_evaluation.json")
        evaluator._save_evaluation_results(
            metrics, 
            self.true_labels, 
            self.pred_labels, 
            self.pred_probs, 
            result_path
        )
        
        # 检查文件
        self.assertTrue(os.path.exists(result_path))
        self.assertGreater(os.path.getsize(result_path), 0)
        
        # 读取并验证保存的数据
        import json
        with open(result_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        self.assertIn('metrics', saved_data)
        self.assertIn('predictions', saved_data)
        self.assertIn('probabilities', saved_data)
        
        print("评估结果保存测试通过")
    
    def test_cross_validation_evaluation(self):
        """测试交叉验证评估"""
        print("测试交叉验证评估...")
        
        evaluator = ExertionEvaluator(self.result_dir, self.config)
        
        # 创建模拟交叉验证结果
        cv_results = {
            'fold_0': {
                'metrics': {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.82, 'f1_score': 0.82},
                'predictions': self.pred_labels[:50],
                'probabilities': self.pred_probs[:50],
                'labels': self.true_labels[:50]
            },
            'fold_1': {
                'metrics': {'accuracy': 0.87, 'precision': 0.85, 'recall': 0.84, 'f1_score': 0.84},
                'predictions': self.pred_labels[50:],
                'probabilities': self.pred_probs[50:],
                'labels': self.true_labels[50:]
            }
        }
        
        # 评估交叉验证结果
        cv_summary = evaluator.evaluate_cross_validation(cv_results)
        
        # 检查结果
        self.assertIsInstance(cv_summary, dict)
        self.assertIn('mean_accuracy', cv_summary)
        self.assertIn('std_accuracy', cv_summary)
        self.assertIn('mean_precision', cv_summary)
        self.assertIn('std_precision', cv_summary)
        self.assertIn('mean_recall', cv_summary)
        self.assertIn('std_recall', cv_summary)
        self.assertIn('mean_f1_score', cv_summary)
        self.assertIn('std_f1_score', cv_summary)
        
        print(f"交叉验证平均准确率: {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")

class TestEvaluationIntegration(unittest.TestCase):
    """评估集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp_dir = tempfile.mkdtemp()
        self.result_dir = os.path.join(self.temp_dir, "result")
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.config = {
            'mfb_dim': 40,
            'mfcc_dim': 13,
            'wav2vec2_dim': 768,
            'num_classes': 5,
            'dropout_rate': 0.5,
            'use_mfcc': True,
            'use_wav2vec2': True,
            'optimize_for_rtx4070': False,
        }
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def test_full_evaluation_pipeline(self):
        """测试完整评估流程"""
        print("测试完整评估流程...")
        
        evaluator = ExertionEvaluator(self.result_dir, self.config)
        
        # 创建测试模型
        model = create_model(self.config).to(self.device)
        
        # 创建测试数据
        batch_size = 4
        seq_len = 100
        num_samples = 16
        
        mfcc_data = torch.randn(num_samples, self.config['mfcc_dim'], seq_len)
        wav2vec2_data = torch.randn(num_samples, self.config['wav2vec2_dim'], seq_len)
        labels = torch.randint(0, self.config['num_classes'], (num_samples,))
        
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(mfcc_data, wav2vec2_data, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 执行完整评估
        metrics, all_labels, all_preds, all_probs = evaluator.evaluate_model(
            model, dataloader, fold_idx=0
        )
        
        # 生成可视化
        evaluator._generate_visualizations(all_labels, all_preds, all_probs, fold_idx=0)
        
        # 优化阈值
        threshold_metrics = evaluator.optimize_auc_threshold(all_labels, all_probs, fold_idx=0)
        
        # 检查所有输出
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(threshold_metrics, dict)
        
        # 检查生成的文件
        expected_files = [
            'fold_0_evaluation.json',
            'confusion_matrix_fold_0.png',
            'multiclass_roc_fold_0.png',
            'prediction_distribution_fold_0.png',
            'class_accuracy_fold_0.png',
            'threshold_optimization_fold_0.png'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(evaluator.eval_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"文件不存在: {filename}")
        
        print("完整评估流程测试通过")
    
    def test_evaluation_with_different_configurations(self):
        """测试不同配置的评估"""
        print("测试不同配置的评估...")
        
        # 测试不同类别数
        for num_classes in [3, 5, 7]:
            config = self.config.copy()
            config['num_classes'] = num_classes
            
            evaluator = ExertionEvaluator(self.result_dir, config)
            
            # 创建测试数据
            true_labels = np.random.randint(0, num_classes, 50)
            pred_probs = np.random.rand(50, num_classes)
            pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)
            pred_labels = np.argmax(pred_probs, axis=1)
            
            # 计算指标
            metrics = evaluator._calculate_metrics(true_labels, pred_labels, pred_probs)
            
            self.assertIn('accuracy', metrics)
            self.assertEqual(metrics['confusion_matrix'].shape, (num_classes, num_classes))
            
            print(f"{num_classes}类评估测试通过")
    
    def test_evaluation_error_handling(self):
        """测试评估错误处理"""
        print("测试评估错误处理...")
        
        evaluator = ExertionEvaluator(self.result_dir, self.config)
        
        # 测试空数据
        with self.assertRaises(ValueError):
            evaluator._calculate_metrics([], [], [])
        
        # 测试不匹配的维度
        with self.assertRaises(ValueError):
            evaluator._calculate_metrics(
                [0, 1, 2], 
                [0, 1], 
                np.random.rand(3, 5)
            )
        
        # 测试无效的概率
        with self.assertRaises(ValueError):
            evaluator._calculate_metrics(
                [0, 1, 2], 
                [0, 1, 2], 
                np.random.rand(3, 5)  # 未归一化的概率
            )
        
        print("评估错误处理测试通过")

def run_evaluation_tests():
    """运行评估测试"""
    print("开始运行评估模块单元测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.makeSuite(TestExertionEvaluator))
    test_suite.addTest(unittest.makeSuite(TestEvaluationIntegration))
    
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
    success = run_evaluation_tests()
    if success:
        print("所有评估测试通过!")
    else:
        print("部分评估测试失败!")
