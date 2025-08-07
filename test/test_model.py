#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型模块单元测试
测试VGG16模型结构、前向传播、参数计算等功能
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vgg16_exertion import (
    VGG16ExertionModel,
    VGG16ExertionModelRTX4070,
    create_model,
    count_parameters
)

class TestVGG16ExertionModel(unittest.TestCase):
    """VGG16运动强度检测模型测试类"""
    
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
            'optimize_for_rtx4070': False
        }
    
    def test_model_creation(self):
        """测试模型创建"""
        print("测试模型创建...")
        
        # 测试基本模型创建
        model = VGG16ExertionModel(
            mfcc_dim=self.config['mfcc_dim'],
            wav2vec2_dim=self.config['wav2vec2_dim'],
            num_classes=self.config['num_classes'],
            dropout_rate=self.config['dropout_rate'],
            use_mfcc=self.config['use_mfcc'],
            use_wav2vec2=self.config['use_wav2vec2']
        )
        
        self.assertIsInstance(model, VGG16ExertionModel)
        self.assertIsInstance(model, nn.Module)
        
        # 检查模型组件
        self.assertIsInstance(model.features, nn.Sequential)
        self.assertIsInstance(model.classifier, nn.Sequential)
        
        print(f"模型创建成功，输入维度: {model.input_dim}")
    
    def test_model_parameters(self):
        """测试模型参数"""
        print("测试模型参数...")
        
        model = VGG16ExertionModel(
            mfcc_dim=self.config['mfcc_dim'],
            wav2vec2_dim=self.config['wav2vec2_dim'],
            num_classes=self.config['num_classes']
        )
        
        # 计算参数数量
        total_params = count_parameters(model)
        self.assertGreater(total_params, 0)
        
        # 检查可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(total_params, trainable_params)
        
        print(f"模型总参数数: {total_params:,}")
        print(f"可训练参数数: {trainable_params:,}")
    
    def test_model_forward_pass(self):
        """测试模型前向传播"""
        print("测试模型前向传播...")
        
        model = VGG16ExertionModel(
            mfcc_dim=self.config['mfcc_dim'],
            wav2vec2_dim=self.config['wav2vec2_dim'],
            num_classes=self.config['num_classes']
        ).to(self.device)
        
        # 创建测试输入
        batch_size = 4
        seq_len = 100
        
        # 测试MFCC输入
        mfcc_input = torch.randn(batch_size, self.config['mfcc_dim'], seq_len).to(self.device)
        
        # 测试wav2vec2输入
        wav2vec2_input = torch.randn(batch_size, self.config['wav2vec2_dim'], seq_len).to(self.device)
        
        # 测试前向传播
        with torch.no_grad():
            # 测试MFCC输入
            output_mfcc = model(mfcc=mfcc_input)
            self.assertEqual(output_mfcc.shape, (batch_size, self.config['num_classes']))
            
            # 测试wav2vec2输入
            output_wav2vec2 = model(wav2vec2=wav2vec2_input)
            self.assertEqual(output_wav2vec2.shape, (batch_size, self.config['num_classes']))
            
            # 测试组合输入
            output_combined = model(mfcc=mfcc_input, wav2vec2=wav2vec2_input)
            self.assertEqual(output_combined.shape, (batch_size, self.config['num_classes']))
        
        print(f"前向传播测试通过，输出形状: {output_combined.shape}")
    
    def test_model_feature_extraction(self):
        """测试模型特征提取"""
        print("测试模型特征提取...")
        
        model = VGG16ExertionModel(
            mfcc_dim=self.config['mfcc_dim'],
            wav2vec2_dim=self.config['wav2vec2_dim'],
            num_classes=self.config['num_classes']
        ).to(self.device)
        
        # 创建测试输入
        batch_size = 2
        seq_len = 200
        mfcc_input = torch.randn(batch_size, self.config['mfcc_dim'], seq_len).to(self.device)
        
        # 测试特征提取
        with torch.no_grad():
            # 获取特征
            features = model.features(mfcc_input)
            self.assertIsInstance(features, torch.Tensor)
            
            # 检查特征维度
            expected_feature_dim = 512
            self.assertEqual(features.shape[1], expected_feature_dim)
            
            # 检查特征长度（经过池化层后）
            expected_seq_len = seq_len // 32  # 5次池化，每次除以2
            self.assertEqual(features.shape[2], expected_seq_len)
        
        print(f"特征提取测试通过，特征形状: {features.shape}")
    
    def test_model_different_configurations(self):
        """测试不同配置的模型"""
        print("测试不同配置的模型...")
        
        # 测试只使用MFCC
        model_mfcc_only = VGG16ExertionModel(
            mfcc_dim=self.config['mfcc_dim'],
            wav2vec2_dim=self.config['wav2vec2_dim'],
            num_classes=self.config['num_classes'],
            use_mfcc=True,
            use_wav2vec2=False
        ).to(self.device)
        
        # 测试只使用wav2vec2
        model_wav2vec2_only = VGG16ExertionModel(
            mfcc_dim=self.config['mfcc_dim'],
            wav2vec2_dim=self.config['wav2vec2_dim'],
            num_classes=self.config['num_classes'],
            use_mfcc=False,
            use_wav2vec2=True
        ).to(self.device)
        
        # 创建测试输入
        batch_size = 2
        seq_len = 100
        mfcc_input = torch.randn(batch_size, self.config['mfcc_dim'], seq_len).to(self.device)
        wav2vec2_input = torch.randn(batch_size, self.config['wav2vec2_dim'], seq_len).to(self.device)
        
        with torch.no_grad():
            # 测试MFCC-only模型
            output_mfcc_only = model_mfcc_only(mfcc=mfcc_input)
            self.assertEqual(output_mfcc_only.shape, (batch_size, self.config['num_classes']))
            
            # 测试wav2vec2-only模型
            output_wav2vec2_only = model_wav2vec2_only(wav2vec2=wav2vec2_input)
            self.assertEqual(output_wav2vec2_only.shape, (batch_size, self.config['num_classes']))
        
        print("不同配置模型测试通过")
    
    def test_rtx4070_optimized_model(self):
        """测试RTX4070优化模型"""
        print("测试RTX4070优化模型...")
        
        model = VGG16ExertionModelRTX4070(
            mfcc_dim=self.config['mfcc_dim'],
            wav2vec2_dim=self.config['wav2vec2_dim'],
            num_classes=self.config['num_classes']
        ).to(self.device)
        
        self.assertIsInstance(model, VGG16ExertionModelRTX4070)
        self.assertIsInstance(model, VGG16ExertionModel)
        
        # 创建测试输入
        batch_size = 4
        seq_len = 100
        mfcc_input = torch.randn(batch_size, self.config['mfcc_dim'], seq_len).to(self.device)
        wav2vec2_input = torch.randn(batch_size, self.config['wav2vec2_dim'], seq_len).to(self.device)
        
        # 测试前向传播
        with torch.no_grad():
            output = model(mfcc=mfcc_input, wav2vec2=wav2vec2_input)
            self.assertEqual(output.shape, (batch_size, self.config['num_classes']))
        
        print("RTX4070优化模型测试通过")
    
    def test_model_gradient_flow(self):
        """测试模型梯度流动"""
        print("测试模型梯度流动...")
        
        model = VGG16ExertionModel(
            mfcc_dim=self.config['mfcc_dim'],
            wav2vec2_dim=self.config['wav2vec2_dim'],
            num_classes=self.config['num_classes']
        ).to(self.device)
        
        # 创建测试输入
        batch_size = 2
        seq_len = 100
        mfcc_input = torch.randn(batch_size, self.config['mfcc_dim'], seq_len).to(self.device)
        wav2vec2_input = torch.randn(batch_size, self.config['wav2vec2_dim'], seq_len).to(self.device)
        
        # 前向传播
        output = model(mfcc=mfcc_input, wav2vec2=wav2vec2_input)
        
        # 计算损失
        target = torch.randint(0, self.config['num_classes'], (batch_size,)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                grad_norm = param.grad.norm().item()
                self.assertGreater(grad_norm, 0)
                print(f"参数 {name} 梯度范数: {grad_norm:.6f}")
        
        self.assertTrue(has_gradients)
        print("梯度流动测试通过")
    
    def test_model_initialization(self):
        """测试模型初始化"""
        print("测试模型初始化...")
        
        model = VGG16ExertionModel(
            mfcc_dim=self.config['mfcc_dim'],
            wav2vec2_dim=self.config['wav2vec2_dim'],
            num_classes=self.config['num_classes']
        )
        
        # 检查权重初始化
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv1d):
                weight = module.weight.data
                # 检查权重是否在合理范围内
                self.assertGreater(weight.std().item(), 0)
                self.assertLess(weight.std().item(), 1.0)
        
        print("模型初始化测试通过")
    
    def test_create_model_function(self):
        """测试create_model函数"""
        print("测试create_model函数...")
        
        # 测试创建基本模型
        model = create_model(self.config)
        self.assertIsInstance(model, VGG16ExertionModel)
        
        # 测试创建RTX4070优化模型
        config_rtx4070 = self.config.copy()
        config_rtx4070['optimize_for_rtx4070'] = True
        model_rtx4070 = create_model(config_rtx4070)
        self.assertIsInstance(model_rtx4070, VGG16ExertionModelRTX4070)
        
        print("create_model函数测试通过")
    
    def test_model_memory_usage(self):
        """测试模型内存使用"""
        print("测试模型内存使用...")
        
        if torch.cuda.is_available():
            # 记录初始内存
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # 创建模型
            model = VGG16ExertionModel(
                mfcc_dim=self.config['mfcc_dim'],
                wav2vec2_dim=self.config['wav2vec2_dim'],
                num_classes=self.config['num_classes']
            ).to(self.device)
            
            # 记录模型内存
            model_memory = torch.cuda.memory_allocated() - initial_memory
            
            # 创建测试输入
            batch_size = 4
            seq_len = 100
            mfcc_input = torch.randn(batch_size, self.config['mfcc_dim'], seq_len).to(self.device)
            wav2vec2_input = torch.randn(batch_size, self.config['wav2vec2_dim'], seq_len).to(self.device)
            
            # 前向传播
            with torch.no_grad():
                output = model(mfcc=mfcc_input, wav2vec2=wav2vec2_input)
            
            # 记录总内存
            total_memory = torch.cuda.memory_allocated() - initial_memory
            
            print(f"模型参数内存: {model_memory / 1024**2:.2f} MB")
            print(f"总内存使用: {total_memory / 1024**2:.2f} MB")
            
            # 清理内存
            del model, mfcc_input, wav2vec2_input, output
            torch.cuda.empty_cache()
        else:
            print("CUDA不可用，跳过内存测试")

class TestModelIntegration(unittest.TestCase):
    """模型集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = {
            'mfb_dim': 40,
            'mfcc_dim': 13,
            'wav2vec2_dim': 768,
            'num_classes': 5,
            'dropout_rate': 0.5,
            'use_mfcc': True,
            'use_wav2vec2': True,
            'optimize_for_rtx4070': False
        }
    
    def test_model_training_cycle(self):
        """测试模型训练周期"""
        print("测试模型训练周期...")
        
        # 创建模型
        model = create_model(self.config).to(self.device)
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 创建损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 创建测试数据
        batch_size = 4
        seq_len = 100
        mfcc_input = torch.randn(batch_size, self.config['mfcc_dim'], seq_len).to(self.device)
        wav2vec2_input = torch.randn(batch_size, self.config['wav2vec2_dim'], seq_len).to(self.device)
        target = torch.randint(0, self.config['num_classes'], (batch_size,)).to(self.device)
        
        # 训练一个epoch
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            
            # 前向传播
            output = model(mfcc=mfcc_input, wav2vec2=wav2vec2_input)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        # 验证模型
        model.eval()
        with torch.no_grad():
            output = model(mfcc=mfcc_input, wav2vec2=wav2vec2_input)
            predictions = torch.argmax(output, dim=1)
            accuracy = (predictions == target).float().mean().item()
            
            print(f"验证准确率: {accuracy:.4f}")
        
        self.assertGreater(accuracy, 0.0)
        print("模型训练周期测试通过")
    
    def test_model_save_load(self):
        """测试模型保存和加载"""
        print("测试模型保存和加载...")
        
        import tempfile
        import os
        
        # 创建模型
        model = create_model(self.config).to(self.device)
        
        # 创建测试输入
        batch_size = 2
        seq_len = 100
        mfcc_input = torch.randn(batch_size, self.config['mfcc_dim'], seq_len).to(self.device)
        wav2vec2_input = torch.randn(batch_size, self.config['wav2vec2_dim'], seq_len).to(self.device)
        
        # 获取原始输出
        model.eval()
        with torch.no_grad():
            original_output = model(mfcc=mfcc_input, wav2vec2=wav2vec2_input)
        
        # 保存模型
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model_path = f.name
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config
        }, model_path)
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        loaded_model = create_model(checkpoint['config']).to(self.device)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 获取加载后的输出
        loaded_model.eval()
        with torch.no_grad():
            loaded_output = loaded_model(mfcc=mfcc_input, wav2vec2=wav2vec2_input)
        
        # 比较输出
        output_diff = torch.abs(original_output - loaded_output).max().item()
        self.assertLess(output_diff, 1e-6)
        
        # 清理临时文件
        os.unlink(model_path)
        
        print("模型保存和加载测试通过")

def run_model_tests():
    """运行模型测试"""
    print("开始运行模型模块单元测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.makeSuite(TestVGG16ExertionModel))
    test_suite.addTest(unittest.makeSuite(TestModelIntegration))
    
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
    success = run_model_tests()
    if success:
        print("所有模型测试通过!")
    else:
        print("部分模型测试失败!")
