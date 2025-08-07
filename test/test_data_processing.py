#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理模块单元测试
测试数据清理、音频分割、标签提取和特征提取功能
"""

import sys
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
import torchaudio
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.clean_data import clean_audio_data, validate_audio_files
from src.data.segment_audio import segment_audio_file, create_segments
from src.data.extract_labels import extract_exertion_labels, validate_labels
from src.data.get_features import (
    OptimizedAcousticExtractor, 
    extract_wav2vec2_features_batch,
    load_and_resample_audio_optimized
)

class TestDataProcessing(unittest.TestCase):
    """数据处理模块测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        
        # 创建测试音频文件
        self._create_test_audio()
        
        # 创建测试标签数据
        self.test_labels = pd.DataFrame({
            'participant_id': ['P001', 'P002', 'P003'],
            'session_id': ['session_1', 'session_2', 'session_3'],
            'exertion_level': [1, 2, 3],
            'speed': ['slow', 'medium', 'fast'],
            'task': ['walking', 'running', 'jumping']
        })
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_audio(self, duration=10.0, sample_rate=16000):
        """创建测试音频文件"""
        # 生成测试音频信号
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        
        # 保存为WAV文件
        torchaudio.save(
            self.test_audio_path,
            torch.tensor(signal).unsqueeze(0),
            sample_rate
        )
    
    def test_clean_audio_data(self):
        """测试音频数据清理功能"""
        print("测试音频数据清理功能...")
        
        # 创建测试音频目录
        audio_dir = os.path.join(self.temp_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        # 复制测试音频文件
        test_file = os.path.join(audio_dir, "test.wav")
        shutil.copy(self.test_audio_path, test_file)
        
        # 测试数据清理
        result = clean_audio_data(audio_dir, target_sr=16000)
        
        self.assertIsInstance(result, dict)
        self.assertIn('cleaned_files', result)
        self.assertIn('removed_files', result)
        self.assertIn('errors', result)
        
        print(f"清理结果: {result}")
    
    def test_validate_audio_files(self):
        """测试音频文件验证功能"""
        print("测试音频文件验证功能...")
        
        # 测试有效音频文件
        is_valid = validate_audio_files([self.test_audio_path])
        self.assertTrue(is_valid)
        
        # 测试无效文件
        invalid_file = os.path.join(self.temp_dir, "invalid.txt")
        with open(invalid_file, 'w') as f:
            f.write("not an audio file")
        
        is_valid = validate_audio_files([invalid_file])
        self.assertFalse(is_valid)
        
        print("音频文件验证测试通过")
    
    def test_segment_audio_file(self):
        """测试音频分割功能"""
        print("测试音频分割功能...")
        
        # 测试音频分割
        segments = segment_audio_file(
            self.test_audio_path,
            segment_duration=5.0,
            overlap=0.5
        )
        
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)
        
        # 检查每个分割段的长度
        for i, segment in enumerate(segments):
            self.assertIsInstance(segment, dict)
            self.assertIn('audio', segment)
            self.assertIn('start_time', segment)
            self.assertIn('end_time', segment)
            
            # 检查音频数据
            audio_data = segment['audio']
            self.assertIsInstance(audio_data, torch.Tensor)
            self.assertEqual(audio_data.dim(), 1)  # 单声道
            
            print(f"分割段 {i}: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s")
    
    def test_create_segments(self):
        """测试批量音频分割功能"""
        print("测试批量音频分割功能...")
        
        # 创建多个测试音频文件
        audio_files = []
        for i in range(3):
            audio_path = os.path.join(self.temp_dir, f"test_{i}.wav")
            shutil.copy(self.test_audio_path, audio_path)
            audio_files.append(audio_path)
        
        # 测试批量分割
        segments = create_segments(
            audio_files,
            segment_duration=3.0,
            overlap=0.0,
            output_dir=os.path.join(self.temp_dir, "segments")
        )
        
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)
        
        print(f"创建了 {len(segments)} 个分割段")
    
    def test_extract_exertion_labels(self):
        """测试运动强度标签提取功能"""
        print("测试运动强度标签提取功能...")
        
        # 创建测试标签文件
        label_file = os.path.join(self.temp_dir, "labels.csv")
        self.test_labels.to_csv(label_file, index=False)
        
        # 测试标签提取
        labels = extract_exertion_labels(label_file)
        
        self.assertIsInstance(labels, pd.DataFrame)
        self.assertEqual(len(labels), len(self.test_labels))
        self.assertIn('exertion_level', labels.columns)
        
        print(f"提取了 {len(labels)} 个标签记录")
    
    def test_validate_labels(self):
        """测试标签验证功能"""
        print("测试标签验证功能...")
        
        # 测试有效标签
        is_valid = validate_labels(self.test_labels)
        self.assertTrue(is_valid)
        
        # 测试无效标签（缺少必需列）
        invalid_labels = self.test_labels.drop('exertion_level', axis=1)
        is_valid = validate_labels(invalid_labels)
        self.assertFalse(is_valid)
        
        print("标签验证测试通过")
    
    def test_optimized_acoustic_extractor(self):
        """测试优化声学特征提取器"""
        print("测试优化声学特征提取器...")
        
        # 创建特征提取器
        extractor = OptimizedAcousticExtractor(device='cpu')
        
        # 加载测试音频
        audio, sr = torchaudio.load(self.test_audio_path)
        audio = audio.squeeze(0)  # 转换为单声道
        
        # 测试特征提取
        features = extractor.extract_all_features_batch(
            [audio.numpy()], 
            sr=sr, 
            target_frames=100
        )
        
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 1)
        
        feature_dict = features[0]
        self.assertIsInstance(feature_dict, dict)
        
        # 检查特征维度
        if 'mfcc' in feature_dict:
            mfcc = feature_dict['mfcc']
            self.assertIsInstance(mfcc, np.ndarray)
            self.assertEqual(mfcc.shape[0], 100)  # 时间维度
            print(f"MFCC特征维度: {mfcc.shape}")
        
        if 'mfb' in feature_dict:
            mfb = feature_dict['mfb']
            self.assertIsInstance(mfb, np.ndarray)
            self.assertEqual(mfb.shape[0], 100)  # 时间维度
            print(f"MFB特征维度: {mfb.shape}")
    
    @patch('src.data.get_features.Wav2Vec2Processor')
    @patch('src.data.get_features.Wav2Vec2Model')
    def test_wav2vec2_feature_extraction(self, mock_model, mock_processor):
        """测试wav2vec2特征提取功能"""
        print("测试wav2vec2特征提取功能...")
        
        # 模拟wav2vec2模型
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # 模拟模型输出
        mock_output = MagicMock()
        mock_output.hidden_states = [
            torch.randn(1, 100, 768),  # 第0层
            torch.randn(1, 100, 768),  # 第1层
            torch.randn(1, 100, 768),  # 第2层
            torch.randn(1, 100, 768),  # 第3层
            torch.randn(1, 100, 768),  # 第4层
        ]
        mock_model_instance.return_value = mock_output
        
        # 测试特征提取
        audio_list = [self.test_audio_path]
        sr_list = [16000]
        
        features = extract_wav2vec2_features_batch(
            audio_list,
            sr_list,
            selected_layers=(4,),
            batch_size=1
        )
        
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 1)
        
        print("wav2vec2特征提取测试通过")
    
    def test_audio_resampling(self):
        """测试音频重采样功能"""
        print("测试音频重采样功能...")
        
        # 测试重采样
        resampled_audio, new_sr = load_and_resample_audio_optimized(
            self.test_audio_path,
            sr_target=8000
        )
        
        self.assertIsInstance(resampled_audio, torch.Tensor)
        self.assertEqual(new_sr, 8000)
        self.assertEqual(resampled_audio.dim(), 1)  # 单声道
        
        print(f"重采样成功: {resampled_audio.shape}, 采样率: {new_sr}Hz")

class TestDataProcessingIntegration(unittest.TestCase):
    """数据处理集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self._create_test_data()
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """创建测试数据"""
        # 创建音频目录
        audio_dir = os.path.join(self.temp_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        # 创建测试音频文件
        for i in range(3):
            audio_path = os.path.join(audio_dir, f"test_{i}.wav")
            t = np.linspace(0, 10, 160000)
            signal = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
            torchaudio.save(audio_path, torch.tensor(signal).unsqueeze(0), 16000)
        
        # 创建标签文件
        labels_df = pd.DataFrame({
            'participant_id': ['P001', 'P002', 'P003'],
            'session_id': ['test_0', 'test_1', 'test_2'],
            'exertion_level': [1, 2, 3],
            'speed': ['slow', 'medium', 'fast'],
            'task': ['walking', 'running', 'jumping']
        })
        labels_df.to_csv(os.path.join(self.temp_dir, "labels.csv"), index=False)
    
    def test_full_data_processing_pipeline(self):
        """测试完整数据处理流程"""
        print("测试完整数据处理流程...")
        
        audio_dir = os.path.join(self.temp_dir, "audio")
        label_file = os.path.join(self.temp_dir, "labels.csv")
        output_dir = os.path.join(self.temp_dir, "output")
        
        # 1. 清理音频数据
        print("步骤1: 清理音频数据")
        clean_result = clean_audio_data(audio_dir, target_sr=16000)
        self.assertIsInstance(clean_result, dict)
        
        # 2. 提取标签
        print("步骤2: 提取标签")
        labels = extract_exertion_labels(label_file)
        self.assertIsInstance(labels, pd.DataFrame)
        
        # 3. 分割音频
        print("步骤3: 分割音频")
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        audio_paths = [os.path.join(audio_dir, f) for f in audio_files]
        
        segments = create_segments(
            audio_paths,
            segment_duration=5.0,
            overlap=0.5,
            output_dir=os.path.join(output_dir, "segments")
        )
        self.assertIsInstance(segments, list)
        
        print(f"数据处理流程测试完成: 清理了{len(clean_result['cleaned_files'])}个文件, "
              f"提取了{len(labels)}个标签, 创建了{len(segments)}个分割段")

def run_data_processing_tests():
    """运行数据处理测试"""
    print("开始运行数据处理模块单元测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.makeSuite(TestDataProcessing))
    test_suite.addTest(unittest.makeSuite(TestDataProcessingIntegration))
    
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
    success = run_data_processing_tests()
    if success:
        print("所有数据处理测试通过!")
    else:
        print("部分数据处理测试失败!")
