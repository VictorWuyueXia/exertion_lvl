#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据标准化模块
支持在train_test_split后进行标准化，并保存标准化元数据
"""

import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import torch


class FeatureNormalizer:
    """
    特征标准化器
    支持在数据分割后进行标准化，并保存标准化参数
    """
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.scalers = {}
        self.feature_names = []
        self.is_fitted = False
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
    
    def fit_transform(self, train_features: Dict[str, np.ndarray], 
                     val_features: Optional[Dict[str, np.ndarray]] = None,
                     test_features: Optional[Dict[str, np.ndarray]] = None) -> Tuple[Dict[str, np.ndarray], ...]:
        """
        在训练集上拟合标准化器，并转换所有数据集
        
        Args:
            train_features: 训练集特征字典
            val_features: 验证集特征字典（可选）
            test_features: 测试集特征字典（可选）
            
        Returns:
            标准化后的特征字典元组 (train_normalized, val_normalized, test_normalized)
        """
        print("开始特征标准化...")
        
        # 获取所有特征名称
        self.feature_names = list(train_features.keys())
        print(f"特征名称: {self.feature_names}")
        
        # 初始化标准化器
        normalized_train = {}
        normalized_val = {}
        normalized_test = {}
        
        for feature_name in self.feature_names:
            print(f"标准化特征: {feature_name}")
            
            # 创建标准化器
            scaler = StandardScaler()
            
            # 在训练集上拟合
            train_feature = train_features[feature_name]
            print(f"  训练集形状: {train_feature.shape}")
            
            # 重塑为2D进行标准化
            if len(train_feature.shape) == 3:  # (samples, time_steps, features)
                n_samples, time_steps, n_features = train_feature.shape
                train_feature_2d = train_feature.reshape(-1, n_features)
            else:
                train_feature_2d = train_feature
            
            # 拟合标准化器
            train_normalized_2d = scaler.fit_transform(train_feature_2d)
            
            # 恢复原始形状
            if len(train_feature.shape) == 3:
                train_normalized = train_normalized_2d.reshape(n_samples, time_steps, n_features)
            else:
                train_normalized = train_normalized_2d
            
            normalized_train[feature_name] = train_normalized
            
            # 保存标准化器
            self.scalers[feature_name] = scaler
            
            # 转换验证集
            if val_features is not None and feature_name in val_features:
                val_feature = val_features[feature_name]
                print(f"  验证集形状: {val_feature.shape}")
                
                if len(val_feature.shape) == 3:
                    n_samples, time_steps, n_features = val_feature.shape
                    val_feature_2d = val_feature.reshape(-1, n_features)
                else:
                    val_feature_2d = val_feature
                
                val_normalized_2d = scaler.transform(val_feature_2d)
                
                if len(val_feature.shape) == 3:
                    val_normalized = val_normalized_2d.reshape(n_samples, time_steps, n_features)
                else:
                    val_normalized = val_normalized_2d
                
                normalized_val[feature_name] = val_normalized
            
            # 转换测试集
            if test_features is not None and feature_name in test_features:
                test_feature = test_features[feature_name]
                print(f"  测试集形状: {test_feature.shape}")
                
                if len(test_feature.shape) == 3:
                    n_samples, time_steps, n_features = test_feature.shape
                    test_feature_2d = test_feature.reshape(-1, n_features)
                else:
                    test_feature_2d = test_feature
                
                test_normalized_2d = scaler.transform(test_feature_2d)
                
                if len(test_feature.shape) == 3:
                    test_normalized = test_normalized_2d.reshape(n_samples, time_steps, n_features)
                else:
                    test_normalized = test_normalized_2d
                
                normalized_test[feature_name] = test_normalized
        
        self.is_fitted = True
        
        # 保存标准化参数
        self.save_scalers()
        
        print("特征标准化完成")
        
        return normalized_train, normalized_val, normalized_test
    
    def transform(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        使用已拟合的标准化器转换特征
        
        Args:
            features: 特征字典
            
        Returns:
            标准化后的特征字典
        """
        if not self.is_fitted:
            raise ValueError("标准化器尚未拟合，请先调用fit_transform方法")
        
        normalized_features = {}
        
        for feature_name in self.feature_names:
            if feature_name not in features:
                print(f"警告: 特征 {feature_name} 不在输入特征中")
                continue
            
            scaler = self.scalers[feature_name]
            feature = features[feature_name]
            
            # 重塑为2D进行标准化
            if len(feature.shape) == 3:
                n_samples, time_steps, n_features = feature.shape
                feature_2d = feature.reshape(-1, n_features)
            else:
                feature_2d = feature
            
            # 转换
            normalized_2d = scaler.transform(feature_2d)
            
            # 恢复原始形状
            if len(feature.shape) == 3:
                normalized = normalized_2d.reshape(n_samples, time_steps, n_features)
            else:
                normalized = normalized_2d
            
            normalized_features[feature_name] = normalized
        
        return normalized_features
    
    def save_scalers(self):
        """保存标准化器参数"""
        scaler_params = {}
        
        for feature_name, scaler in self.scalers.items():
            scaler_params[feature_name] = {
                'mean_': scaler.mean_.tolist(),
                'scale_': scaler.scale_.tolist(),
                'var_': scaler.var_.tolist()
            }
        
        # 保存参数
        params_path = os.path.join(self.save_dir, 'scaler_params.json')
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(scaler_params, f, indent=2, ensure_ascii=False)
        
        print(f"标准化器参数已保存到: {params_path}")
    
    def load_scalers(self):
        """加载标准化器参数"""
        params_path = os.path.join(self.save_dir, 'scaler_params.json')
        
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"标准化器参数文件不存在: {params_path}")
        
        with open(params_path, 'r', encoding='utf-8') as f:
            scaler_params = json.load(f)
        
        for feature_name, params in scaler_params.items():
            scaler = StandardScaler()
            scaler.mean_ = np.array(params['mean_'])
            scaler.scale_ = np.array(params['scale_'])
            scaler.var_ = np.array(params['var_'])
            self.scalers[feature_name] = scaler
        
        self.feature_names = list(self.scalers.keys())
        self.is_fitted = True
        
        print(f"标准化器参数已从 {params_path} 加载")


def normalize_dataset_features(dataset, train_indices: List[int], val_indices: List[int], 
                             test_indices: Optional[List[int]] = None, 
                             save_dir: str = "data/normalization") -> Tuple[Dict, ...]:
    """
    对数据集特征进行标准化
    
    Args:
        dataset: 数据集对象
        train_indices: 训练集索引
        val_indices: 验证集索引
        test_indices: 测试集索引（可选）
        save_dir: 标准化参数保存目录
        
    Returns:
        标准化后的特征字典元组 (train_features, val_features, test_features)
    """
    print("开始数据集特征标准化...")
    
    # 创建标准化器
    normalizer = FeatureNormalizer(save_dir)
    
    # 提取训练集特征
    train_features = extract_features_from_indices(dataset, train_indices)
    print(f"训练集特征提取完成，样本数: {len(train_indices)}")
    
    # 提取验证集特征
    val_features = extract_features_from_indices(dataset, val_indices)
    print(f"验证集特征提取完成，样本数: {len(val_indices)}")
    
    # 提取测试集特征（如果有）
    test_features = None
    if test_indices is not None:
        test_features = extract_features_from_indices(dataset, test_indices)
        print(f"测试集特征提取完成，样本数: {len(test_indices)}")
    
    # 标准化
    normalized_train, normalized_val, normalized_test = normalizer.fit_transform(
        train_features, val_features, test_features
    )
    
    return normalized_train, normalized_val, normalized_test


def extract_features_from_indices(dataset, indices: List[int]) -> Dict[str, np.ndarray]:
    """
    从数据集中提取指定索引的特征
    
    Args:
        dataset: 数据集对象
        indices: 索引列表
        
    Returns:
        特征字典
    """
    features = {}
    
    # 收集所有特征
    all_mfcc = []
    all_wav2vec2 = []
    
    for idx in indices:
        sample = dataset[idx]
        
        if sample['acoustic'] is not None:
            all_mfcc.append(sample['acoustic'])
        
        if sample['embeds'] is not None:
            all_wav2vec2.append(sample['embeds'])
    
    # 转换为numpy数组
    if all_mfcc:
        features['mfcc'] = np.array(all_mfcc)
    
    if all_wav2vec2:
        features['wav2vec2'] = np.array(all_wav2vec2)
    
    return features
