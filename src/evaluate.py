#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化结果评估脚本
分析训练结果并生成基本报告
"""

import os
import json
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.simple_evaluator import SimpleExertionEvaluator

def load_cv_results(result_dir):
    """加载交叉验证结果"""
    cv_results_path = os.path.join(result_dir, "cv_results.json")
    
    if not os.path.exists(cv_results_path):
        print(f"错误: 找不到交叉验证结果文件: {cv_results_path}")
        return None
    
    with open(cv_results_path, 'r', encoding='utf-8') as f:
        cv_results = json.load(f)
    
    return cv_results

def main():
    """主函数"""
    print("简化结果评估开始")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 查找最新的结果目录
    result_base_dir = "result"
    if not os.path.exists(result_base_dir):
        print("错误: 找不到result目录")
        return
    
    result_dirs = [d for d in os.listdir(result_base_dir) 
                  if os.path.isdir(os.path.join(result_base_dir, d)) and d.startswith("VGG16")]
    
    if not result_dirs:
        print("错误: 找不到VGG16训练结果目录")
        return
    
    # 选择最新的结果目录
    latest_result_dir = sorted(result_dirs)[-1]
    result_dir = os.path.join(result_base_dir, latest_result_dir)
    
    print(f"使用结果目录: {result_dir}")
    
    # 加载交叉验证结果
    cv_results = load_cv_results(result_dir)
    if cv_results is None:
        return
    
    print(f"找到 {len(cv_results)} 个fold的结果")
    
    # 创建简化评估器
    evaluator = SimpleExertionEvaluator(result_dir)
    
    # 运行评估
    try:
        overall_results = evaluator.evaluate_cross_validation(cv_results)
        
        print("\n" + "="*50)
        print("评估完成！")
        print("="*50)
        print(f"结果保存目录: {result_dir}")
        
        # 打印主要指标
        metrics = overall_results['metrics']
        print(f"\n主要指标:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  F1宏平均: {metrics['f1_macro']:.4f}")
        print(f"  F1加权平均: {metrics['f1_weighted']:.4f}")
        print(f"  精确率宏平均: {metrics['precision_macro']:.4f}")
        print(f"  召回率宏平均: {metrics['recall_macro']:.4f}")
        
        # 打印每个类别的F1分数
        print(f"\n各类别F1分数:")
        for i, f1_score in enumerate(metrics['f1_per_class']):
            print(f"  类别 {i}: {f1_score:.4f}")
        
        print("\n评估完成！")
        
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
