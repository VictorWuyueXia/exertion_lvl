#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主测试运行器
运行所有模块的单元测试
"""

import sys
import os
import time
import unittest
import subprocess
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_test_module(module_name, test_file):
    """运行单个测试模块"""
    print(f"\n{'='*60}")
    print(f"开始运行 {module_name} 测试")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 运行测试文件
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=True, text=True, timeout=300)  # 5分钟超时
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {module_name} 测试通过 (耗时: {duration:.2f}秒)")
            return True, duration
        else:
            print(f"❌ {module_name} 测试失败 (耗时: {duration:.2f}秒)")
            print(f"错误输出:\n{result.stderr}")
            return False, duration
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {module_name} 测试超时")
        return False, 300
    except Exception as e:
        print(f"💥 {module_name} 测试异常: {e}")
        return False, 0

def run_all_tests():
    """运行所有测试"""
    print("开始运行运动强度检测项目单元测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    
    # 定义测试模块
    test_modules = [
        ("数据处理模块", "test_data_processing.py"),
        ("模型模块", "test_model.py"),
        ("训练模块", "test_training.py"),
        ("评估模块", "test_evaluation.py"),
        ("工具模块", "test_utils.py"),
        ("特征加载", "test_feature_loading.py"),
        ("进度显示", "test_progress_display.py"),
        ("WandB管理", "test_wandb.py"),
        ("RTX4070优化", "test_rtx4070_optimization.py")
    ]
    
    # 运行测试
    results = []
    total_duration = 0
    
    for module_name, test_file in test_modules:
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        
        if os.path.exists(test_path):
            success, duration = run_test_module(module_name, test_path)
            results.append((module_name, success, duration))
            total_duration += duration
        else:
            print(f"⚠️  测试文件不存在: {test_file}")
            results.append((module_name, False, 0))
    
    # 输出测试总结
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    print(f"总测试模块数: {len(results)}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"总耗时: {total_duration:.2f}秒")
    
    if failed > 0:
        print(f"\n失败的测试模块:")
        for module_name, success, duration in results:
            if not success:
                print(f"  - {module_name}")
    
    print(f"\n测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return failed == 0

def run_specific_test(test_name):
    """运行特定测试"""
    test_file_map = {
        "data": "test_data_processing.py",
        "model": "test_model.py",
        "training": "test_training.py",
        "evaluation": "test_evaluation.py",
        "utils": "test_utils.py",
        "feature": "test_feature_loading.py",
        "progress": "test_progress_display.py",
        "wandb": "test_wandb.py",
        "rtx4070": "test_rtx4070_optimization.py"
    }
    
    if test_name in test_file_map:
        test_file = test_file_map[test_name]
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        
        if os.path.exists(test_path):
            module_names = {
                "data": "数据处理模块",
                "model": "模型模块",
                "training": "训练模块",
                "evaluation": "评估模块",
                "utils": "工具模块",
                "feature": "特征加载",
                "progress": "进度显示",
                "wandb": "WandB管理",
                "rtx4070": "RTX4070优化"
            }
            
            success, duration = run_test_module(module_names[test_name], test_path)
            return success
        else:
            print(f"测试文件不存在: {test_file}")
            return False
    else:
        print(f"未知的测试模块: {test_name}")
        print(f"可用的测试模块: {list(test_file_map.keys())}")
        return False

def run_quick_tests():
    """运行快速测试（只运行核心模块）"""
    print("运行快速测试（核心模块）")
    
    core_tests = [
        ("数据处理模块", "test_data_processing.py"),
        ("模型模块", "test_model.py"),
        ("工具模块", "test_utils.py")
    ]
    
    results = []
    total_duration = 0
    
    for module_name, test_file in core_tests:
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        
        if os.path.exists(test_path):
            success, duration = run_test_module(module_name, test_path)
            results.append((module_name, success, duration))
            total_duration += duration
        else:
            print(f"⚠️  测试文件不存在: {test_file}")
            results.append((module_name, False, 0))
    
    # 输出总结
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    print(f"\n快速测试总结:")
    print(f"通过: {passed}/{len(results)}")
    print(f"总耗时: {total_duration:.2f}秒")
    
    return failed == 0

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="运行运动强度检测项目单元测试")
    parser.add_argument("--module", "-m", type=str, help="运行特定测试模块")
    parser.add_argument("--quick", "-q", action="store_true", help="运行快速测试")
    parser.add_argument("--list", "-l", action="store_true", help="列出所有测试模块")
    
    args = parser.parse_args()
    
    if args.list:
        print("可用的测试模块:")
        print("  data      - 数据处理模块")
        print("  model     - 模型模块")
        print("  training  - 训练模块")
        print("  evaluation- 评估模块")
        print("  utils     - 工具模块")
        print("  feature   - 特征加载")
        print("  progress  - 进度显示")
        print("  wandb     - WandB管理")
        print("  rtx4070   - RTX4070优化")
        return
    
    if args.module:
        success = run_specific_test(args.module)
        sys.exit(0 if success else 1)
    
    if args.quick:
        success = run_quick_tests()
        sys.exit(0 if success else 1)
    
    # 默认运行所有测试
    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
