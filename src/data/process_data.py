#!/usr/bin/env python3
"""
主数据处理脚本 - 按顺序执行所有数据处理步骤
针对RTX 4070优化的运动强度语音数据集处理
"""

import os
import sys
import subprocess
import time

def run_script(script_name, description):
    """运行指定的Python脚本"""
    print(f"\n{'='*60}")
    print(f"步骤: {description}")
    print(f"脚本: {script_name}")
    print(f"{'='*60}")
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"错误: 脚本 {script_path} 不存在")
        return False
    
    try:
        # 运行脚本
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"✅ {description} 完成")
            if result.stdout:
                print("输出:", result.stdout)
        else:
            print(f"❌ {description} 失败")
            print("错误:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 运行 {script_name} 时出错: {e}")
        return False
    
    return True

def check_dependencies():
    """检查必要的依赖包"""
    print("检查依赖包...")
    
    required_packages = [
        'torch', 'torchaudio', 'numpy', 'pandas', 
        'soundfile', 'librosa', 'transformers', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - 未安装")
    
    if missing_packages:
        print(f"\n请安装缺失的包: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def create_directories():
    """创建必要的目录"""
    print("\n创建必要的目录...")
    
    directories = [
        "data/cleaned_audio",
        "data/cleaned_respiration_belt", 
        "data/segmented_audio",
        "data/features"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 创建目录: {directory}")

def main():
    """主函数 - 执行所有数据处理步骤"""
    print("🚀 开始数据处理流程")
    print("项目: 运动强度语音数据集处理")
    print("优化: RTX 4070 GPU加速")
    
    # 检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败，请安装缺失的包")
        return
    
    # 创建目录
    create_directories()
    
    # 步骤1: 数据清洗
    if not run_script("clean_data.py", "数据清洗"):
        print("❌ 数据清洗失败，停止处理")
        return
    
    # 步骤2: 数据分割
    if not run_script("split_data.py", "数据分割"):
        print("❌ 数据分割失败，停止处理")
        return
    
    # 步骤3: 音频分段
    if not run_script("segment_audio.py", "音频分段"):
        print("❌ 音频分段失败，停止处理")
        return
    
    # 步骤4: 提取exertion level标签
    if not run_script("extract_labels.py", "提取Exertion Level标签"):
        print("❌ 标签提取失败，停止处理")
        return
    
    # 步骤5: 特征提取
    if not run_script("get_features.py", "特征提取"):
        print("❌ 特征提取失败，停止处理")
        return
    
    print(f"\n{'='*60}")
    print("🎉 所有数据处理步骤完成！")
    print("📁 处理后的数据保存在以下目录:")
    print("   - data/cleaned_audio/: 清洗后的音频文件")
    print("   - data/segmented_audio/: 分段音频文件")
    print("   - data/features/: 提取的特征文件")
    print("   - data/exertion_labels.csv: 原始exertion level标签")
    print("   - data/segment_exertion_labels.csv: 分段exertion level标签")
    print("   - data/metadata_cleaned.csv: 清洗后的元数据")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 