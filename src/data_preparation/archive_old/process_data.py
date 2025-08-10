#!/usr/bin/env python3
"""
Main data processing script - Execute all data processing steps in sequence
Optimized for RTX 4070 exertion level voice dataset processing
"""

import os
import sys
import subprocess
import time
from tqdm import tqdm

def run_script(script_name, description):
    """Run specified Python script"""
    print(f"Step: {description}")
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"ERROR: Script {script_path} does not exist")
        return False
    
    try:
        # Run script with proper output handling for progress bars
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False,  # Don't capture output to show progress bars
                              text=True, 
                              cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"SUCCESS: {description} completed")
        else:
            print(f"ERROR: {description} failed")
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to run {script_name}: {e}")
        return False
    
    return True

def check_dependencies():
    """Check required dependencies"""
    print("Checking dependencies...")
    
    required_packages = [
        'torch', 'torchaudio', 'numpy', 'pandas', 
        'soundfile', 'librosa', 'transformers', 'tqdm', 'pyyaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Please install missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/cleaned_audio",
        "data/cleaned_respiration_belt", 
        "data/segmented_audio",
        "data/features"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main function - Execute all data processing steps"""
    print("Starting data processing pipeline")
    
    # Define processing steps
    steps = [
        ("check_dependencies", "Dependency Check"),
        ("create_directories", "Directory Creation"),
        ("clean_data.py", "Data Cleaning"),
        ("split_data.py", "Data Splitting"),
        ("segment_audio.py", "Audio Segmentation"),
        ("extract_labels.py", "Label Extraction"),
        ("get_features.py", "Feature Extraction")
    ]
    
    # Check dependencies first
    if not check_dependencies():
        print("ERROR: Dependency check failed, please install missing packages")
        return
    
    # Create directories
    create_directories()
    
    # Execute processing steps with progress bar
    for i, (script_name, description) in enumerate(steps[2:], 1):  # Skip dependency check and directory creation
        if script_name.endswith('.py'):
            if not run_script(script_name, description):
                print(f"ERROR: {description} failed, stopping processing")
                return
        else:
            print(f"Step {i}/{len(steps)-2}: {description}")
    
    print("All data processing steps completed!")

if __name__ == "__main__":
    main() 