#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified result evaluation script
Analyzes training results and generates basic reports
"""

import os
import json
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.simple_evaluator import SimpleExertionEvaluator

def load_cv_results(result_dir):
    """Load cross-validation results"""
    cv_results_path = os.path.join(result_dir, "cv_results.json")
    
    if not os.path.exists(cv_results_path):
        print(f"Error: Cross-validation results file not found: {cv_results_path}")
        return None
    
    with open(cv_results_path, 'r', encoding='utf-8') as f:
        cv_results = json.load(f)
    
    return cv_results

def main():
    """Main function"""
    print("Simplified result evaluation started")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find latest result directory
    result_base_dir = "result"
    if not os.path.exists(result_base_dir):
        print("Error: result directory not found")
        return
    
    result_dirs = [d for d in os.listdir(result_base_dir) 
                  if os.path.isdir(os.path.join(result_base_dir, d)) and d.startswith("VGG16")]
    
    if not result_dirs:
        print("Error: No VGG16 training result directories found")
        return
    
    # Select latest result directory
    latest_result_dir = sorted(result_dirs)[-1]
    result_dir = os.path.join(result_base_dir, latest_result_dir)
    
    print(f"Using result directory: {result_dir}")
    
    # Load cross-validation results
    cv_results = load_cv_results(result_dir)
    if cv_results is None:
        return
    
    print(f"Found {len(cv_results)} fold results")
    
    # Create simplified evaluator
    evaluator = SimpleExertionEvaluator(result_dir)
    
    # Run evaluation
    overall_results = evaluator.evaluate_cross_validation(cv_results)
    
    print("\n" + "="*50)
    print("Evaluation completed!")
    print("="*50)
    print(f"Results saved to: {result_dir}")
    
    # Print main metrics
    metrics = overall_results['metrics']
    print(f"\nMain metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"  Precision Macro: {metrics['precision_macro']:.4f}")
    print(f"  Recall Macro: {metrics['recall_macro']:.4f}")
    
    # Print F1 scores for each class
    print(f"\nF1 scores by class:")
    for i, f1_score in enumerate(metrics['f1_per_class']):
        print(f"  Class {i}: {f1_score:.4f}")
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
