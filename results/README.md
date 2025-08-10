# Results Directory

This directory contains all experimental results and outputs from the exertion level detection model.

## Directory Structure

```
results/
├── README.md                    # This file
├── advanced_metrics_results.json # Advanced evaluation metrics
├── *.png                        # Confusion matrices and other plots
├── models/                      # Saved model files
│   ├── VGG16_wav2vec2_layer4_exertion_detection_coral_v2_best.pt
│   ├── exertion_vggish_w2v2_coral_best.pt
│   └── vgg16_wav2vec2_mfcc_exertion_coral_best.pt
└── experiments/                 # Individual experiment results
    ├── latest_results_20250810_010957/  # Latest organized results
    ├── VGG16_wav2vec2_layer4_20250809_183249/
    ├── VGG16_wav2vec2_layer4_20250809_183057/
    └── ...                      # Other experiment directories
```

## Contents

- **advanced_metrics_results.json**: Contains detailed evaluation metrics including confusion matrices, accuracy scores, and other performance indicators
- **Confusion Matrix Plots**: PNG files showing confusion matrices for different evaluation methods
- **models/**: Saved model files containing the best performing weights from each experiment
  - **VGG16_wav2vec2_layer4_exertion_detection_coral_v2_best.pt**: Best model from VGG16 + wav2vec2 layer 4 experiment with CORAL loss
  - **exertion_vggish_w2v2_coral_best.pt**: Best model from VGGish + wav2vec2 experiment with CORAL loss
  - **vgg16_wav2vec2_mfcc_exertion_coral_best.pt**: Best model from VGG16 + wav2vec2 + MFCC experiment with CORAL loss
- **experiments/**: Individual experiment directories containing training logs, model checkpoints, and experiment-specific results
  - **latest_results_20250810_010957/**: Latest organized results with comprehensive documentation

## Notes

- The `experiments/` directory is gitignored to avoid committing large experiment files
- Only summary results, key visualizations, and model files are tracked in git
- Each experiment directory contains timestamped results for reproducibility
- Model files can be loaded using PyTorch for inference or further training
