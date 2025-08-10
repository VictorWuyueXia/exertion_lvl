# Results Directory

This directory contains all experimental results and outputs from the exertion level detection model.

## Directory Structure

```
results/
├── README.md                    # This file
├── advanced_metrics_results.json # Advanced evaluation metrics
├── *.png                        # Confusion matrices and other plots
└── experiments/                 # Individual experiment results
    ├── VGG16_wav2vec2_layer4_20250809_183249/
    ├── VGG16_wav2vec2_layer4_20250809_183057/
    └── ...                      # Other experiment directories
```

## Contents

- **advanced_metrics_results.json**: Contains detailed evaluation metrics including confusion matrices, accuracy scores, and other performance indicators
- **Confusion Matrix Plots**: PNG files showing confusion matrices for different evaluation methods
- **experiments/**: Individual experiment directories containing training logs, model checkpoints, and experiment-specific results

## Notes

- The `experiments/` directory is gitignored to avoid committing large experiment files
- Only summary results and key visualizations are tracked in git
- Each experiment directory contains timestamped results for reproducibility
