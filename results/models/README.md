# Models Directory

This directory contains saved model files from the exertion level detection experiments.

## Model Files

### VGG16_wav2vec2_layer4_exertion_detection_coral_v2_best.pt
- **Architecture**: VGG16 + wav2vec2 layer 4 features
- **Loss Function**: CORAL (Consistent Rank Logits)
- **Purpose**: Best performing model from VGG16 + wav2vec2 experiment
- **Size**: ~6.0MB

### exertion_vggish_w2v2_coral_best.pt
- **Architecture**: VGGish + wav2vec2 features
- **Loss Function**: CORAL (Consistent Rank Logits)
- **Purpose**: Best performing model from VGGish + wav2vec2 experiment
- **Size**: ~6.0MB

### vgg16_wav2vec2_mfcc_exertion_coral_best.pt
- **Architecture**: VGG16 + wav2vec2 + MFCC features
- **Loss Function**: CORAL (Consistent Rank Logits)
- **Purpose**: Best performing model from VGG16 + wav2vec2 + MFCC experiment
- **Size**: ~6.0MB

## Usage

### Loading Models

```python
import torch
from src.models.vgg16_exertion import create_model

# Load model
model = create_model(config)
checkpoint = torch.load('results/models/model_name.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Model Information

Each checkpoint file contains:
- `model_state_dict`: Model weights and parameters
- `optimizer_state_dict`: Optimizer state (if saved)
- `epoch`: Training epoch when saved
- `best_metric`: Best validation metric achieved
- `config`: Model configuration used for training

## Notes

- All models use CORAL loss function for ordinal regression
- Models are trained on exertion level detection task (5 classes)
- Use corresponding config files to recreate model architecture
- Models are compatible with the current codebase structure
