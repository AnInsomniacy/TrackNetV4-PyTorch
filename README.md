# TrackNetV4 PyTorch Implementation

PyTorch implementation of **TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps**.

## Quick Start

```bash
git clone <repository-url>
cd tracknet-v4-pytorch
pip install -r requirements.txt

# Preprocess dataset
python preprocessing/video_to_heatmap.py --source dataset/raw --output dataset/preprocessed

# Train model
python train.py --data dataset/preprocessed --batch 4 --epochs 30

# Run predict
python predict/video_predict.py
```

## Quick Start for ubuntu cuda with UV manager 

```bash
git clone <repository-url>
cd tracknet-v4-pytorch
curl -LsSf https://astral.sh/uv/install.sh | sh

source ~/.bashrc

uv sync
uv run preprocessing/video_to_heatmap.py --source dataset/raw --output dataset/preprocessed
uv run train.py --data dataset/preprocessed --batch 4 --epochs 30

```

## Project Structure

```
tracknet-v4-pytorch/
├── model/
│   ├── tracknet.py              # TrackNetV4 architecture
│   └── loss.py                  # Weighted Binary Cross Entropy
├── preprocessing/
│   ├── video_to_heatmap.py      # Dataset preprocessing
│   ├── tracknet_dataset.py      # PyTorch dataset loader
│   └── data_visualizer.py       # Training data visualization
├── inference/
│   ├── single_frame_inference.py
│   └── video_inference.py
├── train.py                     # Training script
└── requirements.txt
```

## Core Implementation

### Motion-Aware Fusion Architecture

TrackNetV4 integrates motion attention maps with visual features through a lightweight motion prompt layer.

**Motion Prompt Layer:**

```python
class MotionPrompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))  # slope
        self.b = nn.Parameter(torch.randn(1))  # shift

    def forward(self, x):
        # Frame differencing: D_t = |F_{t+1} - F_t|
        diffs = [gray[:, i + 1] - gray[:, i] for i in range(T - 1)]
        D = torch.stack(diffs, dim=1)

        # Motion attention: A = sigmoid(a * |D| + b)
        A = torch.sigmoid(self.a * D.abs() + self.b)
        return A
```

**Motion-Aware Fusion:**

```python
def forward(self, vis, mot):
    # Fuse: [V_t, A_t ⊙ V_{t+1}, A_{t+1} ⊙ V_{t+2}]
    return torch.stack([vis[:, 0],
                        mot[:, 0] * vis[:, 1],
                        mot[:, 1] * vis[:, 2]], dim=1)
```

**Network Architecture:**

- Input: 3 consecutive RGB frames (9 channels, 288×512)
- Encoder: VGG16-style with skip connections
- Decoder: Symmetric upsampling
- Output: 3 probability heatmaps

### Weighted Binary Cross Entropy Loss

```python
def forward(self, y_pred, y_true):
    w = y_pred
    term1 = (1 - w) ** 2 * y_true * torch.log(y_pred)
    term2 = w ** 2 * (1 - y_true) * torch.log(1 - y_pred)
    return -(term1 + term2).mean()
```

## Dataset Format

### Input Structure

```
dataset/raw/match1/
├── csv/rally1_ball.csv          # Frame,Visibility,X,Y
└── video/rally1.mp4
```

### Processed Structure

```
dataset/preprocessed/match1/
├── inputs/rally1/               # 512×288 RGB frames
└── heatmaps/rally1/             # Gaussian heatmaps (sigma=3.0)
```

## Training

```bash
# Basic training
python train.py --data dataset/preprocessed

# Advanced options
python train.py --data dataset/preprocessed --batch 8 --epochs 50 --lr 1.5 --optimizer Adam

# Resume training
python train.py --resume checkpoints/best_model.pth --data dataset/preprocessed
```

## Inference

### Single Frame

```bash
python predict/single_frame_predict.py
```

### Video Processing

```bash
python predict/video_predict.py
```

### Visualization

```bash
python preprocessing/data_visualizer.py --source dataset/preprocessed/match1
```

## Dependencies

```bash
pip install torch torchvision opencv-python pandas numpy scipy tqdm matplotlib
```

## Citation

```bibtex
@article{raj2024tracknetv4,
    title={TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps},
    author={Raj, Arjun and Wang, Lei and Gedeon, Tom},
    journal={arXiv preprint arXiv:2409.14543},
    year={2024}
}
```