# Agent Coding Guidelines for FoundIR

This document provides guidelines for agents working on the FoundIR codebase.

## Project Overview

FoundIR is an image restoration foundation model. It uses PyTorch for deep learning, with diffusion-based architecture for image restoration tasks.

## FoundIR-V2 Support

The codebase now supports FoundIR-V2 with the following features:

### V1 vs V2 Comparison

| Feature | V1 | V2 |
|---------|-----|-----|
| Architecture | Single U-Net | Mixture-of-Experts (MoE) |
| Data Scheduling | Fixed mixture | Data Equilibrium Scheduling |
| Training Steps | 500,000 | 1,000,000 |
| Expert Count | N/A | Configurable (default 8) |

### V2 Classes

- `DataEquilibriumScheduler`: Dynamically adjusts sampling weights for different degradation types
- `ExpertRouter`: Routes input to appropriate expert based on learned features
- `UnetResMoE`: Multi-expert U-Net with weighted or hard routing

### V2 Training Arguments

```python
--version          # 'v1' or 'v2' (default: 'v1')
--num_experts      # Number of experts for MoE (default: 8)
--moe_routing      # 'weighted' or 'hard' (default: 'weighted')
```

## Environment Setup

```bash
# Create environment from yaml
conda env create -f environment.yml
conda activate FoundIR

# For specialist models
cd ./specialist_model
pip install -r requirements.txt
python setup.py develop
```

## Build, Test, and Training Commands

### Running Tests

```bash
# Test generalist model (requires pretrained model in ./premodel)
python test.py --dataroot ./dataset --meta ./Testset_meta_info.txt

# Test with custom data
python test.py --dataroot ./dataset --meta None

# Run training
sh train.sh

# Training commands from train.sh:
# Stage 1 (single degradation):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=7689 train.py --meta ./MillionIRData_single_train_meta_info.txt

# Stage 2 (all degradations):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=7689 train.py --meta ./MillionIRData_train_meta_info.txt
```

### Calculate Metrics

```bash
python cal_metrics.py --inp_imgs ./dataset/restored --gt_imgs ./dataset/GT --log path_save_log
```

### Specialist Model Inference

```bash
cd ./specialist_model
python inference_lowlight.py
# or
python inference_weather.py
```

## Code Style Guidelines

### Imports

- Standard library imports first, then third-party, then local
- Group imports by type with blank lines between groups
- Use explicit imports rather than `import *`

```python
# Correct
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from einops import rearrange, reduce
from ema_pytorch import EMA

from data.combined_dataset import CombinedDataset
from src.model import ResidualDiffusion, Trainer
```

### Naming Conventions

- **Functions/variables**: `snake_case` (e.g., `set_seed`, `tensor2img`)
- **Classes**: `PascalCase` (e.g., `ResidualDiffusion`, `Trainer`, `UnetRes`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_BATCH_SIZE`)
- **Private methods/variables**: prefix with underscore (e.g., `_internal_method`)

### Type Hints

- Use type hints for function parameters and return types when the logic is complex
- Prefer `typing` module for complex types

```python
def tensor2img(tensor: torch.Tensor, rgb2bgr: bool = True) -> np.ndarray:
    ...
```

### Formatting

- Maximum line length: 120 characters
- Use 4 spaces for indentation (not tabs)
- Use blank lines sparingly to separate logical sections
- No trailing whitespace

### Functions

- Keep functions focused and small (under 100 lines when possible)
- Use helper functions for repeated logic
- Use `partial` from functools for function parametrization

```python
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
```

### Classes

- Use `nn.Module` base class for all neural network modules
- Call `super().__init__()` first in `__init__`
- Use `nn.Parameter` for learnable parameters

```python
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
```

### Error Handling

- Use assertions for debugging and validating assumptions
- Raise specific exceptions with clear messages
- Use try/except for recoverable errors

```python
assert not (type(self) == ResidualDiffusion and model.channels != model.out_dim)
raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")
```

### Docstrings

- Use Google-style docstrings for public functions
- Include Args, Returns, and Raises sections when applicable
- Keep brief one-line summaries for simple functions

```python
def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes...
        rgb2bgr (bool): Whether to change rgb to bgr.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C)...
    """
```

### Comments

- Use comments sparingly - code should be self-documenting
- Use comments to explain complex algorithms or non-obvious logic
- Reference papers/code sources when implementing known algorithms

```python
class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
```

### Data Processing

- Use `torch.no_grad()` for inference
- Use `.detach().cpu()` before converting tensors to numpy
- Use `torch.clamp` or `.clamp_()` for in-place clamping

### Device Management

- Use `.to(device)` for tensor/model movement
- Check CUDA availability: `torch.cuda.is_available()`
- Use `accelerator` from `accelerate` library for distributed training

### Random Seeds

- Set random seeds for reproducibility using the existing `set_seed` function

```python
set_seed(10)
```

### Logging

- Use `tqdm` for progress bars during training/inference
- Use `logging` for persistent logs
- Print key metrics during training

### Testing New Code

- Test with small batch sizes first
- Verify shapes at key transformation points
- Test both training and inference modes
- Use smaller image sizes (e.g., 256) for quick iteration

### File Organization

```
FoundIR/
├── src/                 # Core model code
│   ├── model.py        # Main diffusion model and Trainer
│   └── visualization.py
├── data/               # Dataset classes
│   ├── combined_dataset.py
│   └── base_dataset.py
├── metrics/            # Evaluation metrics
├── specialist_model/   # Specialist models (lowlight, weather)
├── eval/               # Evaluation scripts
├── train.py            # Training entry point
├── test.py             # Testing entry point
├── cal_metrics.py      # Metrics calculation
├── train.sh            # Training shell script
└── environment.yml     # Conda environment
```

### Performance Tips

- Use `torch.compile()` if PyTorch version supports it
- Use mixed precision (`amp`) when memory is limited
- Use gradient accumulation for large effective batch sizes
- Use EMA (Exponential Moving Average) for model weights
