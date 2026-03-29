# FoundIR-V2 Guide

This guide covers the FoundIR-V2 implementation, including Data Equilibrium Scheduling and Mixture-of-Experts (MoE) architecture.

## Overview

FoundIR-V2 builds upon FoundIR-V1 with two key innovations:

1. **Data Equilibrium Scheduling**: Dynamically adjusts training data mixture ratios to ensure balanced performance across different degradation types
2. **Mixture-of-Experts (MoE)**: Uses multiple expert networks with task-adaptive routing for better generalization

## Quick Start

### V1 (Original)
```bash
python train.py
```

### V2 (with Data Equilibrium + MoE)
```bash
python train.py --version v2
```

### V2 with Custom Experts
```bash
python train.py --version v2 --num_experts 8 --moe_routing weighted
```

## Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--version` | str | `v1` | Model version: `v1` or `v2` |
| `--num_experts` | int | 8 | Number of experts for MoE (V2 only) |
| `--moe_routing` | str | `weighted` | MoE routing strategy: `weighted` (soft gating) or `hard` (top-k selection) |

## Architecture Details

### DataEquilibriumScheduler

Located in `src/model.py`, this scheduler:

- Tracks loss magnitudes for each degradation type
- Adjusts sampling weights periodically to achieve balanced training
- Prevents overfitting to dominant degradation types

```python
from src.model import DataEquilibriumScheduler

# Initialize with task types
task_types = ['denoising', 'deblurring', 'deraining', 'denoising', 'enhancement']
scheduler = DataEquilibriumScheduler(task_types, equilibrium_interval=1000, lr=0.01)

# Update with batch losses
scheduler.update(task_names=['denoising', 'deblurring'], losses=[0.5, 0.8])

# Get current sampling weights
weights = scheduler.get_weights()
```

### ExpertRouter

The router analyzes input features and determines which expert(s) should process the image:

- **Weighted routing**: All experts contribute with learned weights
- **Hard routing**: Single expert selected based on highest probability

### UnetResMoE

Multi-expert U-Net that combines outputs from multiple specialized networks:

```python
from src.model import UnetResMoE

model = UnetResMoE(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    num_experts=8,           # Number of expert networks
    condition=True,
    objective='pred_res',
    moe_routing='weighted',  # 'weighted' or 'hard'
)
```

## V2 Training Configuration

Default V2 settings differ from V1:

| Parameter | V1 | V2 |
|-----------|-----|-----|
| Training Steps | 500,000 | 1,000,000 |
| Delta End | 1.4e-3 | 1.5e-3 |
| Results Folder | `./ckpt_single_multi` | `./ckpt_v2` |

## Inference with V2

V2 models can be used with the same test.py:

```bash
python test.py --dataroot ./dataset --meta ./Testset_meta_info.txt
```

Ensure the checkpoint folder matches your trained model path.

## Performance Notes

- V2 requires more GPU memory due to multiple experts
- Recommended: 24GB+ GPU memory for training
- Use `--num_experts 4` for memory-constrained environments

## Citation

If using FoundIR-V2, please cite:

```bibtex
@article{chen2024foundirv2,
  title={FoundIR-v2: Optimizing Pre-Training Data Mixtures for Image Restoration Foundation Model},
  author={Chen, Xiang and Pan, Jinshan and Dong, Jiangxin and Yang, Jian and Tang, Jinhui},
  journal={arXiv preprint arXiv:2512.09282},
  year={2025}
}
```
