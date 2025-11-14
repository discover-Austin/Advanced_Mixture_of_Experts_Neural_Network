# Advanced Mixture of Experts Neural Network Framework

A comprehensive, production-ready implementation of Mixture of Experts (MoE) models with state-of-the-art features, exponentially expanded from a simple baseline to include hierarchical architectures, advanced routing mechanisms, and extensive tooling.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
- [Models](#models)
- [Expert Types](#expert-types)
- [Auxiliary Losses](#auxiliary-losses)
- [Configuration](#configuration)
- [Visualization](#visualization)
- [Examples](#examples)
- [Citation](#citation)

## Overview

This framework provides a complete implementation of Mixture of Experts neural networks, from basic MoE to advanced hierarchical and sparse architectures. It includes:

- **Multiple MoE architectures**: Standard, Sparse (Top-K), Hierarchical, Recursive
- **Advanced expert types**: Deep, Residual, Attention-based, Convolutional
- **Sophisticated routing**: Noisy Top-K gating, Expert Choice routing
- **Comprehensive training**: TensorBoard logging, checkpointing, early stopping
- **Extensive visualization**: Expert utilization, routing analysis, specialization metrics
- **Production-ready**: Configuration management, model export, extensive documentation

## Features

### Core Models

1. **Standard MoE**: Classic mixture of experts with temperature-scaled gating
2. **Sparse MoE**: Top-K expert selection with noisy gating (based on GShard/Switch Transformers)
3. **Hierarchical MoE**: Multi-level expert hierarchy (exponential capacity growth)
4. **Recursive MoE**: Arbitrary-depth expert trees
5. **Adaptive Depth MoE**: Dynamic computation with adaptive routing depth
6. **Expert Choice MoE**: Experts select tokens (reversed routing paradigm)

### Expert Architectures

- **Basic Expert**: Simple feedforward with batch normalization and dropout
- **Deep Expert**: Multiple layers with residual connections
- **Residual Expert**: ResNet-style blocks for deep feature learning
- **Attention Expert**: Multi-head self-attention for context-aware processing
- **Convolutional Expert**: For spatial/image data

### Advanced Features

- **Noisy Top-K Routing**: Stochastic expert selection with trainable noise
- **Load Balancing**: Multiple loss formulations (variance, importance, CV-squared, Gini)
- **Router Stability**: Z-loss and other regularization techniques
- **Expert Specialization**: Automatic analysis and visualization
- **TensorBoard Integration**: Real-time training monitoring
- **Checkpointing**: Automatic saving with best model tracking
- **Early Stopping**: Configurable patience-based stopping
- **Learning Rate Scheduling**: Step, Cosine, ReduceLROnPlateau, Exponential
- **Gradient Clipping**: Stable training for large models
- **Configuration Management**: YAML-based config with templates
- **Model Export**: ONNX support for deployment

## Architecture

### Exponential Scaling

The framework demonstrates exponential growth in model capacity:

```
Basic MoE (4 experts)              → 4 expert networks
Hierarchical MoE (4×4)             → 16 effective experts
Recursive MoE (4×4×4)              → 64 effective experts
Large Sparse MoE (64 experts, K=8) → 64 experts, 8 active per input
```

### Component Hierarchy

```
MoE Model
├── Gating Network
│   ├── Standard Gating
│   ├── Noisy Top-K Gating
│   └── Expert Choice Routing
├── Expert Networks
│   ├── Basic Expert
│   ├── Deep Expert
│   ├── Residual Expert
│   ├── Attention Expert
│   └── Convolutional Expert
└── Auxiliary Losses
    ├── Load Balancing
    ├── Importance Loss
    ├── Z-Loss
    ├── Diversity Loss
    └── Entropy Regularization
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements.txt Contents

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tensorboard>=2.13.0
tqdm>=4.65.0
pyyaml>=6.0
```

## Quick Start

### Basic Training

```python
import torch
from m_o_e import MoE
from train import train_model

# Create model
model = MoE(
    num_experts=4,
    input_size=784,
    hidden_size=128,
    output_size=10,
    temperature=1.0
)

# Synthetic data
input_data = torch.randn(1000, 784)
target_data = torch.randn(1000, 10)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

train_model(model, optimizer, criterion, input_data, target_data, epochs=100)
```

### Advanced Training with All Features

```bash
# Using command-line interface
python run_experiment.py --template sparse --experiment-name my_sparse_moe

# Using custom configuration
python run_experiment.py --config configs/custom_config.yaml
```

## Advanced Usage

### 1. Sparse MoE with Top-K Routing

```python
from sparse_moe import TopKSparseMoE

model = TopKSparseMoE(
    num_experts=16,
    input_size=784,
    hidden_size=128,
    output_size=10,
    top_k=4,              # Activate top-4 experts
    noise_std=1.0,        # Noisy gating for exploration
    capacity_factor=1.5   # Expert capacity multiplier
)

# Forward pass
output, gates, aux_info = model(input_data, add_noise=True)

# Load balancing loss
lb_loss = model.get_load_balancing_loss(aux_info)
```

### 2. Hierarchical MoE (Exponential Experts)

```python
from hierarchical_moe import HierarchicalMoE

model = HierarchicalMoE(
    num_high_level_experts=8,   # 8 high-level experts
    num_low_level_experts=8,    # Each contains 8 low-level experts
    input_size=784,
    hidden_size=128,
    output_size=10,
    temperature_high=2.0,
    temperature_low=1.0
)

# Total effective experts: 8 × 8 = 64
output, routing_info = model(input_data)

# Hierarchical load balancing
lb_loss = model.get_hierarchical_load_balancing_loss(routing_info)
```

### 3. Recursive MoE (Arbitrary Depth)

```python
from hierarchical_moe import RecursiveMoE

model = RecursiveMoE(
    depth=3,                           # 3-level hierarchy
    num_experts_per_level=[4, 4, 4],   # 4 experts at each level
    input_size=784,
    hidden_size=128,
    output_size=10
)

# Total experts: 4 × 4 × 4 = 64
print(f"Total experts: {model.count_total_experts()}")

output, gates_per_level = model(input_data)
```

### 4. Advanced Expert Architectures

```python
from advanced_experts import DeepExpert, AttentionExpert, ResidualExpert

# Deep expert with multiple hidden layers
deep_expert = DeepExpert(
    input_size=784,
    hidden_sizes=[256, 512, 256],  # 3 hidden layers
    output_size=10
)

# Attention-based expert
attention_expert = AttentionExpert(
    input_size=784,
    hidden_size=256,
    output_size=10,
    num_heads=8  # 8-head attention
)

# Residual expert (ResNet-style)
residual_expert = ResidualExpert(
    input_size=784,
    hidden_size=256,
    output_size=10,
    num_blocks=4  # 4 residual blocks
)
```

### 5. Comprehensive Training Pipeline

```python
from config import ExperimentConfig, create_config_from_template
from data_loaders import get_data_loaders
from advanced_trainer import AdvancedMoETrainer, create_optimizer, create_scheduler
from losses import CompositeMoELoss

# Load configuration
config = create_config_from_template('large_scale')

# Get data loaders
train_loader, val_loader, test_loader = get_data_loaders(
    dataset_name='mnist',
    batch_size=64,
    num_workers=4
)

# Build model (automatically from config)
model = build_model(config, device='cuda')

# Create optimizer and scheduler
optimizer = create_optimizer(model, config.optimizer_config.__dict__)
scheduler = create_scheduler(optimizer, config.scheduler_config.__dict__)

# Composite MoE loss
moe_loss = CompositeMoELoss(
    load_balance_weight=0.01,
    importance_weight=0.01,
    z_loss_weight=0.001,
    cv_squared_weight=0.01
)

# Create trainer with all features
trainer = AdvancedMoETrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=nn.CrossEntropyLoss(),
    config=training_config,
    device='cuda',
    scheduler=scheduler,
    moe_loss=moe_loss
)

# Train with TensorBoard, checkpointing, early stopping
trainer.train(num_epochs=100)

# Evaluate
test_metrics = trainer.test(test_loader)
trainer.visualize_training_history()
```

## Models

### Standard MoE (`m_o_e.py`)

Basic mixture of experts with temperature-scaled gating.

**Parameters:**
- `num_experts`: Number of expert networks
- `input_size`: Input feature dimension
- `hidden_size`: Hidden layer size for experts
- `output_size`: Output dimension
- `temperature`: Gating temperature (higher = more uniform)

### Sparse MoE (`sparse_moe.py`)

Top-K sparse gating with optional noise for exploration.

**Parameters:**
- `top_k`: Number of experts to activate per input
- `noise_std`: Standard deviation for noisy gating
- `capacity_factor`: Expert capacity multiplier

**Routing Methods:**
- Noisy Top-K Gating
- Expert Choice Routing (experts select tokens)

### Hierarchical MoE (`hierarchical_moe.py`)

Multi-level expert hierarchy for exponential scaling.

**Architectures:**
- `HierarchicalMoE`: 2-level hierarchy
- `RecursiveMoE`: Arbitrary depth
- `AdaptiveDepthMoE`: Dynamic computation depth per input

## Expert Types

### Available Experts (`advanced_experts.py`)

1. **DeepExpert**: Multiple hidden layers with residual connections
2. **AttentionExpert**: Multi-head self-attention mechanism
3. **ResidualExpert**: ResNet-style residual blocks
4. **ConvolutionalExpert**: For image/spatial data

Each expert type can be used in any MoE architecture.

## Auxiliary Losses

### Load Balancing Strategies (`losses.py`)

1. **Variance-based**: Penalize variance in expert usage
2. **CV-squared**: Coefficient of variation (from Switch Transformers)
3. **Gini Coefficient**: Inequality measure for expert distribution
4. **Importance Loss**: Prevent expert collapse

### Router Stability

1. **Z-loss**: Penalize large logits (from ST-MoE)
2. **Entropy Regularization**: Control routing certainty
3. **Diversity Loss**: Encourage diverse expert selection

### Composite Loss

```python
from losses import CompositeMoELoss

loss_fn = CompositeMoELoss(
    load_balance_weight=0.01,
    importance_weight=0.01,
    z_loss_weight=0.001,
    diversity_weight=0.005,
    entropy_weight=0.0,
    gini_weight=0.01,
    cv_squared_weight=0.01
)

aux_loss = loss_fn(gate_probs, gate_logits)
```

## Configuration

### YAML Configuration

```yaml
model_type: sparse_moe

sparse_moe_config:
  num_experts: 16
  input_size: 784
  hidden_size: 128
  output_size: 10
  top_k: 4
  noise_std: 1.0

training_config:
  epochs: 100
  early_stopping_patience: 20
  gradient_clip_norm: 1.0
  use_tensorboard: true

optimizer_config:
  type: adam
  lr: 0.001
  weight_decay: 0.0001

loss_config:
  load_balance_weight: 0.01
  importance_weight: 0.01
  z_loss_weight: 0.001
```

### Predefined Templates

- `basic`: Simple 4-expert MoE
- `sparse`: 16-expert sparse MoE with Top-4
- `hierarchical`: 8×8 hierarchical MoE
- `large_scale`: 64-expert sparse MoE

## Visualization

### Expert Utilization

```python
from visualization import MoEVisualizer

visualizer = MoEVisualizer(save_dir='./visualizations')

# Plot expert usage
visualizer.plot_expert_utilization(gate_probs, epoch=10)

# Routing heatmap
visualizer.plot_routing_heatmap(gate_probs, epoch=10)

# Expert specialization by class
visualizer.plot_expert_specialization(gate_probs, labels, epoch=10)

# Routing entropy distribution
visualizer.plot_gating_entropy(gate_probs, epoch=10)
```

### TensorBoard Integration

```bash
tensorboard --logdir=./logs/tensorboard
```

Tracks:
- Training/validation loss
- Expert utilization per epoch
- Routing entropy
- Learning rate
- Custom metrics

## Examples

### Example 1: Train on MNIST

```bash
python run_experiment.py \
    --template basic \
    --experiment-name mnist_basic_moe
```

### Example 2: Large-Scale Sparse MoE

```bash
python run_experiment.py \
    --template large_scale \
    --experiment-name large_sparse_moe
```

### Example 3: Hierarchical MoE

```bash
python run_experiment.py \
    --template hierarchical \
    --experiment-name hierarchical_moe_64experts
```

### Example 4: Custom Configuration

```bash
python run_experiment.py \
    --config my_config.yaml \
    --experiment-name custom_experiment
```

## File Structure

```
Advanced_Mixture_of_Experts_Neural_Network/
├── README.md                      # This file
├── requirements.txt               # Dependencies
│
├── Core Models
├── m_o_e.py                      # Standard MoE
├── expert.py                     # Basic expert module
├── sparse_moe.py                 # Sparse/Top-K MoE
├── hierarchical_moe.py           # Hierarchical MoE architectures
├── advanced_experts.py           # Advanced expert types
│
├── Training & Evaluation
├── train.py                      # Basic training script
├── advanced_trainer.py           # Full-featured trainer
├── evaluator.py                  # Comprehensive evaluation
├── losses.py                     # Auxiliary loss functions
│
├── Utilities
├── config.py                     # Configuration management
├── data_loaders.py               # Data loading utilities
├── checkpoint.py                 # Checkpointing & early stopping
├── visualization.py              # Visualization tools
│
├── Scripts
├── main.py                       # Simple demo script
├── run_experiment.py             # Full experiment runner
│
└── Outputs (created during training)
    ├── checkpoints/              # Model checkpoints
    ├── logs/                     # Training logs
    ├── visualizations/           # Generated plots
    └── configs/                  # Saved configurations
```

## Key Improvements from Baseline

This framework exponentially expands the original simple MoE implementation with:

1. **10+ Model Architectures**: From basic to hierarchical
2. **5+ Expert Types**: Simple to attention-based
3. **8+ Auxiliary Losses**: Comprehensive load balancing
4. **Production Features**: Checkpointing, logging, visualization
5. **Configuration System**: YAML-based, templated
6. **Extensive Evaluation**: Metrics, analysis, specialization
7. **Real Dataset Support**: MNIST, CIFAR-10, CIFAR-100
8. **Exponential Scaling**: Up to 64+ effective experts

## Performance Tips

1. **Start small**: Use `basic` template for initial experiments
2. **Tune temperature**: Lower temperature = sharper routing
3. **Balance auxiliary losses**: Start with small weights (0.001-0.01)
4. **Use Top-K for scale**: Sparse MoE is more efficient for many experts
5. **Monitor expert usage**: Ensure balanced utilization
6. **Enable gradient clipping**: Prevents training instability

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{advanced_moe_framework,
  title={Advanced Mixture of Experts Neural Network Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Advanced_Mixture_of_Experts_Neural_Network}
}
```

## References

- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017)
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models" (2021)
- Zhou et al., "Mixture-of-Experts with Expert Choice Routing" (2022)
- Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models" (2022)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

---

**Built with PyTorch** | **Production-Ready** | **Exponentially Scalable**
