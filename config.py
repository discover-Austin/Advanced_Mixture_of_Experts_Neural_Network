from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
import json
from pathlib import Path


@dataclass
class ExpertConfig:
    """Configuration for individual experts."""
    type: str = 'basic'  # 'basic', 'deep', 'residual', 'attention', 'convolutional'
    hidden_size: int = 64
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])  # For deep experts
    num_heads: int = 4  # For attention experts
    num_blocks: int = 2  # For residual experts
    dropout_prob: float = 0.1
    in_channels: int = 3  # For convolutional experts
    kernel_size: int = 3  # For convolutional experts


@dataclass
class MoEConfig:
    """Configuration for standard MoE."""
    num_experts: int = 4
    input_size: int = 784
    hidden_size: int = 64
    output_size: int = 10
    temperature: float = 1.0
    expert_config: ExpertConfig = field(default_factory=ExpertConfig)


@dataclass
class SparseMoEConfig:
    """Configuration for Sparse MoE with Top-K routing."""
    num_experts: int = 8
    input_size: int = 784
    hidden_size: int = 64
    output_size: int = 10
    top_k: int = 2
    noise_std: float = 1.0
    capacity_factor: float = 1.5
    expert_type: str = 'basic'
    dropout_prob: float = 0.1


@dataclass
class HierarchicalMoEConfig:
    """Configuration for Hierarchical MoE."""
    num_high_level_experts: int = 4
    num_low_level_experts: int = 4
    input_size: int = 784
    hidden_size: int = 64
    output_size: int = 10
    temperature_high: float = 1.0
    temperature_low: float = 1.0
    dropout_prob: float = 0.1


@dataclass
class RecursiveMoEConfig:
    """Configuration for Recursive MoE."""
    depth: int = 3
    num_experts_per_level: List[int] = field(default_factory=lambda: [4, 4, 4])
    input_size: int = 784
    hidden_size: int = 64
    output_size: int = 10
    temperature: float = 1.0
    dropout_prob: float = 0.1


@dataclass
class LossConfig:
    """Configuration for loss functions."""
    # Primary loss
    primary_loss: str = 'mse'  # 'mse', 'cross_entropy', 'bce'

    # Auxiliary loss weights
    load_balance_weight: float = 0.01
    importance_weight: float = 0.01
    z_loss_weight: float = 0.001
    diversity_weight: float = 0.0
    entropy_weight: float = 0.0
    gini_weight: float = 0.0
    cv_squared_weight: float = 0.0

    # Adaptive loss weights
    use_adaptive_weights: bool = False


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    type: str = 'adam'  # 'adam', 'sgd', 'adamw', 'adagrad'
    lr: float = 0.001
    weight_decay: float = 0.0001
    momentum: float = 0.9  # For SGD
    betas: tuple = (0.9, 0.999)  # For Adam


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""
    type: str = 'none'  # 'none', 'step', 'cosine', 'reduce_on_plateau', 'exponential'
    step_size: int = 30
    gamma: float = 0.1
    T_max: int = 100  # For cosine
    patience: int = 10  # For reduce on plateau
    min_lr: float = 1e-6


@dataclass
class DataConfig:
    """Configuration for data loading."""
    dataset: str = 'synthetic'  # 'synthetic', 'mnist', 'cifar10', 'cifar100'
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.8
    shuffle: bool = True
    augmentation: bool = False
    data_dir: str = './data'


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 100
    log_interval: int = 10
    save_interval: int = 20
    eval_interval: int = 5
    early_stopping_patience: int = 20
    gradient_clip_norm: Optional[float] = 1.0
    use_tensorboard: bool = True
    save_dir: str = './checkpoints'
    log_dir: str = './logs'
    visualization_dir: str = './visualizations'
    seed: int = 42
    device: str = 'cuda'  # 'cuda' or 'cpu'


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Model configuration
    model_type: str = 'moe'  # 'moe', 'sparse_moe', 'hierarchical_moe', 'recursive_moe'
    moe_config: Optional[MoEConfig] = None
    sparse_moe_config: Optional[SparseMoEConfig] = None
    hierarchical_moe_config: Optional[HierarchicalMoEConfig] = None
    recursive_moe_config: Optional[RecursiveMoEConfig] = None

    # Training configuration
    loss_config: LossConfig = field(default_factory=LossConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment metadata
    experiment_name: str = "default_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def save(self, path: str):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self._to_dict()

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls._from_dict(config_dict)

    def _to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            else:
                return obj

        return dataclass_to_dict(self)

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        # Helper function to reconstruct dataclass objects
        def dict_to_dataclass(dataclass_type, data):
            if data is None:
                return None
            if not isinstance(data, dict):
                return data

            field_types = {f.name: f.type for f in dataclass_type.__dataclass_fields__.values()}
            kwargs = {}

            for key, value in data.items():
                if key in field_types:
                    field_type = field_types[key]
                    # Check if field type is a dataclass
                    if hasattr(field_type, '__dataclass_fields__'):
                        kwargs[key] = dict_to_dataclass(field_type, value)
                    else:
                        kwargs[key] = value

            return dataclass_type(**kwargs)

        return dict_to_dataclass(cls, config_dict)


# Predefined configurations for common scenarios
def get_basic_config() -> ExperimentConfig:
    """Get basic MoE configuration."""
    return ExperimentConfig(
        model_type='moe',
        moe_config=MoEConfig(
            num_experts=4,
            input_size=784,
            hidden_size=64,
            output_size=10
        ),
        experiment_name='basic_moe'
    )


def get_sparse_config() -> ExperimentConfig:
    """Get sparse MoE configuration with Top-K routing."""
    return ExperimentConfig(
        model_type='sparse_moe',
        sparse_moe_config=SparseMoEConfig(
            num_experts=16,
            input_size=784,
            hidden_size=128,
            output_size=10,
            top_k=4
        ),
        experiment_name='sparse_moe_topk'
    )


def get_hierarchical_config() -> ExperimentConfig:
    """Get hierarchical MoE configuration."""
    return ExperimentConfig(
        model_type='hierarchical_moe',
        hierarchical_moe_config=HierarchicalMoEConfig(
            num_high_level_experts=8,
            num_low_level_experts=8,
            input_size=784,
            hidden_size=128,
            output_size=10
        ),
        experiment_name='hierarchical_moe'
    )


def get_large_scale_config() -> ExperimentConfig:
    """Get large-scale MoE configuration."""
    return ExperimentConfig(
        model_type='sparse_moe',
        sparse_moe_config=SparseMoEConfig(
            num_experts=64,
            input_size=784,
            hidden_size=256,
            output_size=10,
            top_k=8,
            noise_std=0.5
        ),
        loss_config=LossConfig(
            load_balance_weight=0.01,
            importance_weight=0.01,
            z_loss_weight=0.001,
            cv_squared_weight=0.01
        ),
        training_config=TrainingConfig(
            epochs=200,
            early_stopping_patience=30,
            gradient_clip_norm=1.0
        ),
        experiment_name='large_scale_sparse_moe'
    )


def create_config_from_template(template: str = 'basic') -> ExperimentConfig:
    """
    Create configuration from predefined template.

    Args:
        template: One of 'basic', 'sparse', 'hierarchical', 'large_scale'

    Returns:
        ExperimentConfig instance
    """
    templates = {
        'basic': get_basic_config,
        'sparse': get_sparse_config,
        'hierarchical': get_hierarchical_config,
        'large_scale': get_large_scale_config
    }

    if template not in templates:
        raise ValueError(f"Unknown template: {template}. Choose from {list(templates.keys())}")

    return templates[template]()


if __name__ == "__main__":
    # Example: Create and save configurations
    config = get_basic_config()
    config.save("configs/basic_moe.yaml")

    sparse_config = get_sparse_config()
    sparse_config.save("configs/sparse_moe.yaml")

    hierarchical_config = get_hierarchical_config()
    hierarchical_config.save("configs/hierarchical_moe.yaml")

    print("Configuration files created successfully!")
