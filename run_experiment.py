#!/usr/bin/env python3
"""
Comprehensive experiment runner for MoE models.

This script demonstrates all the advanced features of the expanded MoE framework.
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path

# Import MoE models
from m_o_e import MoE
from sparse_moe import TopKSparseMoE, ExpertChoiceMoE
from hierarchical_moe import HierarchicalMoE, RecursiveMoE
from advanced_experts import DeepExpert, AttentionExpert, ResidualExpert

# Import utilities
from config import ExperimentConfig, create_config_from_template
from data_loaders import get_data_loaders, get_flattened_mnist_loaders
from advanced_trainer import AdvancedMoETrainer, create_optimizer, create_scheduler
from losses import CompositeMoELoss
from evaluator import MoEEvaluator


def build_model(config: ExperimentConfig, device: str) -> nn.Module:
    """
    Build MoE model from configuration.

    Args:
        config: Experiment configuration
        device: Device to build model on

    Returns:
        MoE model
    """
    model_type = config.model_type.lower()

    if model_type == 'moe':
        moe_config = config.moe_config
        model = MoE(
            num_experts=moe_config.num_experts,
            input_size=moe_config.input_size,
            hidden_size=moe_config.hidden_size,
            output_size=moe_config.output_size,
            temperature=moe_config.temperature
        )

    elif model_type == 'sparse_moe':
        sparse_config = config.sparse_moe_config
        model = TopKSparseMoE(
            num_experts=sparse_config.num_experts,
            input_size=sparse_config.input_size,
            hidden_size=sparse_config.hidden_size,
            output_size=sparse_config.output_size,
            top_k=sparse_config.top_k,
            noise_std=sparse_config.noise_std,
            capacity_factor=sparse_config.capacity_factor,
            expert_type=sparse_config.expert_type,
            dropout_prob=sparse_config.dropout_prob
        )

    elif model_type == 'hierarchical_moe':
        hier_config = config.hierarchical_moe_config
        model = HierarchicalMoE(
            num_high_level_experts=hier_config.num_high_level_experts,
            num_low_level_experts=hier_config.num_low_level_experts,
            input_size=hier_config.input_size,
            hidden_size=hier_config.hidden_size,
            output_size=hier_config.output_size,
            temperature_high=hier_config.temperature_high,
            temperature_low=hier_config.temperature_low,
            dropout_prob=hier_config.dropout_prob
        )

    elif model_type == 'recursive_moe':
        rec_config = config.recursive_moe_config
        model = RecursiveMoE(
            depth=rec_config.depth,
            num_experts_per_level=rec_config.num_experts_per_level,
            input_size=rec_config.input_size,
            hidden_size=rec_config.hidden_size,
            output_size=rec_config.output_size,
            temperature=rec_config.temperature,
            dropout_prob=rec_config.dropout_prob
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def run_experiment(config: ExperimentConfig):
    """
    Run a complete experiment with the given configuration.

    Args:
        config: Experiment configuration
    """
    # Set random seed
    torch.manual_seed(config.training_config.seed)

    # Device setup
    device = config.training_config.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"\n{'='*80}")
    print(f"Running Experiment: {config.experiment_name}")
    print(f"Description: {config.description}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading data...")
    dataset_name = config.data_config.dataset

    if dataset_name == 'mnist':
        # For MNIST with fully connected MoE, use flattened images
        train_loader, val_loader, test_loader = get_flattened_mnist_loaders(
            batch_size=config.data_config.batch_size,
            num_workers=config.data_config.num_workers,
            train_split=config.data_config.train_split,
            data_dir=config.data_config.data_dir
        )
    else:
        train_loader, val_loader, test_loader = get_data_loaders(
            dataset_name=dataset_name,
            batch_size=config.data_config.batch_size,
            num_workers=config.data_config.num_workers,
            train_split=config.data_config.train_split,
            data_dir=config.data_config.data_dir,
            augmentation=config.data_config.augmentation
        )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Build model
    print("\nBuilding model...")
    model = build_model(config, device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    loss_config = config.loss_config
    if loss_config.primary_loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_config.primary_loss == 'mse':
        criterion = nn.MSELoss()
    elif loss_config.primary_loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    # MoE auxiliary losses
    moe_loss = CompositeMoELoss(
        load_balance_weight=loss_config.load_balance_weight,
        importance_weight=loss_config.importance_weight,
        z_loss_weight=loss_config.z_loss_weight,
        diversity_weight=loss_config.diversity_weight,
        entropy_weight=loss_config.entropy_weight,
        gini_weight=loss_config.gini_weight,
        cv_squared_weight=loss_config.cv_squared_weight
    )

    # Optimizer
    print("\nSetting up optimizer and scheduler...")
    optimizer = create_optimizer(model, config.optimizer_config.__dict__)
    scheduler = create_scheduler(optimizer, config.scheduler_config.__dict__)

    # Training configuration
    training_config = {
        'save_dir': config.training_config.save_dir,
        'log_dir': config.training_config.log_dir,
        'visualization_dir': config.training_config.visualization_dir,
        'early_stopping_patience': config.training_config.early_stopping_patience,
        'gradient_clip_norm': config.training_config.gradient_clip_norm,
        'use_tensorboard': config.training_config.use_tensorboard,
        'log_interval': config.training_config.log_interval,
        'save_interval': config.training_config.save_interval,
        'eval_interval': config.training_config.eval_interval,
        'visualization_interval': 20,
        'model_name': config.experiment_name,
        'keep_last_n': 5
    }

    # Create trainer
    trainer = AdvancedMoETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=training_config,
        device=device,
        scheduler=scheduler,
        moe_loss=moe_loss
    )

    # Train
    print("\nStarting training...")
    trainer.train(num_epochs=config.training_config.epochs)

    # Test
    print("\nTesting on test set...")
    test_metrics = trainer.test(test_loader)

    # Visualize training history
    print("\nGenerating visualizations...")
    trainer.visualize_training_history()

    print(f"\n{'='*80}")
    print("Experiment completed successfully!")
    print(f"Results saved to: {config.training_config.save_dir}")
    print(f"Logs saved to: {config.training_config.log_dir}")
    print(f"Visualizations saved to: {config.training_config.visualization_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Run MoE experiments")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (YAML)'
    )
    parser.add_argument(
        '--template',
        type=str,
        default='basic',
        choices=['basic', 'sparse', 'hierarchical', 'large_scale'],
        help='Configuration template to use'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Name for the experiment'
    )

    args = parser.parse_args()

    # Load or create configuration
    if args.config is not None:
        print(f"Loading configuration from: {args.config}")
        config = ExperimentConfig.load(args.config)
    else:
        print(f"Using template: {args.template}")
        config = create_config_from_template(args.template)

    # Override experiment name if provided
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name

    # Save configuration
    config_dir = Path(config.training_config.save_dir) / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{config.experiment_name}_config.yaml"
    config.save(str(config_path))
    print(f"Configuration saved to: {config_path}")

    # Run experiment
    run_experiment(config)


if __name__ == "__main__":
    main()
