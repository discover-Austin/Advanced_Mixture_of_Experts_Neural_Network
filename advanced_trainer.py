import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any
from tqdm import tqdm
import os
from pathlib import Path

from checkpoint import CheckpointManager, EarlyStopping, MetricsTracker
from evaluator import MoEEvaluator
from visualization import MoEVisualizer, ExpertMetrics
from losses import CompositeMoELoss


class AdvancedMoETrainer:
    """
    Advanced trainer for MoE models with comprehensive features:
    - TensorBoard logging
    - Checkpointing
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping
    - Visualization
    - Expert utilization tracking
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        config: Dict[str, Any],
        device: str = 'cuda',
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        moe_loss: Optional[CompositeMoELoss] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.scheduler = scheduler
        self.moe_loss = moe_loss

        # Initialize components
        self.checkpoint_manager = CheckpointManager(
            save_dir=config.get('save_dir', './checkpoints'),
            keep_last_n=config.get('keep_last_n', 5)
        )

        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 20),
            mode='min',
            verbose=True
        )

        self.metrics_tracker = MetricsTracker(
            save_dir=config.get('log_dir', './logs')
        )

        self.evaluator = MoEEvaluator(device=device)

        self.visualizer = MoEVisualizer(
            save_dir=config.get('visualization_dir', './visualizations')
        )

        # TensorBoard
        if config.get('use_tensorboard', True):
            log_dir = Path(config.get('log_dir', './logs')) / 'tensorboard'
            self.writer = SummaryWriter(log_dir=str(log_dir))
        else:
            self.writer = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0

        # Expert metrics
        self.expert_metrics = ExpertMetrics()

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_pred_loss = 0.0
        total_aux_loss = 0.0
        all_gate_probs = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs, gate_probs = self.model(inputs)

            # Prediction loss
            pred_loss = self.criterion(outputs, labels)

            # Auxiliary MoE losses
            aux_loss = torch.tensor(0.0, device=self.device)
            if self.moe_loss is not None:
                aux_loss = self.moe_loss(gate_probs)

            # Total loss
            loss = pred_loss + aux_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.get('gradient_clip_norm') is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip_norm']
                )

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_aux_loss += aux_loss.item()
            all_gate_probs.append(gate_probs.detach().cpu())

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'pred': pred_loss.item(),
                'aux': aux_loss.item()
            })

            # TensorBoard logging (per batch)
            if self.writer is not None and batch_idx % self.config.get('log_interval', 10) == 0:
                self.writer.add_scalar('Train/batch_loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/pred_loss', pred_loss.item(), self.global_step)
                self.writer.add_scalar('Train/aux_loss', aux_loss.item(), self.global_step)

            self.global_step += 1

        # Epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_pred_loss = total_pred_loss / len(self.train_loader)
        avg_aux_loss = total_aux_loss / len(self.train_loader)

        # Concatenate all gate probabilities
        all_gate_probs = torch.cat(all_gate_probs, dim=0)

        # Compute expert metrics
        gate_entropy = self.expert_metrics.compute_routing_entropy(all_gate_probs).mean().item()
        gini_coef = self.expert_metrics.compute_gini_coefficient(all_gate_probs)
        expert_diversity = self.expert_metrics.compute_expert_diversity(all_gate_probs)

        metrics = {
            'train_loss': avg_loss,
            'train_pred_loss': avg_pred_loss,
            'train_aux_loss': avg_aux_loss,
            'train_gate_entropy': gate_entropy,
            'train_gini_coef': gini_coef,
            'train_expert_diversity': expert_diversity
        }

        return metrics, all_gate_probs

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        metrics = self.evaluator.evaluate(
            self.model,
            self.val_loader,
            self.criterion,
            return_gate_probs=True
        )

        # Rename keys to add 'val_' prefix
        val_metrics = {f'val_{k}': v for k, v in metrics.items()}

        return val_metrics

    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        # Resume from checkpoint if specified
        if resume_from is not None:
            checkpoint_info = self.checkpoint_manager.load_checkpoint(
                resume_from,
                self.model,
                self.optimizer,
                self.scheduler,
                self.device
            )
            self.current_epoch = checkpoint_info['epoch'] + 1
            print(f"Resumed from epoch {checkpoint_info['epoch']}")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_metrics, train_gate_probs = self.train_epoch()

            # Validate
            if (epoch + 1) % self.config.get('eval_interval', 5) == 0:
                val_metrics = self.validate()
            else:
                val_metrics = {}

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('val_loss', train_metrics['train_loss']))
                else:
                    self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']
                all_metrics['learning_rate'] = current_lr

            # Track metrics
            self.metrics_tracker.update(all_metrics, epoch)

            # TensorBoard logging
            if self.writer is not None:
                for name, value in all_metrics.items():
                    self.writer.add_scalar(f'Metrics/{name}', value, epoch)

                # Log expert utilization
                expert_usage = train_gate_probs.mean(dim=0)
                for expert_idx, usage in enumerate(expert_usage):
                    self.writer.add_scalar(f'ExpertUsage/expert_{expert_idx}', usage.item(), epoch)

            # Print metrics
            print("\nMetrics:")
            for name, value in all_metrics.items():
                print(f"  {name}: {value:.4f}")

            # Visualization
            if (epoch + 1) % self.config.get('visualization_interval', 20) == 0:
                self.visualizer.plot_expert_utilization(train_gate_probs, epoch)
                self.visualizer.plot_routing_heatmap(train_gate_probs, epoch)
                self.visualizer.plot_gating_entropy(train_gate_probs, epoch)

            # Checkpointing
            if (epoch + 1) % self.config.get('save_interval', 20) == 0:
                is_best = val_metrics.get('val_loss', train_metrics['train_loss']) < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics.get('val_loss', train_metrics['train_loss'])

                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    all_metrics,
                    self.scheduler,
                    is_best=is_best,
                    model_name=self.config.get('model_name', 'moe_model')
                )

            # Early stopping
            if val_metrics:
                if self.early_stopping(val_metrics['val_loss'], epoch):
                    print("Early stopping triggered!")
                    break

        # Save final metrics
        self.metrics_tracker.save()

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()

        print("\nTraining completed!")

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Test the model on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Test metrics
        """
        print("\nEvaluating on test set...")
        test_metrics, routing_info = self.evaluator.evaluate_with_routing_analysis(
            self.model,
            test_loader,
            self.criterion
        )

        print("\nTest Results:")
        for name, value in test_metrics.items():
            print(f"  {name}: {value:.4f}")

        # Analyze expert specialization
        specialization_info = self.evaluator.analyze_expert_specialization(
            self.model,
            test_loader
        )

        print("\nExpert Specialization:")
        print(f"  Specialization Matrix Shape: {specialization_info['specialization_matrix'].shape}")
        print(f"  Dominant Class Per Expert: {specialization_info['dominant_class_per_expert']}")
        print(f"  Expert Diversity (entropy): {specialization_info['expert_diversity']}")

        return test_metrics

    def visualize_training_history(self):
        """Visualize training history."""
        self.visualizer.plot_training_curves(self.metrics_tracker.metrics_history)


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer from configuration."""
    optimizer_type = config.get('type', 'adam').lower()
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)

    if optimizer_type == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=config.get('betas', (0.9, 0.999))
        )
    elif optimizer_type == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=config.get('momentum', 0.9)
        )
    elif optimizer_type == 'adagrad':
        return optim.Adagrad(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: optim.Optimizer,
    config: Dict[str, Any]
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler from configuration."""
    scheduler_type = config.get('type', 'none').lower()

    if scheduler_type == 'none':
        return None
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.1)
        )
    elif scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', 100),
            eta_min=config.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config.get('patience', 10),
            factor=config.get('gamma', 0.1),
            min_lr=config.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'exponential':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.get('gamma', 0.95)
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
