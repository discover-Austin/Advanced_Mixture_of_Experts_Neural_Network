import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime


class CheckpointManager:
    """Manage model checkpoints with save/load functionality."""

    def __init__(self, save_dir: str = './checkpoints', keep_last_n: int = 5):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep (0 = keep all)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoints = []

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        model_name: str = "moe_model"
    ) -> str:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Dictionary of metrics
            scheduler: Learning rate scheduler (optional)
            extra_state: Any additional state to save
            is_best: Whether this is the best model so far
            model_name: Name prefix for checkpoint file

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if extra_state is not None:
            checkpoint['extra_state'] = extra_state

        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"{model_name}_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(str(checkpoint_path))

        # Save best model separately
        if is_best:
            best_path = self.save_dir / f"{model_name}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

        # Save latest checkpoint
        latest_path = self.save_dir / f"{model_name}_latest.pt"
        torch.save(checkpoint, latest_path)

        # Clean up old checkpoints
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints()

        print(f"Saved checkpoint to {checkpoint_path}")
        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load tensors to

        Returns:
            Dictionary with checkpoint information
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")

        return {
            'epoch': checkpoint['epoch'],
            'metrics': checkpoint.get('metrics', {}),
            'extra_state': checkpoint.get('extra_state', {})
        }

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        # Get all checkpoint files (excluding best and latest)
        checkpoint_files = [
            f for f in self.save_dir.glob("*.pt")
            if 'best' not in f.name and 'latest' not in f.name
        ]

        # Sort by modification time
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime)

        # Remove old checkpoints
        while len(checkpoint_files) > self.keep_last_n:
            old_checkpoint = checkpoint_files.pop(0)
            old_checkpoint.unlink()
            print(f"Removed old checkpoint: {old_checkpoint}")

    def load_best_model(
        self,
        model: torch.nn.Module,
        model_name: str = "moe_model",
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load the best saved model.

        Args:
            model: Model to load state into
            model_name: Name prefix of the model
            device: Device to load to

        Returns:
            Dictionary with checkpoint information
        """
        best_path = self.save_dir / f"{model_name}_best.pt"
        return self.load_checkpoint(str(best_path), model, device=device)

    def load_latest_model(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        model_name: str = "moe_model",
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load the latest checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            model_name: Name prefix of the model
            device: Device to load to

        Returns:
            Dictionary with checkpoint information
        """
        latest_path = self.save_dir / f"{model_name}_latest.pt"
        return self.load_checkpoint(
            str(latest_path),
            model,
            optimizer,
            scheduler,
            device
        )

    def save_model_only(
        self,
        model: torch.nn.Module,
        save_path: str
    ):
        """
        Save only the model state dict (for deployment).

        Args:
            model: Model to save
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")

    def export_to_onnx(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        save_path: str,
        device: str = 'cpu'
    ):
        """
        Export model to ONNX format.

        Args:
            model: Model to export
            input_shape: Shape of input tensor (excluding batch dimension)
            save_path: Path to save ONNX model
            device: Device to use for export
        """
        try:
            import onnx
            import onnxruntime
        except ImportError:
            print("ONNX export requires onnx and onnxruntime packages")
            return

        model.eval()
        dummy_input = torch.randn(1, *input_shape).to(device)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            str(save_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        print(f"Exported model to ONNX: {save_path}")


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better) or 'max' for accuracy (higher is better)
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, metric: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            metric: Current metric value
            epoch: Current epoch

        Returns:
            True if training should stop, False otherwise
        """
        score = -metric if self.mode == 'min' else metric

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! Best epoch: {self.best_epoch}")
                return True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

        return False

    def reset(self):
        """Reset early stopping."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class MetricsTracker:
    """Track and save training metrics."""

    def __init__(self, save_dir: str = './logs'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = {}
        self.epoch_metrics = []

    def update(self, metrics: Dict[str, float], epoch: int):
        """
        Update metrics for current epoch.

        Args:
            metrics: Dictionary of metric name -> value
            epoch: Current epoch
        """
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append(value)

        self.epoch_metrics.append({
            'epoch': epoch,
            **metrics
        })

    def save(self, filename: str = "metrics.json"):
        """Save metrics to JSON file."""
        save_path = self.save_dir / filename

        with open(save_path, 'w') as f:
            json.dump({
                'metrics_history': self.metrics_history,
                'epoch_metrics': self.epoch_metrics
            }, f, indent=2)

        print(f"Saved metrics to {save_path}")

    def get_best(self, metric_name: str, mode: str = 'min') -> tuple:
        """
        Get best value and epoch for a metric.

        Args:
            metric_name: Name of metric
            mode: 'min' or 'max'

        Returns:
            (best_value, best_epoch)
        """
        if metric_name not in self.metrics_history:
            return None, None

        values = self.metrics_history[metric_name]
        if mode == 'min':
            best_idx = min(range(len(values)), key=lambda i: values[i])
        else:
            best_idx = max(range(len(values)), key=lambda i: values[i])

        return values[best_idx], best_idx
