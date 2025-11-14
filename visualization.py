import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path


class MoEVisualizer:
    """Visualization tools for analyzing Mixture of Experts models."""

    def __init__(self, save_dir: str = "./visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        sns.set_style("whitegrid")

    def plot_expert_utilization(
        self,
        gate_probs: torch.Tensor,
        epoch: int,
        title: str = "Expert Utilization"
    ):
        """
        Plot how frequently each expert is used.

        Args:
            gate_probs: Tensor of shape (batch_size, num_experts)
            epoch: Current training epoch
            title: Plot title
        """
        gate_probs_np = gate_probs.detach().cpu().numpy()
        expert_usage = np.mean(gate_probs_np, axis=0)

        plt.figure(figsize=(10, 6))
        num_experts = len(expert_usage)
        plt.bar(range(num_experts), expert_usage)
        plt.xlabel("Expert Index")
        plt.ylabel("Average Utilization")
        plt.title(f"{title} (Epoch {epoch})")
        plt.axhline(y=1.0/num_experts, color='r', linestyle='--', label='Uniform Distribution')
        plt.legend()
        plt.tight_layout()

        save_path = self.save_dir / f"expert_utilization_epoch_{epoch}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return expert_usage

    def plot_routing_heatmap(
        self,
        gate_probs: torch.Tensor,
        epoch: int,
        max_samples: int = 100,
        title: str = "Routing Heatmap"
    ):
        """
        Create heatmap showing which experts process which inputs.

        Args:
            gate_probs: Tensor of shape (batch_size, num_experts)
            epoch: Current epoch
            max_samples: Maximum number of samples to visualize
        """
        gate_probs_np = gate_probs.detach().cpu().numpy()[:max_samples]

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            gate_probs_np.T,
            cmap="YlOrRd",
            cbar_kws={'label': 'Gating Probability'},
            xticklabels=False
        )
        plt.xlabel("Sample Index")
        plt.ylabel("Expert Index")
        plt.title(f"{title} (Epoch {epoch})")
        plt.tight_layout()

        save_path = self.save_dir / f"routing_heatmap_epoch_{epoch}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_expert_specialization(
        self,
        gate_probs: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
        title: str = "Expert Specialization by Class"
    ):
        """
        Analyze if experts specialize in particular classes.

        Args:
            gate_probs: Tensor of shape (batch_size, num_experts)
            labels: Ground truth labels (batch_size,)
            epoch: Current epoch
        """
        gate_probs_np = gate_probs.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        num_experts = gate_probs_np.shape[1]
        num_classes = len(np.unique(labels_np))

        # Compute average gating per class for each expert
        specialization = np.zeros((num_experts, num_classes))
        for class_idx in range(num_classes):
            class_mask = labels_np == class_idx
            if class_mask.any():
                specialization[:, class_idx] = np.mean(gate_probs_np[class_mask], axis=0)

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            specialization,
            annot=True,
            fmt='.3f',
            cmap="coolwarm",
            cbar_kws={'label': 'Average Gating Probability'}
        )
        plt.xlabel("Class Label")
        plt.ylabel("Expert Index")
        plt.title(f"{title} (Epoch {epoch})")
        plt.tight_layout()

        save_path = self.save_dir / f"expert_specialization_epoch_{epoch}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return specialization

    def plot_load_balance_over_time(
        self,
        load_history: List[np.ndarray],
        title: str = "Load Balance Over Time"
    ):
        """
        Plot how load balance evolves during training.

        Args:
            load_history: List of load arrays over epochs
        """
        load_history = np.array(load_history)
        num_epochs, num_experts = load_history.shape

        plt.figure(figsize=(14, 6))
        for expert_idx in range(num_experts):
            plt.plot(load_history[:, expert_idx], label=f'Expert {expert_idx}', alpha=0.7)

        plt.xlabel("Epoch")
        plt.ylabel("Load (Fraction of Inputs)")
        plt.title(title)
        plt.axhline(y=1.0/num_experts, color='black', linestyle='--',
                   label='Ideal Load', linewidth=2)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        save_path = self.save_dir / "load_balance_over_time.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_gating_entropy(
        self,
        gate_probs: torch.Tensor,
        epoch: int,
        title: str = "Routing Entropy Distribution"
    ):
        """
        Plot entropy of gating distribution per sample.
        High entropy = uncertain routing, Low entropy = confident routing

        Args:
            gate_probs: Tensor of shape (batch_size, num_experts)
            epoch: Current epoch
        """
        gate_probs_np = gate_probs.detach().cpu().numpy()

        # Compute entropy for each sample
        epsilon = 1e-10
        entropy = -np.sum(gate_probs_np * np.log(gate_probs_np + epsilon), axis=1)
        max_entropy = np.log(gate_probs_np.shape[1])

        plt.figure(figsize=(10, 6))
        plt.hist(entropy, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(max_entropy, color='r', linestyle='--',
                   label=f'Max Entropy ({max_entropy:.2f})')
        plt.xlabel("Routing Entropy")
        plt.ylabel("Frequency")
        plt.title(f"{title} (Epoch {epoch})")
        plt.legend()
        plt.tight_layout()

        save_path = self.save_dir / f"gating_entropy_epoch_{epoch}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return entropy

    def plot_hierarchical_routing(
        self,
        high_gates: torch.Tensor,
        low_gates: torch.Tensor,
        epoch: int,
        max_samples: int = 50
    ):
        """
        Visualize routing in hierarchical MoE.

        Args:
            high_gates: High-level gates (batch, num_high)
            low_gates: Low-level gates (batch, num_high, num_low)
            epoch: Current epoch
            max_samples: Max samples to show
        """
        high_gates_np = high_gates.detach().cpu().numpy()[:max_samples]
        low_gates_np = low_gates.detach().cpu().numpy()[:max_samples]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # High-level routing
        sns.heatmap(high_gates_np.T, ax=ax1, cmap="YlOrRd", cbar_kws={'label': 'Probability'})
        ax1.set_xlabel("Sample Index")
        ax1.set_ylabel("High-Level Expert")
        ax1.set_title(f"High-Level Routing (Epoch {epoch})")

        # Low-level routing (flatten for visualization)
        batch, num_high, num_low = low_gates_np.shape
        low_flat = low_gates_np.reshape(batch, num_high * num_low)
        sns.heatmap(low_flat.T, ax=ax2, cmap="YlOrRd", cbar_kws={'label': 'Probability'})
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Low-Level Expert (grouped by high-level)")
        ax2.set_title(f"Low-Level Routing (Epoch {epoch})")

        plt.tight_layout()
        save_path = self.save_dir / f"hierarchical_routing_epoch_{epoch}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_training_curves(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Training Metrics"
    ):
        """
        Plot multiple training metrics over time.

        Args:
            metrics: Dictionary mapping metric names to lists of values
        """
        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics))

        if num_metrics == 1:
            axes = [axes]

        for ax, (metric_name, values) in zip(axes, metrics.items()):
            ax.plot(values, linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, y=1.001)
        plt.tight_layout()

        save_path = self.save_dir / "training_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class ExpertMetrics:
    """Compute various metrics for MoE analysis."""

    @staticmethod
    def compute_gini_coefficient(gate_probs: torch.Tensor) -> float:
        """
        Compute Gini coefficient of expert utilization (0 = perfect balance, 1 = complete imbalance).

        Args:
            gate_probs: (batch_size, num_experts)
        """
        usage = gate_probs.mean(dim=0).cpu().numpy()
        sorted_usage = np.sort(usage)
        n = len(sorted_usage)
        cumsum = np.cumsum(sorted_usage)
        return (2 * np.sum((n - np.arange(n)) * sorted_usage)) / (n * cumsum[-1]) - (n + 1) / n

    @staticmethod
    def compute_routing_entropy(gate_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute routing entropy per sample.

        Args:
            gate_probs: (batch_size, num_experts)

        Returns:
            entropy: (batch_size,)
        """
        epsilon = 1e-10
        return -torch.sum(gate_probs * torch.log(gate_probs + epsilon), dim=1)

    @staticmethod
    def compute_expert_diversity(gate_probs: torch.Tensor, threshold: float = 0.1) -> int:
        """
        Count number of experts used above threshold.

        Args:
            gate_probs: (batch_size, num_experts)
            threshold: Minimum usage to count as "active"

        Returns:
            Number of diverse experts
        """
        avg_usage = gate_probs.mean(dim=0)
        return int((avg_usage > threshold).sum().item())

    @staticmethod
    def compute_specialization_score(
        gate_probs: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Compute how specialized experts are to particular classes.

        Higher score = more specialization

        Args:
            gate_probs: (batch_size, num_experts)
            labels: (batch_size,)
        """
        num_experts = gate_probs.size(1)
        num_classes = len(torch.unique(labels))

        specialization_matrix = torch.zeros(num_experts, num_classes)

        for class_idx in range(num_classes):
            mask = (labels == class_idx)
            if mask.any():
                specialization_matrix[:, class_idx] = gate_probs[mask].mean(dim=0)

        # Compute variance across classes for each expert
        specialization_scores = specialization_matrix.var(dim=1)
        return specialization_scores.mean().item()

    @staticmethod
    def compute_load_coefficient_of_variation(gate_probs: torch.Tensor) -> float:
        """
        Compute coefficient of variation for load distribution.

        Lower is better (more balanced).

        Args:
            gate_probs: (batch_size, num_experts)
        """
        load = gate_probs.mean(dim=0)
        cv = load.std() / (load.mean() + 1e-10)
        return cv.item()
