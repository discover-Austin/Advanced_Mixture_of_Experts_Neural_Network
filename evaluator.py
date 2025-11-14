import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm


class MoEEvaluator:
    """Comprehensive evaluation for MoE models."""

    def __init__(self, device: str = 'cuda'):
        self.device = device

    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        return_gate_probs: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            model: MoE model
            data_loader: Data loader
            criterion: Loss function
            return_gate_probs: Whether to collect and return gate probabilities

        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_gate_probs = [] if return_gate_probs else None

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs, gate_probs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Collect predictions
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    # Classification
                    predictions = torch.argmax(outputs, dim=1)
                else:
                    # Regression
                    predictions = outputs.squeeze()

                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())

                if return_gate_probs:
                    all_gate_probs.append(gate_probs.cpu())

        # Concatenate all batches
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        # Compute metrics
        avg_loss = total_loss / len(data_loader)
        metrics = {'loss': avg_loss}

        # Classification metrics
        if all_predictions.dtype in [torch.long, torch.int]:
            accuracy = accuracy_score(all_labels.numpy(), all_predictions.numpy())
            metrics['accuracy'] = accuracy

            # Precision, recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels.numpy(),
                all_predictions.numpy(),
                average='weighted',
                zero_division=0
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1

        else:
            # Regression metrics
            mse = nn.MSELoss()(all_predictions, all_labels).item()
            mae = nn.L1Loss()(all_predictions, all_labels).item()
            metrics['mse'] = mse
            metrics['mae'] = mae

        # Gate probability statistics
        if return_gate_probs:
            all_gate_probs = torch.cat(all_gate_probs)
            metrics['avg_gate_entropy'] = self._compute_avg_entropy(all_gate_probs)
            metrics['gate_variance'] = torch.var(all_gate_probs.mean(dim=0)).item()

        return metrics

    def evaluate_with_routing_analysis(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """
        Evaluate with detailed routing analysis.

        Returns:
            metrics: Standard evaluation metrics
            routing_info: Detailed routing information
        """
        model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_gate_probs = []

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Evaluating with routing analysis"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs, gate_probs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1) if outputs.dim() > 1 else outputs.squeeze()

                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                all_gate_probs.append(gate_probs.cpu())

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_gate_probs = torch.cat(all_gate_probs)

        # Standard metrics
        metrics = {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy_score(all_labels.numpy(), all_predictions.numpy())
        }

        # Routing analysis
        routing_info = {
            'gate_probs': all_gate_probs,
            'labels': all_labels,
            'predictions': all_predictions,
            'expert_utilization': all_gate_probs.mean(dim=0),
            'routing_entropy': self._compute_entropy(all_gate_probs)
        }

        return metrics, routing_info

    def compute_confusion_matrix(
        self,
        model: nn.Module,
        data_loader: DataLoader
    ) -> np.ndarray:
        """Compute confusion matrix."""
        model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                outputs, _ = model(inputs)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)

        return confusion_matrix(all_labels, all_predictions)

    def print_classification_report(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        class_names: Optional[list] = None
    ):
        """Print detailed classification report."""
        model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                outputs, _ = model(inputs)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)

        print("\nClassification Report:")
        print(classification_report(
            all_labels,
            all_predictions,
            target_names=class_names,
            zero_division=0
        ))

    def analyze_expert_specialization(
        self,
        model: nn.Module,
        data_loader: DataLoader
    ) -> Dict[str, np.ndarray]:
        """
        Analyze how experts specialize on different classes.

        Returns:
            Dictionary with specialization analysis
        """
        model.eval()

        all_gate_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                _, gate_probs = model(inputs)

                all_gate_probs.append(gate_probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_gate_probs = np.concatenate(all_gate_probs)
        all_labels = np.concatenate(all_labels)

        num_experts = all_gate_probs.shape[1]
        num_classes = len(np.unique(all_labels))

        # Compute average gating per class for each expert
        specialization_matrix = np.zeros((num_experts, num_classes))

        for class_idx in range(num_classes):
            class_mask = (all_labels == class_idx)
            if class_mask.any():
                specialization_matrix[:, class_idx] = all_gate_probs[class_mask].mean(axis=0)

        # Compute expert diversity (entropy across classes)
        expert_diversity = []
        for expert_idx in range(num_experts):
            probs = specialization_matrix[expert_idx] + 1e-10
            probs = probs / probs.sum()
            entropy = -np.sum(probs * np.log(probs))
            expert_diversity.append(entropy)

        return {
            'specialization_matrix': specialization_matrix,
            'expert_diversity': np.array(expert_diversity),
            'dominant_class_per_expert': np.argmax(specialization_matrix, axis=1)
        }

    @staticmethod
    def _compute_entropy(gate_probs: torch.Tensor) -> torch.Tensor:
        """Compute routing entropy per sample."""
        epsilon = 1e-10
        return -torch.sum(gate_probs * torch.log(gate_probs + epsilon), dim=1)

    @staticmethod
    def _compute_avg_entropy(gate_probs: torch.Tensor) -> float:
        """Compute average routing entropy."""
        epsilon = 1e-10
        entropy = -torch.sum(gate_probs * torch.log(gate_probs + epsilon), dim=1)
        return entropy.mean().item()


def calculate_top_k_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5
) -> float:
    """
    Calculate top-k accuracy.

    Args:
        predictions: Model predictions (batch_size, num_classes)
        labels: Ground truth labels (batch_size,)
        k: Top-k value

    Returns:
        Top-k accuracy
    """
    _, top_k_preds = predictions.topk(k, dim=1)
    labels_expanded = labels.unsqueeze(1).expand_as(top_k_preds)
    correct = (top_k_preds == labels_expanded).any(dim=1).float()
    return correct.mean().item()


def calculate_calibration_error(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    Args:
        predictions: Softmax probabilities (batch_size, num_classes)
        labels: Ground truth labels (batch_size,)
        n_bins: Number of bins for calibration

    Returns:
        Expected Calibration Error
    """
    confidences, predicted_labels = torch.max(predictions, dim=1)
    accuracies = (predicted_labels == labels).float()

    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()
