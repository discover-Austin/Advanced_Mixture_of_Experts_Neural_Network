import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class MoELosses:
    """
    Advanced auxiliary losses for Mixture of Experts training.

    These losses help ensure:
    - Balanced expert utilization
    - Stable training
    - Efficient routing
    """

    @staticmethod
    def load_balancing_loss(
        gate_probs: torch.Tensor,
        num_experts: int,
        epsilon: float = 1e-10
    ) -> torch.Tensor:
        """
        Standard load balancing loss using variance.

        Encourages uniform distribution across experts.

        Args:
            gate_probs: (batch_size, num_experts)
            num_experts: Number of experts
            epsilon: Small constant for numerical stability
        """
        avg_probs = torch.mean(gate_probs, dim=0)
        ideal_load = 1.0 / num_experts
        return torch.mean((avg_probs - ideal_load) ** 2)

    @staticmethod
    def importance_loss(
        gate_probs: torch.Tensor,
        epsilon: float = 1e-10
    ) -> torch.Tensor:
        """
        Importance loss to prevent expert collapse.

        Penalizes when the total importance (sum of gate probabilities)
        becomes too concentrated.

        Args:
            gate_probs: (batch_size, num_experts)
        """
        importance = torch.sum(gate_probs, dim=0)
        return torch.var(importance) / (torch.mean(importance) ** 2 + epsilon)

    @staticmethod
    def z_loss(
        gate_logits: torch.Tensor,
        weight: float = 1e-4
    ) -> torch.Tensor:
        """
        Z-loss for router stability.

        From "ST-MoE: Designing Stable and Transferable Sparse Expert Models"
        https://arxiv.org/abs/2202.08906

        Penalizes large logits to prevent overflow and improve stability.

        Args:
            gate_logits: Raw logits before softmax (batch_size, num_experts)
            weight: Weight for the loss
        """
        return weight * torch.mean(torch.logsumexp(gate_logits, dim=1) ** 2)

    @staticmethod
    def router_z_loss(
        gate_logits: torch.Tensor,
        epsilon: float = 1e-10
    ) -> torch.Tensor:
        """
        Alternative formulation of router z-loss.

        Encourages router logits to stay small.
        """
        log_z = torch.logsumexp(gate_logits, dim=-1)
        return torch.mean(log_z ** 2)

    @staticmethod
    def expert_diversity_loss(
        gate_probs: torch.Tensor,
        target_diversity: float = 0.8
    ) -> torch.Tensor:
        """
        Encourage diversity in expert selection.

        Penalizes when too few experts are used.

        Args:
            gate_probs: (batch_size, num_experts)
            target_diversity: Target fraction of experts to use
        """
        batch_size, num_experts = gate_probs.shape

        # Count effective number of experts per sample
        # Using exponential of Shannon entropy
        epsilon = 1e-10
        entropy = -torch.sum(gate_probs * torch.log(gate_probs + epsilon), dim=1)
        effective_experts = torch.exp(entropy)

        target_count = target_diversity * num_experts
        diversity_penalty = F.mse_loss(effective_experts, torch.full_like(effective_experts, target_count))

        return diversity_penalty

    @staticmethod
    def entropy_regularization(
        gate_probs: torch.Tensor,
        target_entropy: Optional[float] = None
    ) -> torch.Tensor:
        """
        Regularize routing entropy.

        Can encourage either high entropy (exploration) or low entropy (exploitation).

        Args:
            gate_probs: (batch_size, num_experts)
            target_entropy: Target entropy (if None, uses max entropy)
        """
        epsilon = 1e-10
        entropy = -torch.sum(gate_probs * torch.log(gate_probs + epsilon), dim=1)

        if target_entropy is None:
            # Encourage maximum entropy (uniform distribution)
            num_experts = gate_probs.size(1)
            target_entropy = torch.log(torch.tensor(num_experts, dtype=torch.float32))

        return F.mse_loss(entropy, torch.full_like(entropy, target_entropy))

    @staticmethod
    def gini_loss(gate_probs: torch.Tensor) -> torch.Tensor:
        """
        Gini coefficient as a differentiable loss.

        Lower Gini = more balanced expert usage.

        Args:
            gate_probs: (batch_size, num_experts)
        """
        usage = torch.mean(gate_probs, dim=0)
        sorted_usage, _ = torch.sort(usage)
        n = len(sorted_usage)
        indices = torch.arange(1, n + 1, dtype=torch.float32, device=usage.device)
        cumsum = torch.cumsum(sorted_usage, dim=0)
        gini = (2 * torch.sum(indices * sorted_usage)) / (n * cumsum[-1]) - (n + 1) / n
        return gini

    @staticmethod
    def cv_squared_loss(gate_probs: torch.Tensor) -> torch.Tensor:
        """
        Coefficient of variation squared loss.

        From "Switch Transformers: Scaling to Trillion Parameter Models"
        https://arxiv.org/abs/2101.03961

        Args:
            gate_probs: (batch_size, num_experts)
        """
        num_experts = gate_probs.size(1)
        # Count how many times each expert is selected (approximated by sum of probs)
        expert_counts = torch.sum(gate_probs, dim=0)
        mean_count = torch.mean(expert_counts)
        cv_squared = (torch.var(expert_counts) / (mean_count ** 2 + 1e-10)) * num_experts
        return cv_squared

    @staticmethod
    def expert_capacity_loss(
        gate_probs: torch.Tensor,
        capacity: int,
        penalty_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Penalize when expert capacity is exceeded.

        Args:
            gate_probs: (batch_size, num_experts)
            capacity: Maximum number of tokens per expert
            penalty_weight: Weight for the penalty
        """
        batch_size = gate_probs.size(0)
        expert_loads = torch.sum(gate_probs, dim=0) * batch_size

        # Penalize loads exceeding capacity
        excess = F.relu(expert_loads - capacity)
        return penalty_weight * torch.mean(excess)

    @staticmethod
    def mutual_information_loss(
        gate_probs: torch.Tensor,
        weight: float = 0.01
    ) -> torch.Tensor:
        """
        Encourage high mutual information between inputs and expert assignments.

        Higher mutual information = experts specialize more.

        Args:
            gate_probs: (batch_size, num_experts)
            weight: Weight for the loss (negative because we want to maximize MI)
        """
        epsilon = 1e-10

        # P(expert)
        p_expert = torch.mean(gate_probs, dim=0)

        # H(expert) - entropy of expert distribution
        h_expert = -torch.sum(p_expert * torch.log(p_expert + epsilon))

        # H(expert|input) - conditional entropy (average entropy per input)
        h_expert_given_input = -torch.mean(torch.sum(gate_probs * torch.log(gate_probs + epsilon), dim=1))

        # MI = H(expert) - H(expert|input)
        mutual_info = h_expert - h_expert_given_input

        # Negative because we want to maximize MI (minimize negative MI)
        return -weight * mutual_info


class CompositeMoELoss(nn.Module):
    """
    Composite loss combining multiple MoE auxiliary losses.

    This makes it easy to train with multiple objectives.
    """

    def __init__(
        self,
        load_balance_weight: float = 0.01,
        importance_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        diversity_weight: float = 0.0,
        entropy_weight: float = 0.0,
        gini_weight: float = 0.0,
        cv_squared_weight: float = 0.0
    ):
        super(CompositeMoELoss, self).__init__()

        self.load_balance_weight = load_balance_weight
        self.importance_weight = importance_weight
        self.z_loss_weight = z_loss_weight
        self.diversity_weight = diversity_weight
        self.entropy_weight = entropy_weight
        self.gini_weight = gini_weight
        self.cv_squared_weight = cv_squared_weight

        self.moe_losses = MoELosses()

    def forward(
        self,
        gate_probs: torch.Tensor,
        gate_logits: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute composite loss.

        Args:
            gate_probs: Gating probabilities (batch_size, num_experts)
            gate_logits: Raw gate logits before softmax (batch_size, num_experts)
            return_components: Whether to return individual loss components

        Returns:
            total_loss or dict of losses
        """
        num_experts = gate_probs.size(1)
        losses = {}

        # Load balancing loss
        if self.load_balance_weight > 0:
            losses['load_balance'] = self.load_balance_weight * self.moe_losses.load_balancing_loss(
                gate_probs, num_experts
            )

        # Importance loss
        if self.importance_weight > 0:
            losses['importance'] = self.importance_weight * self.moe_losses.importance_loss(gate_probs)

        # Z-loss (requires logits)
        if self.z_loss_weight > 0 and gate_logits is not None:
            losses['z_loss'] = self.z_loss_weight * self.moe_losses.z_loss(gate_logits)

        # Diversity loss
        if self.diversity_weight > 0:
            losses['diversity'] = self.diversity_weight * self.moe_losses.expert_diversity_loss(gate_probs)

        # Entropy regularization
        if self.entropy_weight > 0:
            losses['entropy'] = self.entropy_weight * self.moe_losses.entropy_regularization(gate_probs)

        # Gini loss
        if self.gini_weight > 0:
            losses['gini'] = self.gini_weight * self.moe_losses.gini_loss(gate_probs)

        # CV squared loss
        if self.cv_squared_weight > 0:
            losses['cv_squared'] = self.cv_squared_weight * self.moe_losses.cv_squared_loss(gate_probs)

        if return_components:
            return losses

        # Return sum of all losses
        total_loss = sum(losses.values()) if losses else torch.tensor(0.0, device=gate_probs.device)
        return total_loss


class AdaptiveLossWeights(nn.Module):
    """
    Learnable loss weights for multi-objective MoE training.

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses"
    https://arxiv.org/abs/1705.07115
    """

    def __init__(self, num_losses: int):
        super(AdaptiveLossWeights, self).__init__()
        # Log variance parameters (learnable)
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted sum of losses with uncertainty weighting.

        Args:
            losses: Dictionary of named losses

        Returns:
            Weighted total loss
        """
        total_loss = 0
        for i, (name, loss) in enumerate(losses.items()):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]

        return total_loss
