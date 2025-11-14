import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from expert import Expert


class TopKSparseMoE(nn.Module):
    """
    Top-K Sparse Mixture of Experts with Noisy Gating.

    Based on "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
    https://arxiv.org/abs/1701.06538

    Only the top-k experts are activated per input, making it more efficient
    for large numbers of experts.
    """

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        top_k: int = 2,
        noise_std: float = 1.0,
        capacity_factor: float = 1.5,
        expert_type: str = 'basic',
        dropout_prob: float = 0.1
    ):
        super(TopKSparseMoE, self).__init__()

        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor
        self.input_size = input_size

        # Create experts
        self.experts = nn.ModuleList([
            Expert(input_size, hidden_size, output_size, dropout_prob)
            for _ in range(num_experts)
        ])

        # Gating network
        self.gate = nn.Linear(input_size, num_experts)
        self.gate_noise = nn.Linear(input_size, num_experts)

    def forward(
        self,
        x: torch.Tensor,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: Input tensor of shape (batch_size, input_size)
            add_noise: Whether to add noise to gating (for training)

        Returns:
            output: Mixed output from top-k experts
            gate_probs: Gating probabilities for all experts
            aux_info: Auxiliary information (load, importance, etc.)
        """
        batch_size = x.size(0)

        # Compute gating logits
        gate_logits = self.gate(x)  # (batch_size, num_experts)

        # Add noise during training (Noisy Top-K Gating)
        if add_noise and self.training:
            noise_stddev = F.softplus(self.gate_noise(x))
            noise = torch.randn_like(gate_logits) * noise_stddev * self.noise_std
            gate_logits = gate_logits + noise

        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=1)

        # Compute gating probabilities for top-k experts
        top_k_gates = F.softmax(top_k_logits, dim=1)

        # Sparse gating: zero out non-top-k experts
        gates = torch.zeros_like(gate_logits).scatter_(1, top_k_indices, top_k_gates)

        # Compute expert outputs for all experts (can be optimized to only compute top-k)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, output_size)

        # Weighted combination of expert outputs
        gates_expanded = gates.unsqueeze(2)  # (batch_size, num_experts, 1)
        output = torch.sum(expert_outputs * gates_expanded, dim=1)  # (batch_size, output_size)

        # Compute auxiliary information for losses
        aux_info = {
            'load': self._compute_load(gates),
            'importance': torch.sum(gates, dim=0),
            'top_k_indices': top_k_indices,
            'gates': gates
        }

        return output, gates, aux_info

    def _compute_load(self, gates: torch.Tensor) -> torch.Tensor:
        """Compute the load (fraction of inputs) assigned to each expert."""
        return torch.mean(gates, dim=0)

    def get_load_balancing_loss(self, aux_info: dict) -> torch.Tensor:
        """
        Compute load balancing loss to encourage equal expert utilization.

        Returns coefficient of variation of expert loads.
        """
        load = aux_info['load']
        return torch.var(load) / (torch.mean(load) ** 2 + 1e-10)

    def get_importance_loss(self, aux_info: dict) -> torch.Tensor:
        """
        Compute importance loss (variance of importance across experts).
        """
        importance = aux_info['importance']
        return torch.var(importance)


class NoisyTopKRouter(nn.Module):
    """
    Advanced noisy top-k routing mechanism.

    Includes trainable noise parameters and dynamic capacity allocation.
    """

    def __init__(
        self,
        input_size: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 1.0,
        use_tunable_noise: bool = True
    ):
        super(NoisyTopKRouter, self).__init__()

        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std
        self.use_tunable_noise = use_tunable_noise

        # Gating network
        self.gate_linear = nn.Linear(input_size, num_experts)

        # Tunable noise parameters
        if use_tunable_noise:
            self.noise_linear = nn.Linear(input_size, num_experts)

    def forward(
        self,
        x: torch.Tensor,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route inputs to top-k experts.

        Returns:
            gates: Gating weights (sparse, only top-k are non-zero)
            top_k_indices: Indices of selected experts
        """
        # Compute logits
        logits = self.gate_linear(x)

        # Add noise during training
        if add_noise and self.training:
            if self.use_tunable_noise:
                noise_stddev = F.softplus(self.noise_linear(x))
                noise = torch.randn_like(logits) * noise_stddev * self.noise_std
            else:
                noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=1)

        # Normalize only among top-k
        top_k_gates = F.softmax(top_k_logits, dim=1)

        # Create sparse gate tensor
        gates = torch.zeros_like(logits).scatter_(1, top_k_indices, top_k_gates)

        return gates, top_k_indices


class ExpertChoiceMoE(nn.Module):
    """
    Expert Choice routing: experts choose which tokens to process.

    Based on "Expert Choice: A New Perspective on Mixture-of-Experts"
    https://arxiv.org/abs/2202.09368

    Instead of tokens choosing experts, experts choose tokens up to their capacity.
    """

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        expert_capacity: int,
        dropout_prob: float = 0.1
    ):
        super(ExpertChoiceMoE, self).__init__()

        self.num_experts = num_experts
        self.expert_capacity = expert_capacity

        # Create experts
        self.experts = nn.ModuleList([
            Expert(input_size, hidden_size, output_size, dropout_prob)
            for _ in range(num_experts)
        ])

        # Router
        self.router = nn.Linear(input_size, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: Input of shape (batch_size, input_size)

        Returns:
            output: Processed output
            aux_info: Auxiliary routing information
        """
        batch_size = x.size(0)

        # Compute router scores
        router_logits = self.router(x)  # (batch_size, num_experts)
        router_probs = F.softmax(router_logits, dim=0)  # Softmax over batch dimension

        # Each expert selects top-k tokens
        output = torch.zeros(batch_size, self.experts[0].fc2.out_features, device=x.device)
        expert_mask = torch.zeros(batch_size, self.num_experts, device=x.device)

        for expert_idx, expert in enumerate(self.experts):
            # Get expert's preference for tokens
            expert_scores = router_probs[:, expert_idx]

            # Select top expert_capacity tokens
            top_scores, top_indices = torch.topk(
                expert_scores,
                min(self.expert_capacity, batch_size),
                dim=0
            )

            # Process selected tokens
            selected_inputs = x[top_indices]
            expert_output = expert(selected_inputs)

            # Accumulate output with gating weights
            output[top_indices] += expert_output * top_scores.unsqueeze(1)
            expert_mask[top_indices, expert_idx] = 1

        aux_info = {
            'router_probs': router_probs,
            'expert_mask': expert_mask
        }

        return output, aux_info
