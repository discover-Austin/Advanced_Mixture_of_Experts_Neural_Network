import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from m_o_e import MoE
from expert import Expert


class HierarchicalMoE(nn.Module):
    """
    Hierarchical Mixture of Experts: A MoE where each expert is itself a MoE.

    This creates an exponential expansion of model capacity:
    - Level 0: Single gating network
    - Level 1: N high-level experts (each is a MoE)
    - Level 2: Each high-level expert contains M low-level experts

    Total effective experts: N * M
    """

    def __init__(
        self,
        num_high_level_experts: int,
        num_low_level_experts: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        temperature_high: float = 1.0,
        temperature_low: float = 1.0,
        dropout_prob: float = 0.1
    ):
        super(HierarchicalMoE, self).__init__()

        self.num_high_level = num_high_level_experts
        self.num_low_level = num_low_level_experts
        self.temperature_high = temperature_high
        self.temperature_low = temperature_low

        # High-level gating network
        self.high_level_gate = nn.Linear(input_size, num_high_level_experts)

        # Each high-level expert is itself a MoE
        self.high_level_experts = nn.ModuleList([
            MoE(
                num_experts=num_low_level_experts,
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                temperature=temperature_low
            )
            for _ in range(num_high_level_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through hierarchical MoE.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            output: Final output
            routing_info: Dictionary containing routing information at both levels
        """
        batch_size = x.size(0)

        # High-level gating
        high_gate_logits = self.high_level_gate(x)
        high_scaled_logits = high_gate_logits / self.temperature_high
        high_gate_probs = F.softmax(high_scaled_logits, dim=1)

        # Process through each high-level expert (which are MoEs)
        high_expert_outputs = []
        low_gate_probs_all = []

        for expert_moe in self.high_level_experts:
            expert_output, low_gate_probs = expert_moe(x)
            high_expert_outputs.append(expert_output)
            low_gate_probs_all.append(low_gate_probs)

        # Stack outputs
        high_expert_outputs = torch.stack(high_expert_outputs, dim=1)  # (batch, num_high, output_size)
        low_gate_probs_all = torch.stack(low_gate_probs_all, dim=1)    # (batch, num_high, num_low)

        # Combine high-level expert outputs
        high_gate_probs_expanded = high_gate_probs.unsqueeze(2)
        output = torch.sum(high_expert_outputs * high_gate_probs_expanded, dim=1)

        routing_info = {
            'high_level_gates': high_gate_probs,
            'low_level_gates': low_gate_probs_all,
            'combined_gates': self._compute_combined_gates(high_gate_probs, low_gate_probs_all)
        }

        return output, routing_info

    def _compute_combined_gates(
        self,
        high_gates: torch.Tensor,
        low_gates: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the effective gating probabilities for all low-level experts.

        Args:
            high_gates: (batch, num_high)
            low_gates: (batch, num_high, num_low)

        Returns:
            combined: (batch, num_high * num_low) - probability for each low-level expert
        """
        batch_size = high_gates.size(0)
        high_gates_expanded = high_gates.unsqueeze(2)  # (batch, num_high, 1)
        combined = high_gates_expanded * low_gates      # (batch, num_high, num_low)
        combined = combined.view(batch_size, -1)        # (batch, num_high * num_low)
        return combined

    def get_hierarchical_load_balancing_loss(self, routing_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute load balancing loss at both hierarchical levels.
        """
        # High-level load balancing
        high_avg_probs = torch.mean(routing_info['high_level_gates'], dim=0)
        high_lb_loss = torch.var(high_avg_probs)

        # Low-level load balancing (averaged across high-level experts)
        low_gates = routing_info['low_level_gates']
        low_avg_probs = torch.mean(low_gates, dim=(0, 1))  # Average over batch and high-level
        low_lb_loss = torch.var(low_avg_probs)

        # Combined load balancing loss
        return high_lb_loss + low_lb_loss


class RecursiveMoE(nn.Module):
    """
    Recursive Mixture of Experts with arbitrary depth.

    This allows creating exponentially deep hierarchies:
    depth=1: Regular MoE
    depth=2: MoE of MoEs
    depth=3: MoE of MoE of MoEs
    etc.
    """

    def __init__(
        self,
        depth: int,
        num_experts_per_level: List[int],
        input_size: int,
        hidden_size: int,
        output_size: int,
        temperature: float = 1.0,
        dropout_prob: float = 0.1
    ):
        super(RecursiveMoE, self).__init__()

        assert len(num_experts_per_level) == depth, "Must specify number of experts for each level"

        self.depth = depth
        self.num_experts_per_level = num_experts_per_level
        self.temperature = temperature

        if depth == 1:
            # Base case: create regular MoE
            self.moe = MoE(
                num_experts=num_experts_per_level[0],
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                temperature=temperature
            )
        else:
            # Recursive case: create gating + multiple recursive MoEs
            self.gate = nn.Linear(input_size, num_experts_per_level[0])
            self.experts = nn.ModuleList([
                RecursiveMoE(
                    depth=depth - 1,
                    num_experts_per_level=num_experts_per_level[1:],
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    temperature=temperature,
                    dropout_prob=dropout_prob
                )
                for _ in range(num_experts_per_level[0])
            ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through recursive MoE.

        Returns:
            output: Final output
            gates_per_level: List of gating probabilities at each level
        """
        if self.depth == 1:
            # Base case
            output, gates = self.moe(x)
            return output, [gates]
        else:
            # Recursive case
            gate_logits = self.gate(x)
            scaled_logits = gate_logits / self.temperature
            gate_probs = F.softmax(scaled_logits, dim=1)

            expert_outputs = []
            all_gates = [gate_probs]

            for expert in self.experts:
                expert_output, expert_gates = expert(x)
                expert_outputs.append(expert_output)
                # Only keep gates from first expert to avoid redundancy
                if len(all_gates) == 1:
                    all_gates.extend(expert_gates)

            expert_outputs = torch.stack(expert_outputs, dim=1)
            gate_probs_expanded = gate_probs.unsqueeze(2)
            output = torch.sum(expert_outputs * gate_probs_expanded, dim=1)

            return output, all_gates

    def count_total_experts(self) -> int:
        """Calculate total number of leaf experts in the hierarchy."""
        total = 1
        for num_experts in self.num_experts_per_level:
            total *= num_experts
        return total


class AdaptiveDepthMoE(nn.Module):
    """
    Adaptive Depth MoE: Dynamically determines routing depth per input.

    Some inputs may need simple processing (shallow routing),
    while others need complex processing (deep routing).
    """

    def __init__(
        self,
        max_depth: int,
        num_experts: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        threshold: float = 0.8
    ):
        super(AdaptiveDepthMoE, self).__init__()

        self.max_depth = max_depth
        self.threshold = threshold

        # Create MoE modules at each depth
        self.moe_layers = nn.ModuleList([
            MoE(
                num_experts=num_experts,
                input_size=input_size if i == 0 else output_size,
                hidden_size=hidden_size,
                output_size=output_size,
                temperature=1.0
            )
            for i in range(max_depth)
        ])

        # Halting probability predictor
        self.halting_predictor = nn.ModuleList([
            nn.Linear(output_size if i > 0 else input_size, 1)
            for i in range(max_depth)
        ])

    def forward(self, x: torch.Tensor, max_steps: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Adaptive computation time routing.

        Args:
            x: Input tensor
            max_steps: Maximum computation steps (default: self.max_depth)

        Returns:
            output: Accumulated output
            info: Dictionary with routing information
        """
        if max_steps is None:
            max_steps = self.max_depth

        batch_size = x.size(0)
        device = x.device

        # Initialize
        halting_prob = torch.zeros(batch_size, device=device)
        remainders = torch.zeros(batch_size, device=device)
        n_updates = torch.zeros(batch_size, device=device)
        output = torch.zeros(batch_size, self.moe_layers[0].experts[0].fc2.out_features, device=device)

        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        depth_per_sample = torch.zeros(batch_size, device=device)

        for step in range(max_steps):
            if not active_mask.any():
                break

            # Current input for active samples
            current_input = x if step == 0 else output

            # Process through MoE
            step_output, _ = self.moe_layers[step](current_input)

            # Predict halting probability
            p = torch.sigmoid(self.halting_predictor[step](current_input)).squeeze(1)

            # Update halting probability
            still_running = active_mask.float()
            new_halted = (halting_prob + p * still_running >= self.threshold).float() * still_running
            still_running = still_running - new_halted

            # Update remainders and outputs
            halting_prob_increment = new_halted * (self.threshold - halting_prob)
            halting_prob += p * still_running + halting_prob_increment
            remainders += new_halted * (1 - halting_prob)

            # Update output
            update_weights = p * still_running + halting_prob_increment
            output = output * (1 - update_weights.unsqueeze(1)) + step_output * update_weights.unsqueeze(1)

            # Track depth
            depth_per_sample += active_mask.float()

            # Update active mask
            active_mask = (halting_prob < self.threshold)
            n_updates += update_weights

        # Handle remaining samples
        if active_mask.any():
            output = output * (1 - remainders.unsqueeze(1)) + output * remainders.unsqueeze(1)

        info = {
            'n_updates': n_updates,
            'remainders': remainders,
            'depth_per_sample': depth_per_sample
        }

        return output, info
