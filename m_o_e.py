import torch
import torch.nn as nn
import torch.nn.functional as F
from expert import Expert

class MoE(nn.Module):
  
    def __init__(self, num_experts: int, input_size: int, hidden_size: int, output_size: int, temperature: float = 1.0):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        self.experts = nn.ModuleList([
            AdvancedExpert(input_size, hidden_size, output_size) for _ in range(num_experts)
        ])
        self.gating_layer = nn.Linear(input_size, num_experts)
    
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        gate_scores = self.gating_layer(x)  # Shape: (batch, num_experts)
        scaled_scores = gate_scores / self.temperature
        gate_probs = F.softmax(scaled_scores, dim=1) 
        expert_outputs = [expert(x) for expert in self.experts] 
        expert_outputs = torch.stack(expert_outputs, dim=1)
        gate_probs_expanded = gate_probs.unsqueeze(2)
        output = torch.sum(expert_outputs * gate_probs_expanded, dim=1) 
        return output, gate_probs
   
    def get_load_balancing_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        avg_probs = torch.mean(gate_probs, dim=0)  # Shape: (num_experts,)
        load_balancing_loss = torch.var(avg_probs)
        return load_balancing_loss