# Mixture of Experts (MoE) 

I've demonstrated a simple but advanced Mixture of Experts (MoE) neural network from scratch utilizing my own edge technique using PyTorch:

# Architecture  
- Custom Expert Module: An expert with dropout and batch normalization.
- Temperature-Scaled Gating: Controllable sharpness of gating distribution.
- Load Balancing Loss: Encourages equal usage of experts to prevent dominance.

# Files
- expert.py: Contains the core expert module.
- m_o_e.py: Implements the core MoE model, incorporating temperature scaling and load balancing.
- train.py: Train MoE on a synthetic dataset.
- main.py: Runs the model

# Install Requirements 
'pip install -r requirements.txt'

# To Run
'python train_moe.py'