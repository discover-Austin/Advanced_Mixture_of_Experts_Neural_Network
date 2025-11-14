import torch
import torch.nn as nn
import torch.optim as optim
from m_o_e import MoE

def train_model(model, optimizer, criterion, data, targets, epochs: int = 1000, lb_lambda: float = 0.1):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs, gate_probs = model(data)
        pred_loss = criterion(outputs, targets)       
        lb_loss = model.get_load_balancing_loss(gate_probs)
        total_loss = pred_loss + lb_lambda * lb_loss       
        total_loss.backward()
        optimizer.step()       
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Prediction Loss = {pred_loss.item():.4f}, "
                  f"Load Balancing Loss = {lb_loss.item():.4f}, Total Loss = {total_loss.item():.4f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    input_data = torch.randn(200, 15)
    target_data = (torch.tanh(input_data.sum(dim=1)) + 0.1 * torch.randn(200)).unsqueeze(1)
    NUM_EXPERTS = 4
    INPUT_SIZE = 15
    HIDDEN_SIZE = 30
    OUTPUT_SIZE = 1
    TEMPERATURE = 2.0
    moe = MoE(NUM_EXPERTS, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, TEMPERATURE)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(moe.parameters(), lr=0.005)
    
    print("Training MoE...")
    train_model(moe, optimizer, loss_function, input_data, target_data, epochs=1000, lb_lambda=0.1)
    moe.eval()
    with torch.no_grad():
        predictions, gate_probs = moe(input_data)
        final_loss = loss_function(predictions, target_data).item()
    print(f"Final Training Loss: {final_loss:.4f}")