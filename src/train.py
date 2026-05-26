import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from model import BeamPINN
from utils import get_derivatives

# 1. Physical Parameters
E, I, L, q0 = 1.0, 1.0, 1.0, -1.0 

# 2. Setup
model = BeamPINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5000
N_f = 100 

print("Training started...")
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Domain Points
    x_colloc = torch.rand((N_f, 1), requires_grad=True) * L
    
    # Boundary Condition Points
    x_fixed_1 = torch.tensor([[0.0]], requires_grad=True)
    x_fixed_2 = torch.tensor([[L]], requires_grad=True)
    
    # PDE Loss
    v_colloc = model(x_colloc)
    _, _, _, d4v_dx4 = get_derivatives(v_colloc, x_colloc)
    loss_pde = torch.mean(((E * I * d4v_dx4) - q0)**2)
    
    # Boundary Condition Losses (Simply Supported)
    v_fixed_1 = model(x_fixed_1)
    _, d2v_fixed_1, _, _ = get_derivatives(v_fixed_1, x_fixed_1)
    
    v_fixed_2 = model(x_fixed_2)
    _, d2v_fixed_2, _, _ = get_derivatives(v_fixed_2, x_fixed_2)
    
    loss_bc_disp_1 = torch.mean(v_fixed_1**2) 
    loss_bc_moment_1 = torch.mean((E * I * d2v_fixed_1)**2)
    loss_bc_disp_2 = torch.mean(v_fixed_2**2) 
    loss_bc_moment_2 = torch.mean((E * I * d2v_fixed_2)**2)  
   
    loss = loss_pde + loss_bc_disp_1 + loss_bc_moment_1 + loss_bc_disp_2 + loss_bc_moment_2
    
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Total Loss: {loss.item():.6f}')

# Save Model
os.makedirs('../saved_models', exist_ok=True)
torch.save(model.state_dict(), '../saved_models/pinn_beam.pth')

# 3. Plotting
x_plot = torch.linspace(0, L, 200).view(-1, 1).requires_grad_(True)
v_plot = model(x_plot)
_, d2v_plot, d3v_plot, _ = get_derivatives(v_plot, x_plot)

x = x_plot.detach().numpy()
v = v_plot.detach().numpy()
M = E * I * d2v_plot.detach().numpy()
V = -E * I * d3v_plot.detach().numpy()

v_exact = (q0 * x) / (24 * E * I) * (x**3 - 2 * L * x**2 + L**3)
M_exact = (q0 * x / 2) * (x - L)
V_exact = q0 * (L / 2 - x)

fig, axs = plt.subplots(3, 1, figsize=(8, 10))
fig.tight_layout(pad=5.0)

axs[0].plot(x, v_exact, 'k--', label='Exact', linewidth=2)
axs[0].plot(x, v, 'r-', label='PINN', alpha=0.7, linewidth=3)
axs[0].set_title('Deflection v(x)')
axs[0].invert_yaxis() 
axs[0].grid(True); axs[0].legend()

axs[1].plot(x, M_exact, 'k--', label='Exact', linewidth=2)
axs[1].plot(x, M, 'b-', label='PINN', alpha=0.7, linewidth=3)
axs[1].set_title('Bending Moment Diagram (M)')
axs[1].grid(True); axs[1].legend()

axs[2].plot(x, V_exact, 'k--', label='Exact', linewidth=2)
axs[2].plot(x, V, 'g-', label='PINN', alpha=0.7, linewidth=3)
axs[2].set_title('Shear Force Diagram (V)')
axs[2].grid(True); axs[2].legend()

os.makedirs('../output_graphs', exist_ok=True)
plt.savefig('../output_graphs/pinn_beam_results.png', dpi=300, bbox_inches='tight')
plt.show()
