import torch

def get_derivatives(v, x):
    """Computes up to the 4th derivative of v with respect to x using Autograd"""
    # 1st Derivative (Slope)
    dv_dx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), 
                                create_graph=True, retain_graph=True)[0]
    # 2nd Derivative (Moment / EI)
    d2v_dx2 = torch.autograd.grad(dv_dx, x, grad_outputs=torch.ones_like(dv_dx), 
                                  create_graph=True, retain_graph=True)[0]
    # 3rd Derivative (Shear / EI)
    d3v_dx3 = torch.autograd.grad(d2v_dx2, x, grad_outputs=torch.ones_like(d2v_dx2), 
                                  create_graph=True, retain_graph=True)[0]
    # 4th Derivative (Load / EI)
    d4v_dx4 = torch.autograd.grad(d3v_dx3, x, grad_outputs=torch.ones_like(d3v_dx3), 
                                  create_graph=True, retain_graph=True)[0]
    
    return dv_dx, d2v_dx2, d3v_dx3, d4v_dx4
