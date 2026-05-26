# 🌉 Physics-Informed Neural Network (PINN) for Beam Mechanics

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)

A data-free Machine Learning project utilizing a Physics-Informed Neural Network (PINN) to solve the 4th-order Euler-Bernoulli beam differential equation. 

This network does not use a dataset. Instead, it is trained purely by evaluating its own spatial derivatives using PyTorch's `autograd` and minimizing the mathematical residual of the governing physics equations.

---

## ✨ The Physics

The network is trained to satisfy the Euler-Bernoulli governing equation across the continuous domain $x \in [0, L]$:

$$EI \frac{d^4v}{dx^4} = q(x)$$

Simultaneously, it enforces the hard physical boundary conditions for a **Simply Supported Beam**:
* Deflection at supports: $v(0) = 0$ and $v(L) = 0$
* Bending Moment at supports: $EI v''(0) = 0$ and $EI v''(L) = 0$

By differentiating the neural network's displacement output $v(x)$ with respect to spatial coordinate $x$, we can directly extract the **Bending Moment Diagram (BMD)** and **Shear Force Diagram (SFD)** purely from the network's learned weights.

---

## 📂 Project Structure

```text
pinn-beam-mechanics/
│
├── saved_models/           # Trained .pth model weights
├── output_graphs/          # Structural diagrams (Deflection, BMD, SFD)
│
├── src/                    
│   ├── model.py            # Neural network architecture (Tanh activations)
│   ├── utils.py            # Autograd helper functions for 4th-order derivatives
│   └── train.py            # Collocation sampling, physics loss, and plotting
│
├── .gitignore              
├── requirements.txt        
└── README.md
