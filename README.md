# CSGD: Custom Optimizer via Matrix Reparameterization

This repository contains the implementation and theoretical analysis of a custom stochastic gradient descent algorithm (CSGD), developed as part of a university course project. The optimizer is based on matrix reparameterization of fully connected layers and aims to improve convergence and generalization in training neural networks.

---

##  Overview

Instead of updating a weight matrix `A` directly, we factor it as `A = W₁W₂` and derive a new gradient update rule that incorporates the internal structure of `W₁` and `W₂`. This reparameterization leads to a different geometry of the optimization process and introduces implicit regularization.

---

##  Key Contributions

- ✔️ Analytical derivation of a custom update rule for reparameterized weights
- ✔️ Theoretical comparison between standard SGD and CSGD
- ✔️ Implementation of both optimizers in a simple autoencoder model
- ✔️ Experimental evaluation on the **Fashion-MNIST** dataset
- ✔️ Demonstrated improved performance using **MSE** and **SSIM** metrics

---

##  Model Architecture

The autoencoder consists of:

**Encoder**:
- Input layer: 28×28 → 784
- Linear → ReLU → Linear → `latent_dim`

**Decoder**:
- Linear → ReLU → Linear → Sigmoid
- Output reshaped to 28×28

---

## Results

| Metric | SGD    | CSGD   |
|--------|--------|--------|
| SSIM   | 0.6020 | **0.6555** |
| MSE    | 15.0783 | **12.3370** |

- CSGD achieved **faster convergence** and **better reconstruction** than standard SGD.
- The optimizer includes a term involving `AᵗA`, which stabilizes updates and improves training dynamics.
