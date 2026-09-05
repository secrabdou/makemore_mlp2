Here is the complete, raw **`README.md`** file combining all sections (MLP architecture, BatchNorm mechanics, diagnostic suite, manual backpropagation equations, and verification code) into one block.

```markdown
# Deep MLP with Custom BatchNorm, Manual Backprop & Diagnostic Suite

A modular, from-scratch PyTorch implementation of a Deep Multi-Layer Perceptron (MLP) built without high-level abstractions (`torch.nn` or `loss.backward()`). This project explores deep network stability, custom Batch Normalization dynamics, parameter initialization theory, explicit manual backpropagation, and real-time activation/gradient diagnostics.

---

## Architecture Overview

* **Input Embedding:** Continuous character/token lookup matrix ($C$) mapping context tokens into hidden dimensions.
* **Network Depth:** 6 sequential blocks of `Linear` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `Tanh` activations.
* **Parameter Count:** ~47.5k trainable parameters.
* **Bias Strategy:** Explicitly toggled `bias=False` on Linear layers preceding BatchNorm (since BatchNorm's $\beta$ parameter absorbs the shift).

```python
# Layer Stack Definition
layers = [
    Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),           BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),           BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),           BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),           BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),                      BatchNorm1d(vocab_size)
]

```

---

## Key Features & Diagnostics

### 1. Custom `BatchNorm1d` Module

* **Dual-Mode Operation:** Uses dynamic mini-batch statistics ($\mu_B, \sigma_B^2$) during training (`self.training = True`) and frozen population statistics (`self.running_mean`, `self.running_var`) during evaluation (`self.training = False`).
* **Shift-Invariant Normalization:** Applies learnable $\gamma$ (gain) and $\beta$ (bias) parameters, allowing backprop to restore expressive capacity if normalized inputs limit activation ranges.

### 2. Parametric Weight Initialization (Kaiming / He)

* Weights initialized using Gaussian scaling $W \sim \mathcal{N}\left(0, \frac{\text{gain}}{\sqrt{\text{fan\_in}}}\right)$ with a $\text{gain} = \frac{5}{3}$ adapted for non-linear `Tanh` activations to prevent early saturation.

### 3. Diagnostic & Monitoring Suite

* **Activation Saturation Tracking:** Monitors distribution means, standard deviations, and clipping percentages across `Tanh` layers.
* **Gradient Distribution Monitoring:** Tracks magnitude vanishing/explosion layer-by-layer.
* **Update-to-Data Ratio:** Computes $\log_{10}\left(\frac{\text{step\_update}}{\text{data\_scale}}\right)$ across optimization steps to maintain the target optimal learning benchmark of $\approx -3.0$.

---

## Manual Backpropagation Engine

The training pipeline replaces PyTorch's `autograd` engine with an explicit, hand-derived computational graph.

### 1. Vectorized Embedding Gradient (`dC`)

Instead of slow Python loops, gradients arriving at the embedding layer are accumulated in parallel using CUDA-friendly index updates:

```python
dC = torch.zeros_like(C)
dC.index_add_(0, Xb.view(-1), demb.view(-1, n_embd))

```

### 2. Fast Analytic Softmax + Cross-Entropy Gradient (`dlogits`)

Rather than backpropagating through separate exponentiation, normalization, and log steps, the unified analytic gradient is computed directly in matrix form:

$$\text{dlogits} = \frac{\text{probs} - \text{one\_hot}(Y)}{N}$$

### 3. Decoupled BatchNorm Backpropagation

Exploiting multivariate chain rules and the shift-invariance property of variance relative to mean ($\frac{\partial \sigma^2}{\partial \mu} = 0$), the gradients for scale ($\gamma$), shift ($\beta$), normalized activations ($\hat{x}$), and batch variance ($\sigma^2$) are calculated directly:

$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma$$

$$\frac{\partial L}{\partial \sigma^2} = \sum_{i=1}^{N} \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu_B) \cdot \left(-\frac{1}{2}(\sigma_B^2 + \epsilon)^{-3/2}\right)$$

$$\frac{\partial L}{\partial \mu_B} = \left( \sum_{i=1}^{N} \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_B^2 + \epsilon}} \right) + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\sum_{i=1}^{N} -2(x_i - \mu_B)}{N}$$

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_B^2 + \epsilon}} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{2(x_i - \mu_B)}{N} + \frac{\partial L}{\partial \mu_B} \cdot \frac{1}{N}$$

---

## Gradient Verification & Correctness

To ensure absolute mathematical correctness, all manually derived gradients ($\text{dW}, \text{db}, \text{dC}, \text{dgamma}, \text{dbeta}$) were validated against PyTorch's native `autograd` tensors using direct tensor comparisons:

```python
# Verification helper across parameters
for p, dp in zip(parameters, grads):
    exact_match = torch.allclose(p.grad, dp)
    max_diff = (p.grad - dp).abs().max().item()
    print(f"Match: {exact_match} | Max Absolute Difference: {max_diff:.2e}")

```

> **Result:** All manual gradients match PyTorch `autograd` outputs to within machine epsilon precision ($< 10^{-8}$ maximum difference).

```

```
