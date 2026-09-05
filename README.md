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
