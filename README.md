# Deep MLP with Custom BatchNorm & Diagnostic Suite

A modular, from-scratch PyTorch implementation of a Deep Multi-Layer Perceptron (MLP) built without using `torch.nn`. This project focuses on deep network stability, custom Batch Normalization dynamics, parameter initialization theory, and real-time activation/gradient diagnostics.

---

## Key Features

- **Custom `BatchNorm1d`**: Built from scratch with running mean/variance buffers, momentum scaling, and explicit handling for training vs. evaluation modes.
- **Custom `Linear` Layer**: Supports bias toggle (`bias=False` when followed by BatchNorm) and custom weight scaling.
- **Parametric Initialization**: Implementation of Kaiming (He) initialization adapted for `Tanh` activations with theoretical gain scaling.
- **Model Checkpointing**: Automated parameter snapshot generation using `.clone().detach()` to capture the optimal parameters at peak validation performance.
- **Diagnostic Suite**:
  - Activation mean, standard deviation, and saturation tracking for `Tanh` layers.
  - Layer-wise gradient distribution monitoring.
  - Update-to-Data ratio tracking ($\log_{10}(\text{step\_update} / \text{data\_scale})$) targeted at the optimal $\sim -3.0$ diagnostic benchmark.

---

## Network Architecture & Parameters

- **Input Embedding**: Continuous character/token embeddings mapped to hidden space.
- **Depth**: 6 Linear + BatchNorm + Tanh layer blocks.
- **Parameter Count**: ~47.5k trainable parameters.

```python
# Layer Stack Definition
layers = [
    Linear(n_embd * block_size, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden),           BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden),           BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden),           BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden),           BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),          BatchNorm1d(vocab_size)
]
