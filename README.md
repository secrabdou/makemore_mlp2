
### 2. Fast Softmax + Cross-Entropy Gradient (`dlogits`)
Rather than backpropagating through separate exponentiation, normalization, and log steps, the unified analytic gradient is computed directly in matrix form:

$$\text{dlogits} = \frac{\text{probs} - \text{one\_hot}(Y)}{N}$$

### 3. Decoupled BatchNorm Backpropagation
Leveraging multivariate chain rules and the mathematical identity showing that variance is shift-invariant to the mean ($\frac{\partial \sigma^2}{\partial \mu} = 0$), the backward pass for normalized activations $\hat{x}$ and batch variance $\sigma^2$ is computed directly across batch dimensions:

$$\frac{\partial L}{\partial \sigma^2} = \sum_{i=1}^{N} \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu) \cdot \left(-\frac{1}{2}(\sigma^2 + \epsilon)^{-3/2}\right)$$

---

## Network Architecture & Parameters

* **Input Embedding:** Continuous character/token embeddings mapped to hidden space.
* **Depth:** 6 Linear + BatchNorm + Tanh layer blocks.
* **Parameter Count:** ~47.5k trainable parameters.

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
