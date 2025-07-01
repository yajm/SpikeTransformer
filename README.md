# Spike Transformer

A from-scratch implementation of Spike-driven Transformers using only NumPy.

## Inspired by the following Papers

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) - Original Transformer
- [Spike-driven Transformer](https://arxiv.org/pdf/2307.01694) - First spike-based transformer
- [Spike-driven Transformer V2](https://arxiv.org/pdf/2404.03663) - Improved architecture
- [Scaling Spike-driven Transformer](https://arxiv.org/pdf/2411.16061v1) - Scaling techniques

## Objective:

Impressive Spike Transformers demo maximizing accuracy given the following constraints.

## Contraints:

- Keep everything simple and easy to understand
- No external libraries except numpy.
- Keep the implementation below 1000 lines
- Keep the training data less tha 1000 examples

## Getting Started

```bash
python3 SpikeTransformer
```

## High Score

```bash
============================================================
Training Spike-Driven Transformer with 780 examples
Architecture: 2 layers, 4 heads, 64 dimensions, Timesteps: 4
============================================================
Epoch   0: Loss = 2.0220, Accuracy = 20.6%, Avg Spike Rate = 0.229
Epoch  50: Loss = 1.3725, Accuracy = 50.0%, Avg Spike Rate = 0.353
Epoch 100: Loss = 1.1149, Accuracy = 60.3%, Avg Spike Rate = 0.348
Epoch 200: Loss = 0.9280, Accuracy = 67.2%, Avg Spike Rate = 0.375
Epoch 300: Loss = 0.7556, Accuracy = 72.3%, Avg Spike Rate = 0.361
Epoch 400: Loss = 0.6928, Accuracy = 75.9%, Avg Spike Rate = 0.359
Epoch 500: Loss = 0.6467, Accuracy = 77.3%, Avg Spike Rate = 0.365
```

## Dataset Planning

1. 100 Wikipedia Articles Famous Person
2. 100 Wikipedia Articles most viewed biology articles
3. 100 Wikipedia articles most viewed physices articles
4. 100 Wikipedia articles most viewed history articles
5. 100 Wikipedia articles most viewed chemistry articles
6. 100 Wikipedia articles most viewed mathematics articles
7. 100 Wikipedia articles most viewed geography articles
8. 100 Most famous books
9. 100 most famous philosophical pieces
10. 100 most citied papers of all time
