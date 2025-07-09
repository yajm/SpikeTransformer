# Spike Transformer

A from-scratch implementation of Spike-driven Transformers using only NumPy & MLX.

## Inspired by the following Papers

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) - Original Transformer
- [Spike-driven Transformer](https://arxiv.org/pdf/2307.01694) - First spike-based transformer
- [Spike-driven Transformer V2](https://arxiv.org/pdf/2404.03663) - Improved architecture
- [Scaling Spike-driven Transformer](https://arxiv.org/pdf/2411.16061v1) - Scaling techniques
- [Spike-HAR++: an energy-efficient and lightweight parallel spiking transformer](https://pmc.ncbi.nlm.nih.gov/articles/PMC11628275/pdf/fncom-18-1508297.pdf)
- [SpikeGPT: Generative Pre-trained Language Model
  with Spiking Neural Networks](https://arxiv.org/pdf/2302.13939)
- [SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking](https://arxiv.org/pdf/2407.04752)
- [Metadata Conditioning Accelerates Language
  Model Pre-training](https://arxiv.org/pdf/2501.01956v2)
- [Artificial Neurons Based on Ag/V2C/W Threshold
  Switching Memristors](https://pmc.ncbi.nlm.nih.gov/articles/PMC8623555/pdf/nanomaterials-11-02860.pdf)

## Objective:

Impressive Spike Transformers demo maximizing accuracy given the following constraints.

## Contraints:

- Keep everything simple and easy to understand
- No external libraries except NumPy & MLX.
- Keep the implementation below 1000 lines
- Keep the training data less tha 1000 examples

## Getting Started

```bash
python3 SpikeTransformer
```

## Roadmap

1. Scraper
2. Tokenizer
3. SpikeLLM
4. Implemenent MLX support for the M1 chip
5. Instruction Tuning
6. RLHF
7. Fine-tuning
8. Streaming
9. Reasoning Model

## ToDo

- increase context window
- increase training chunk per episode
- remove all chunks with <UNK>
- remove all chunks with sex or porn
- add retrieval id per generation
- maybe category id?
- change to float 8
- implement MLX
- remove all instruction tuning examples with UNK
- remove all instructino tuning examples question and answer above 128, 96, 64?

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
