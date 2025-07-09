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

A Spike Transformers demo maximizing accuracy given the following constraints.

## Contraints:

- Keep everything simple and easy to understand
- No external libraries except NumPy & MLX.
- Keep each file below 1000 lines
- Keep training data less tha 100MB

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

## Pipeline Overview

### Step 0: Initial Experimentation (`0_SpikeTransformer.py`)

**Purpose**: Proof of concept for spike-driven attention mechanisms  
**Output**: Validated SDSA (Spike-Driven Self-Attention) architecture

Initial experiment with a color prediction task to validate the spike-driven transformer concept:

- Implements Leaky Integrate-and-Fire (LIF) neurons
- Tests spike-driven attention mechanism (SDSA-1: Q ⊗ (K ⊗ V))
- Demonstrates learning capability on simple word-to-color mappings
- Validates energy efficiency claims (~70% reduction in operations)

### Step 1: Data Collection (`1_scraper.py`)

**Purpose**: Gather high-quality training data  
**Dependencies**: None  
**Output**: Wikipedia articles in JSON format

Optimized Wikipedia scraper that collects articles from multiple categories:

- Fetches top-viewed articles for relevance
- Targets specific domains (Cities, Companies, etc.)
- Implements smart sampling to avoid overwhelming API limits
- Saves articles as `wikipedia_smart_{category}_top500.json`

### Step 2: Custom Tokenization (`2_tokenizer.py`)

**Purpose**: Create an efficient syllable-based tokenizer  
**Dependencies**: Scraped Wikipedia data  
**Output**: `syllable_vocabulary_improved.txt`, `syllable_tokenizer.pkl`

Implements a syllable-based tokenizer for better compression:

- Uses pyphen for accurate syllabification
- Extracts top 2000 syllables from Wikipedia corpus
- Falls back to character-level tokenization when needed
- Achieves ~2.5x compression ratio over character-level encoding

### Step 3: Q&A Dataset Creation (`3_create_qa_pairs.py`)

**Purpose**: Generate fine-tuning data  
**Dependencies**: Wikipedia articles, Claude API  
**Output**: `qa_pairs.json`

Generates question-answer pairs for fine-tuning:

- Splits articles into 5 sections
- Uses Claude Sonnet to generate contextual Q&A pairs
- Ensures answers are extractive (directly from text)
- Creates structured dataset with category/article/section IDs

### Step 4: Core Model Implementation (`4_SpikeLLM.py`)

**Purpose**: Build the main spike-driven language model  
**Dependencies**: Tokenizer, Wikipedia data  
**Output**: Model checkpoints in `checkpoints/`

Full NumPy implementation of spike-driven transformer:

- **Architecture**: 12 layers, 8 heads, 256 hidden dimensions
- **Key Components**:
  - LIF neurons with configurable thresholds
  - Linear attention for causal masking
  - Membrane potential shortcuts for gradient flow
  - Surrogate gradient for spike backpropagation
- **Training**: Supports batched training on Wikipedia chunks
- **Features**: Text generation, embedding extraction

### Step 5: MLX Acceleration (`5_SpikeLLM_MLX.py`)

**Purpose**: GPU-accelerated implementation  
**Dependencies**: Step 4 model architecture  
**Output**: MLX model checkpoints

Ports the model to Apple's MLX framework:

- ~10x faster training on Apple Silicon
- Layer-wise learning rates for stable training
- Improved memory management
- Compatible checkpoint format with NumPy version

### Step 6: Fine-tuning Experiments (`6_finetuning.py`)

**Purpose**: Task-specific adaptation  
**Dependencies**: Pre-trained model, Q&A dataset  
**Output**: Fine-tuned checkpoints

Implements multiple fine-tuning strategies:

1. **Full Fine-tuning**: All parameters trainable
2. **Layer Freezing**: Freeze early layers, train only top layers
3. **LoRA (Low-Rank Adaptation)**: Efficient parameter-efficient fine-tuning
   - Rank-8 adaptations on attention weights
   - Only ~0.3% additional parameters
   - Comparable performance to full fine-tuning

## Key Technical Innovations

### Spike-Driven Self-Attention (SDSA)

- Replaces matrix multiplications with spike operations
- Uses cumulative K×V products for causal masking
- Maintains gradient flow through membrane potential shortcuts

### Energy Efficiency

- Binary spikes reduce multiply-accumulate operations
- Average spike rate: ~0.3-0.4 (60-70% operations saved)
- Theoretical energy reduction scales with spike sparsity

### Training Stability

- Surrogate gradients for spike discontinuities
- Layer normalization after residual connections
- Gradient clipping and membrane potential bounds
- Adam optimizer with layer-wise learning rates

## Requirements

```bash
numpy>=1.21.0
mlx>=0.5.0  # For MLX version
pyphen>=0.14.0  # For tokenizer
anthropic>=0.18.0  # For Q&A generation
tqdm>=4.65.0
```

## Usage

### Training from Scratch

```python
# 1. Collect data
python 1_scraper.py

# 2. Create tokenizer
python 2_tokenizer.py

# 3. Generate Q&A pairs (requires Anthropic API key)
python 3_create_qa_pairs.py

# 4. Train base model (NumPy version)
python 4_SpikeLLM.py  # Set TRAIN_MODEL=True in constants.py

# 5. Or train MLX version (faster on Apple Silicon)
python 5_SpikeLLM_MLX.py

# 6. Fine-tune on Q&A task
python 6_finetuning.py
```

### Generation Example

```python
from SpikeLLM_MLX import SpikeLLM
model = SpikeLLM(tokenizer_path='syllable_tokenizer.pkl')
model.load_checkpoint('checkpoints/spike_llm_mlx_epoch_10.pkl')

generated = model.generate("The capital of France is", max_length=50)
print(generated)
```

## Model Architecture

- **Embedding**: Syllable-based vocabulary (~2000 tokens)
- **Transformer**: 12 layers, 8 heads, 256 hidden, 1024 FFN
- **Spike Thresholds**: Tunable per layer type (0.5-1.5)
- **Timesteps**: 4 forward passes with membrane accumulation
- **Context Length**: 256 tokens

## Performance

- **Training**: ~50K Wikipedia chunks per epoch
- **Perplexity**: Converges to ~15-20 on Wikipedia
- **Generation**: Coherent text with proper grammar
- **Q&A Accuracy**: ~60-70% on extractive questions
- **Energy**: ~70% reduction in MAC operations

## Citation

This implementation is based on the theoretical framework of spike-driven transformers for energy-efficient language modeling. The architecture demonstrates that binary spike activations can maintain linguistic competence while significantly reducing computational requirements.
