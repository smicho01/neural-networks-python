# Lesson 6: How GPT-3 Works - Architecture

**Learning Objectives:**
- Understand GPT-3's scale and structure
- Learn how text becomes input
- Understand tokenization and embeddings
- See the 175 billion parameters in context

---

## What is GPT-3?

**GPT-3** (Generative Pre-trained Transformer 3) is a large language model developed by OpenAI in 2020.

### Key Specifications

| Attribute | Value |
|-----------|-------|
| **Parameters** | 175 billion |
| **Layers** | 96 transformer layers |
| **Context window** | 4,096 tokens (~3,000 words) |
| **Vocabulary** | 50,257 tokens |
| **Hidden size** | 12,288 dimensions |
| **Training data** | ~570 GB of text |
| **Training cost** | $4-12 million (estimated) |

---

## Input Processing

### Text-Only Model

**Important:** GPT-3 processes only text, not images or audio.

```
Supported Input:
  ✓ Text prompts
  ✓ Questions
  ✓ Conversations
  ✓ Code

Not Supported:
  ✗ Images (added in GPT-4)
  ✗ Audio
  ✗ Video
```

### Input Size Limit

```
Maximum input: 4,096 tokens

Token ≈ 0.75 words on average

Examples:
  "Hello" = 1 token
  "Hello world" = 2 tokens
  "The quick brown fox" = 4 tokens
  
Maximum ≈ 3,000 words per request
```

---

## From Text to Numbers

### Step 1: Tokenization

Text is broken into tokens (subwords):

```
Example Text:
"Hello, how are you?"

Tokenized:
"Hello"  → Token ID: 15496
","      → Token ID: 11
" how"   → Token ID: 703
" are"   → Token ID: 389
" you"   → Token ID: 345
"?"      → Token ID: 30

Result: [15496, 11, 703, 389, 345, 30]
6 tokens
```

### Why Tokens, Not Words?

```
Word-based problems:
  - "running" and "run" treated as different
  - Unknown words can't be processed
  - Vocabulary would be too large

Token-based advantages:
  ✓ "run", "running", "runner" share parts
  ✓ Can handle any word (break into pieces)
  ✓ Manageable vocabulary size (50,257)
```

### Tokenization Examples

```
Simple words:
  "cat" → ["cat"] (1 token)
  "dog" → ["dog"] (1 token)

Complex words:
  "unbelievable" → ["un", "believ", "able"] (3 tokens)
  "preprocessing" → ["pre", "process", "ing"] (3 tokens)

Numbers:
  "2024" → ["202", "4"] (2 tokens)
  "3.14159" → ["3", ".", "1415", "9"] (4 tokens)

Code:
  "function" → ["function"] (1 token)
  "console.log" → ["console", ".", "log"] (3 tokens)
```

---

## Step 2: Embedding

Each token is converted to a high-dimensional vector:

```
Token ID: 15496 ("Hello")
      ↓
Embedding Layer (learned during training)
      ↓
12,288-dimensional vector:
[0.123, -0.456, 0.789, -0.234, 0.567, ..., 0.890]
(12,288 numbers that represent "Hello")
```

### Embedding Visualization (Simplified to 2D)

```
In reality: 12,288 dimensions
For visualization: 2 dimensions

   Dimension 2
        ↑
        |    "king"●
        |              ●"queen"
        |
        |    "man"●
        |              ●"woman"
        |
        └─────────────────────→ Dimension 1
        
Similar words cluster together!
Related concepts have similar vectors!
```

### Embedding Parameters

```
Embedding Layer:
  Vocabulary size: 50,257 tokens
  Embedding dimensions: 12,288
  
Parameters: 50,257 × 12,288 = 617,155,584
           (~617 million parameters)

This is 0.35% of GPT-3's total parameters!
```

---

## Architecture Overview

### Complete Flow

```
User Input: "Explain quantum physics"
      ↓
┌─────────────────────────────────────────────┐
│ TOKENIZATION                                │
│ ["Explain", "quantum", "physics"]           │
│ → [18438, 14821, 25571]                    │
└─────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────┐
│ EMBEDDING LAYER (~617M params)              │
│ Convert tokens to 12,288-dim vectors        │
│ [18438] → [0.12, -0.34, ..., 0.89]        │
│ [14821] → [0.45, -0.12, ..., 0.23]        │
│ [25571] → [0.67, -0.89, ..., 0.45]        │
└─────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────┐
│ TRANSFORMER LAYER 1 (~1.8B params)          │
│ - Multi-head attention                      │
│ - Feedforward network                       │
│ - Layer normalization                       │
└─────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────┐
│ TRANSFORMER LAYER 2 (~1.8B params)          │
└─────────────────────────────────────────────┘
      ↓
      ... (94 more layers)
      ↓
┌─────────────────────────────────────────────┐
│ TRANSFORMER LAYER 96 (~1.8B params)         │
└─────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────┐
│ OUTPUT LAYER (~617M params)                 │
│ Predict next token probabilities            │
│ 50,257 probabilities (one per token)        │
└─────────────────────────────────────────────┘
      ↓
Generated Text: "Quantum physics is..."
```

---

## Parameter Distribution

### Breakdown by Component

```
┌──────────────────────────────────────────────┐
│ Embedding Layer                              │
│ Parameters: 617 million (0.35%)              │
└──────────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────────┐
│ 96 Transformer Layers                        │
│ Each: ~1.8 billion parameters                │
│ Total: 96 × 1.8B = 172.8 billion (98.7%)     │
└──────────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────────┐
│ Output Layer                                 │
│ Parameters: 617 million (0.35%)              │
└──────────────────────────────────────────────┘

═══════════════════════════════════════════════
TOTAL: ~175 billion parameters
═══════════════════════════════════════════════

Distribution:
  Embedding:    ▌ 0.35%
  Layers 1-96:  ████████████████████████ 98.7%
  Output:       ▌ 0.35%
```

### Single Transformer Layer

Each of the 96 layers contains:

```
Layer Structure (~1.8B parameters each):

┌─────────────────────────────────────┐
│ Multi-Head Attention                │
│ - Query weights                     │
│ - Key weights                       │
│ - Value weights                     │
│ - Output projection                 │
│ Parameters: ~900 million            │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ Feedforward Network                 │
│ - Dense layer 1 (12288 → 49152)    │
│ - Dense layer 2 (49152 → 12288)    │
│ Parameters: ~900 million            │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ Layer Normalization                 │
│ Parameters: ~25,000                 │
└─────────────────────────────────────┘

Total per layer: ~1.8 billion
```

---

## Context Window

### What is Context?

The context window is the maximum amount of text GPT-3 can "see" at once:

```
Context Window: 4,096 tokens

Can process:
  ✓ Prompt: 100 tokens
  ✓ Previous conversation: 500 tokens  
  ✓ Reference document: 2,000 tokens
  ✓ Space for response: 1,496 tokens
  Total: 4,096 tokens ✓

Cannot process:
  ✗ Entire book (100,000+ tokens)
  ✗ Very long conversations
  ✗ Large datasets
```

### Context Window Evolution

| Model | Context Window | Approx Words |
|-------|---------------|--------------|
| GPT-3 (2020) | 4,096 tokens | ~3,000 words |
| GPT-3.5 | 4,096 tokens | ~3,000 words |
| GPT-3.5-16k | 16,384 tokens | ~12,000 words |
| GPT-4 | 8,192 tokens | ~6,000 words |
| GPT-4-32k | 32,768 tokens | ~24,000 words |
| Claude 2 | 100,000 tokens | ~75,000 words |

---

## How GPT-3 Generates Text

### Autoregressive Generation

GPT-3 predicts one token at a time:

```
User Prompt: "Write a poem about"

Step 1:
  Input: "Write a poem about"
  Predict next token: "artificial" (highest probability)
  
Step 2:
  Input: "Write a poem about artificial"
  Predict next token: "intelligence"
  
Step 3:
  Input: "Write a poem about artificial intelligence"
  Predict next token: "In"
  
Step 4:
  Input: "Write a poem about artificial intelligence In"
  Predict next token: "circuits"

... continues until complete sentence or max length
```

### Probability Distribution

```
After processing input, output layer produces:

Probabilities for all 50,257 tokens:
  "artificial" → 0.35 (35%)
  "robots"     → 0.20 (20%)
  "nature"     → 0.15 (15%)
  "love"       → 0.10 (10%)
  "the"        → 0.05 (5%)
  ... (50,252 other tokens with lower probabilities)

GPT-3 can:
  - Pick highest (deterministic)
  - Sample from distribution (creative)
  - Use temperature to control randomness
```

---

## Attention Mechanism

### What is Attention?

Attention allows GPT-3 to focus on relevant parts of the input:

```
Input: "The cat sat on the mat because it was tired"

When processing "it":
  Attention weights:
    "The"     → 0.01
    "cat"     → 0.85 ← High attention!
    "sat"     → 0.02
    "on"      → 0.01
    "the"     → 0.01
    "mat"     → 0.05
    "because" → 0.02
    "was"     → 0.03

GPT-3 understands "it" refers to "cat"!
```

### Multi-Head Attention

```
GPT-3 uses 96 attention heads per layer:

Each head learns different patterns:
  Head 1: Syntax (grammar structure)
  Head 2: Semantics (word meaning)
  Head 3: Coreference (what "it" refers to)
  Head 4: Long-range dependencies
  ... (92 more heads)

96 heads × 96 layers = 9,216 different attention patterns!
```

---

## Comparing Scales

### From Perceptron to GPT-3

```
Single Perceptron (Lesson 2):
  Inputs: 2
  Neurons: 1
  Parameters: 3
  Computation: ~10 operations

Small Network (Lesson 5):
  Architecture: 2 → 4 → 3 → 1
  Parameters: 31
  Computation: ~100 operations

MNIST Network:
  Architecture: 784 → 128 → 64 → 10
  Parameters: 109,386
  Computation: ~200,000 operations

GPT-3:
  Architecture: 96 transformer layers
  Parameters: 175,000,000,000
  Computation: ~350,000,000,000 operations per token!

Scale increase:
  Perceptron → GPT-3:  58,000,000,000x more parameters!
```

---

## Storage and Computation

### Storage Requirements

```
Each parameter: 4 bytes (32-bit float)

GPT-3 storage:
  175 billion parameters × 4 bytes
  = 700 GB

For comparison:
  - Single perceptron: 12 bytes
  - MNIST network: ~437 KB
  - BERT-Base: ~440 MB
  - GPT-3: ~700 GB

Your computer's hard drive: typically 256GB-1TB
GPT-3 takes most/all of it!
```

### Computational Cost

```
Training GPT-3:
  - GPUs: ~10,000 NVIDIA V100s
  - Time: ~2-3 months
  - Cost: $4-12 million
  - Energy: ~1,287 MWh

Running GPT-3 (inference):
  - Requires: Multiple high-end GPUs
  - Cost per query: ~$0.01-0.10
  - Response time: 1-5 seconds
```

---

## Key Architectural Innovations

### Transformer Architecture

```
Key components:
  1. Self-attention (learns relationships)
  2. Positional encoding (understands order)
  3. Layer normalization (stable training)
  4. Residual connections (enables depth)

These innovations enabled:
  ✓ Processing long sequences
  ✓ Parallel computation
  ✓ Scalability to 96+ layers
```

### Why Transformers Over RNNs?

```
Previous architectures (RNNs):
  ✗ Sequential processing (slow)
  ✗ Limited memory of past
  ✗ Vanishing gradients

Transformers:
  ✓ Parallel processing (fast)
  ✓ Full attention to all tokens
  ✓ Better gradient flow
```

---

## Input Examples

### Example 1: Simple Question

```
Input: "What is the capital of France?"

Tokenization:
  ["What", " is", " the", " capital", " of", " France", "?"]
  = 7 tokens

Embedding:
  7 × 12,288 = 86,016 numbers

Processing:
  Through 96 layers of transformers
  Using 175 billion parameters

Output:
  "The capital of France is Paris."
```

### Example 2: Code Generation

```
Input: "Write a Python function to calculate factorial"

Tokenization:
  ["Write", " a", " Python", " function", " to", 
   " calculate", " factorial"]
  = 7 tokens

Output (generated token by token):
  "def factorial(n):
      if n == 0:
          return 1
      return n * factorial(n-1)"
```

### Example 3: Long Context

```
Input: [A 3,000-word essay]

Tokenization:
  ~4,000 tokens (near limit!)

Processing:
  All 4,000 tokens processed together
  Attention mechanism relates all tokens
  
Output:
  Summary or response based on full context
```

---

## Limitations

### What GPT-3 Cannot Do

```
1. Cannot process inputs > 4,096 tokens
   Solution: Break into chunks

2. No real-time information (trained on old data)
   Solution: Provide context in prompt

3. No image understanding (text-only)
   Solution: Use GPT-4 (multimodal)

4. Can generate incorrect information confidently
   Solution: Verify important facts

5. No memory between conversations
   Solution: Include conversation history in prompt
```

---

## Key Takeaways

**Architecture:**
- 96 transformer layers
- 175 billion parameters (mostly in layers)
- 12,288-dimensional embeddings
- 50,257 token vocabulary

**Input Processing:**
- Text → Tokens → Embeddings → Layers → Output
- Maximum 4,096 tokens per request
- One token at a time generation

**Scale:**
- 700 GB to store model
- Billions of operations per prediction
- Millions of dollars to train
- 58 billion times larger than a perceptron

**Key Innovation:**
- Transformer architecture with self-attention
- Parallel processing of sequences
- Hierarchical feature learning across 96 layers

---

## Next Steps

**Continue Learning:**  
Proceed to **[Lesson 7: How GPT-3 Was Trained](07_gpt3_training.md)** to understand how this massive network learned from internet data.

**Deepen Understanding:**
- Study transformer architecture in detail
- Learn about attention mechanisms
- Explore tokenization algorithms (BPE, WordPiece)

**Explore Further:**
- Compare GPT-3 with other models (BERT, T5, PaLM)
- Research recent developments (GPT-4, Claude, LLaMA)
- Try using GPT-3 API to see it in action

