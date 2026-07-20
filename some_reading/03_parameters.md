# Lesson 3: Understanding Parameters

**Learning Objectives:**
- Understand what parameters are in neural networks
- Distinguish between parameters and inputs
- Learn how to count parameters
- Understand parameter storage and scale

---

## What Are Parameters?

**Parameters** are the learnable values inside a neural network that get adjusted during training. They represent the network's "knowledge" learned from data.

### Parameters = Weights + Biases

```
All Parameters in a Neural Network:
├─ Weights (connections between neurons)
└─ Biases (one per neuron)

NOT Parameters:
├─ Inputs (data fed into the network)
├─ Outputs (predictions from the network)
└─ Hyperparameters (learning rate, epochs, etc.)
```

---

## Parameters vs Inputs

A common source of confusion for beginners:

### Clear Distinction

| Aspect | Parameters | Inputs |
|--------|-----------|--------|
| **What they are** | Weights and biases | Data to process |
| **Where stored** | Inside the model | Provided at runtime |
| **When created** | At initialization | Each prediction |
| **During training** | Adjusted/learned | Examples for learning |
| **After training** | Fixed values | Still variable |
| **Count** | Fixed per architecture | Varies per use |

### Example: 2-Input Perceptron

```python
class Perceptron:
    def __init__(self, input_size):
        # PARAMETERS - stored in model
        self.weights = [random(), random()]  # 2 parameters
        self.bias = random()                  # 1 parameter
        # Total: 3 parameters
    
    def predict(self, inputs):
        # INPUTS - provided at runtime
        # inputs = [0.5, 0.8]  ← temporary data
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
        return 1 if weighted_sum + self.bias >= 0 else 0
```

**Usage:**
```python
perceptron = Perceptron(input_size=2)
# Created 3 parameters: [w₁, w₂, b]

result = perceptron.predict([0.5, 0.8])
# Inputs: [0.5, 0.8] ← NOT parameters
# Parameters: [w₁, w₂, b] ← stored in model
```

---

## Counting Parameters

### Formula for Single Layer

```
Parameters = (num_inputs × num_neurons) + num_neurons
           = (weights)                  + (biases)
```

### Examples

**Example 1: Single Perceptron (2 inputs)**
```
Inputs: 2
Neurons: 1

Weights: 2 × 1 = 2
Biases: 1
Total: 3 parameters
```

**Example 2: Single Layer (3 inputs → 4 neurons)**
```
Inputs: 3
Neurons: 4

Weights: 3 × 4 = 12
Biases: 4
Total: 16 parameters
```

**Example 3: Two Layers (2 → 3 → 1)**
```
Layer 1: 2 inputs → 3 neurons
  Weights: 2 × 3 = 6
  Biases: 3
  Subtotal: 9 parameters

Layer 2: 3 inputs → 1 neuron
  Weights: 3 × 1 = 3
  Biases: 1
  Subtotal: 4 parameters

Total: 13 parameters
```

---

## Multi-Layer Network Parameters

### Example Network

```
Architecture: 1000 → 500 → 300 → 10

┌─────────────────────────────────────────┐
│ Input Layer: 1000 features              │
│ (not counted as parameters)             │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Hidden Layer 1: 500 neurons             │
│ Weights: 1000 × 500 = 500,000          │
│ Biases: 500                             │
│ Subtotal: 500,500 parameters            │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Hidden Layer 2: 300 neurons             │
│ Weights: 500 × 300 = 150,000           │
│ Biases: 300                             │
│ Subtotal: 150,300 parameters            │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Output Layer: 10 neurons                │
│ Weights: 300 × 10 = 3,000              │
│ Biases: 10                              │
│ Subtotal: 3,010 parameters              │
└─────────────────────────────────────────┘

═══════════════════════════════════════════
TOTAL PARAMETERS: 653,810
═══════════════════════════════════════════

Note: Input size (1000) is NOT counted!
```

---

## Parameter Lifecycle

### 1. Initialization

Parameters start with random values:

```python
# Before training
perceptron = Perceptron(input_size=2)
# weights = [0.234, -0.567]  ← random
# bias = 0.123                ← random
```

### 2. Training

Parameters are adjusted based on errors:

```python
# During training
for epoch in range(100):
    for inputs, target in training_data:
        prediction = perceptron.predict(inputs)
        error = target - prediction
        # Update parameters based on error
        perceptron.weights[0] += learning_rate * error * inputs[0]
        perceptron.weights[1] += learning_rate * error * inputs[1]
        perceptron.bias += learning_rate * error
```

### 3. Convergence

Parameters settle at values that minimize errors:

```python
# After training
# weights = [0.847, 0.923]  ← learned
# bias = -0.412              ← learned
```

### 4. Deployment

Parameters remain fixed during inference:

```python
# Using trained model
result1 = perceptron.predict([0.0, 1.0])  # Input varies
result2 = perceptron.predict([1.0, 0.0])  # Input varies
# Parameters stay the same: [0.847, 0.923, -0.412]
```

---

## Storing Parameters

### Model Persistence

When saving a trained model, you save all parameter values:

```python
# Saving a model
model.save("trained_model.pkl")
# Saves: all weights and biases

# Loading a model
model = load("trained_model.pkl")
# Loads: all learned parameter values
```

### Storage Size

Parameters are typically stored as 32-bit floating-point numbers (4 bytes each):

| Model | Parameters | Storage Size |
|-------|-----------|--------------|
| Single perceptron | 3 | 12 bytes |
| Small network | 10,000 | ~40 KB |
| Medium network | 1 million | ~4 MB |
| ResNet-50 | 25 million | ~100 MB |
| BERT-Base | 110 million | ~440 MB |
| GPT-2 | 1.5 billion | ~6 GB |
| GPT-3 | 175 billion | ~700 GB |

**Formula:** Storage (bytes) = Parameters × 4

---

## Why Parameter Count Matters

### 1. Model Capacity

More parameters = ability to learn more complex patterns

```
Simple task (OR gate):
  3 parameters sufficient

Medium task (digit recognition):
  ~100,000 parameters needed

Complex task (language understanding):
  billions of parameters required
```

### 2. Computational Cost

More parameters = more computations

```
Forward pass: Calculate output for all parameters
Backward pass: Update all parameters
Inference: Process through all parameters

10x parameters ≈ 10x computation time
```

### 3. Memory Requirements

```
Training Memory Needs:
├─ Parameters themselves
├─ Gradients (same size as parameters)
├─ Optimizer state (often 2x parameters)
└─ Activations (depends on batch size)

Total training memory ≈ 4-5x parameter storage

GPT-3 example:
  Parameters: 700 GB
  Training memory: ~3-4 TB
```

### 4. Overfitting Risk

```
Too few parameters:
  └─ Underfitting: Can't learn the pattern

Right amount:
  └─ Good generalization

Too many parameters:
  └─ Overfitting: Memorizes training data
     (doesn't generalize to new data)
```

---

## How Large Models Reach Billions of Parameters

### GPT-3 Breakdown (Simplified)

```
┌─────────────────────────────────────────┐
│ Embedding Layer                         │
│ Vocabulary: 50,257 tokens               │
│ Dimensions: 12,288                      │
│ Parameters: 50,257 × 12,288             │
│ = 617 million                           │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Transformer Layer 1                     │
│ - Self-attention mechanism              │
│ - Feedforward networks                  │
│ - Layer normalization                   │
│ Parameters: ~1.8 billion                │
└─────────────────────────────────────────┘
              ↓
        (94 more layers)
              ↓
┌─────────────────────────────────────────┐
│ Transformer Layer 96                    │
│ Parameters: ~1.8 billion                │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Output Layer                            │
│ Parameters: ~617 million                │
└─────────────────────────────────────────┘

═══════════════════════════════════════════
CALCULATION:
  Embedding: 0.6B
  96 layers × 1.8B = 172.8B
  Output: 0.6B
  TOTAL: ~175 billion parameters
═══════════════════════════════════════════
```

---

## Parameter Counting Code

### Python Function

```python
def count_parameters(architecture):
    """
    Count total parameters in a neural network.
    
    Args:
        architecture: List of layer sizes
        Example: [784, 128, 64, 10]
    
    Returns:
        Total parameter count
    """
    total = 0
    
    for i in range(len(architecture) - 1):
        inputs = architecture[i]
        outputs = architecture[i + 1]
        
        # Weights + biases for this layer
        layer_params = (inputs * outputs) + outputs
        total += layer_params
        
        print(f"Layer {i+1}: {inputs} → {outputs}")
        print(f"  Weights: {inputs} × {outputs} = {inputs * outputs}")
        print(f"  Biases: {outputs}")
        print(f"  Subtotal: {layer_params}")
    
    return total

# Example usage
architecture = [784, 128, 64, 10]
total = count_parameters(architecture)
print(f"\nTotal parameters: {total:,}")

# Output:
# Layer 1: 784 → 128
#   Weights: 784 × 128 = 100,352
#   Biases: 128
#   Subtotal: 100,480
# Layer 2: 128 → 64
#   Weights: 128 × 64 = 8,192
#   Biases: 64
#   Subtotal: 8,256
# Layer 3: 64 → 10
#   Weights: 64 × 10 = 640
#   Biases: 10
#   Subtotal: 650
# 
# Total parameters: 109,386
```

---

## Common Misconceptions

### Misconception 1: "Parameters are the inputs"
**Reality:** Parameters are learned weights/biases. Inputs are data fed into the network.

### Misconception 2: "More parameters always means better performance"
**Reality:** Too many parameters can cause overfitting. Need appropriate size for task and data.

### Misconception 3: "Parameters are created during inference"
**Reality:** Parameters are initialized randomly, then learned during training, then fixed during inference.

### Misconception 4: "GPT-3's 175B parameters are input values"
**Reality:** 175B is the count of weights and biases. Inputs are the text prompts (much smaller).

---

## Real-World Model Sizes

### Historical Progression

| Year | Model | Parameters | Application |
|------|-------|-----------|-------------|
| 1958 | Perceptron | ~10 | Logic gates |
| 1998 | LeNet-5 | 60K | Handwriting |
| 2012 | AlexNet | 60M | Image classification |
| 2014 | VGG-16 | 138M | Image classification |
| 2018 | BERT-Base | 110M | Language understanding |
| 2019 | GPT-2 | 1.5B | Text generation |
| 2020 | GPT-3 | 175B | Language modeling |
| 2023 | GPT-4 | ~1.7T (estimated) | Multimodal AI |

### Scale Comparison

```
Perceptron:        3 parameters
                   ●

Small Network:     100K parameters
                   ████

BERT:              110M parameters
                   ████████████████████████

GPT-2:             1.5B parameters
                   ████████████████████████████████████

GPT-3:             175B parameters
                   ████████████████████████████████████████████████
                   ████████████████████████████████████████████████
                   (and much more...)
```

---

## Key Takeaways

**Core Concepts:**
- Parameters = weights + biases (the model's learned knowledge)
- Inputs = data fed into the model (varies each use)
- Parameters are learned during training, then fixed for inference
- Parameter count indicates model capacity and computational cost

**Counting Parameters:**
- Per layer: (inputs × neurons) + neurons
- Total network: sum across all layers
- Input layer size not counted as parameters

**Practical Implications:**
- More parameters ≠ always better
- Storage scales linearly with parameter count
- Computation scales with parameter count
- Memory needs exceed parameter storage during training

---

## Next Steps

**Continue Learning:**  
Proceed to **[Lesson 4: Understanding Bias](04_bias.md)** to learn about the specific role of bias in neural networks.

**Deepen Understanding:**
- Calculate parameters for different network architectures
- Explore parameter initialization strategies
- Research the relationship between parameters and overfitting

**Hands-On:**
- Modify the perceptron code to track parameter values
- Visualize how parameters change during training
- Experiment with networks of different sizes

