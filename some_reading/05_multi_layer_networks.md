# Lesson 5: Multi-Layer Neural Networks

**Learning Objectives:**
- Understand how layers stack to create deep networks
- Learn why depth enables complex pattern recognition
- See how parameters distribute across layers
- Understand the concept of hierarchical feature learning

---

## From Single to Multi-Layer

### The Limitation of Single Neurons

A single perceptron can only learn **linearly separable** patterns:

```
Can Learn (Linear):        Cannot Learn (Non-Linear):
    OR Gate                    XOR Gate
    AND Gate                   Complex boundaries
    Simple classification      Image recognition
```

**Solution:** Stack multiple neurons in layers!

---

## Multi-Layer Network Architecture

### Basic Structure

```
┌─────────────────┐
│  Input Layer    │  ← Receives data (not counted as a layer)
└─────────────────┘
        ↓
┌─────────────────┐
│  Hidden Layer 1 │  ← First processing layer
└─────────────────┘
        ↓
┌─────────────────┐
│  Hidden Layer 2 │  ← Second processing layer
└─────────────────┘
        ↓
┌─────────────────┐
│  Output Layer   │  ← Final predictions
└─────────────────┘
```

### Layer Types

| Layer Type | Position | Purpose | Parameters |
|------------|----------|---------|------------|
| **Input** | First | Receives data | None (just passes data) |
| **Hidden** | Middle | Processes features | Weights + biases |
| **Output** | Last | Produces predictions | Weights + biases |

---

## Example: 3-Layer Network

### Architecture: 3 → 4 → 3 → 1

```
INPUT LAYER: 3 features
    (no parameters - just input data)
              ↓
HIDDEN LAYER 1: 4 neurons
    Each neuron: 3 weights + 1 bias = 4 params
    Total: 4 × 4 = 16 parameters
              ↓
HIDDEN LAYER 2: 3 neurons
    Each neuron: 4 weights + 1 bias = 5 params
    Total: 3 × 5 = 15 parameters
              ↓
OUTPUT LAYER: 1 neuron
    Each neuron: 3 weights + 1 bias = 4 params
    Total: 1 × 4 = 4 parameters

═════════════════════════════════════
TOTAL PARAMETERS: 16 + 15 + 4 = 35
═════════════════════════════════════
```

### Detailed Breakdown

```
Layer 1: 3 inputs → 4 neurons
┌──────────────────────────────┐
│ Neuron 1: 3w + 1b = 4 params │
│ Neuron 2: 3w + 1b = 4 params │
│ Neuron 3: 3w + 1b = 4 params │
│ Neuron 4: 3w + 1b = 4 params │
│ Subtotal: 16 parameters      │
└──────────────────────────────┘

Layer 2: 4 inputs → 3 neurons
┌──────────────────────────────┐
│ Neuron 1: 4w + 1b = 5 params │
│ Neuron 2: 4w + 1b = 5 params │
│ Neuron 3: 4w + 1b = 5 params │
│ Subtotal: 15 parameters      │
└──────────────────────────────┘

Layer 3: 3 inputs → 1 neuron
┌──────────────────────────────┐
│ Neuron 1: 3w + 1b = 4 params │
│ Subtotal: 4 parameters       │
└──────────────────────────────┘
```

---

## How Information Flows

### Forward Pass (Making Predictions)

```
Step 1: Input data
    x = [1.0, 0.5, 0.8]

Step 2: Hidden Layer 1 processes inputs
    neuron₁ = activation(1.0×w₁₁ + 0.5×w₁₂ + 0.8×w₁₃ + b₁)
    neuron₂ = activation(1.0×w₂₁ + 0.5×w₂₂ + 0.8×w₂₃ + b₂)
    neuron₃ = activation(1.0×w₃₁ + 0.5×w₃₂ + 0.8×w₃₃ + b₃)
    neuron₄ = activation(1.0×w₄₁ + 0.5×w₄₂ + 0.8×w₄₃ + b₄)
    
    Output: [0.7, 0.9, 0.2, 0.8]

Step 3: Hidden Layer 2 processes Layer 1 outputs
    neuron₁ = activation(0.7×w₁₁ + 0.9×w₁₂ + 0.2×w₁₃ + 0.8×w₁₄ + b₁)
    neuron₂ = activation(0.7×w₂₁ + 0.9×w₂₂ + 0.2×w₂₃ + 0.8×w₂₄ + b₂)
    neuron₃ = activation(0.7×w₃₁ + 0.9×w₃₂ + 0.2×w₃₃ + 0.8×w₃₄ + b₃)
    
    Output: [0.6, 0.8, 0.3]

Step 4: Output Layer produces final prediction
    output = activation(0.6×w₁ + 0.8×w₂ + 0.3×w₃ + b)
    
    Final: 1 (or 0)
```

---

## Why Multiple Layers Work

### Hierarchical Feature Learning

Each layer learns increasingly abstract features:

```
Image Recognition Example:

Input: Raw pixels (28×28 = 784 values)
    ↓
Layer 1: Detects edges and lines
    - Horizontal edges
    - Vertical edges
    - Diagonal lines
    (~100,000 parameters)
    ↓
Layer 2: Detects shapes
    - Circles
    - Curves
    - Corners
    (~50,000 parameters)
    ↓
Layer 3: Detects parts
    - Eyes
    - Noses
    - Ears
    (~25,000 parameters)
    ↓
Output: Identifies objects
    - Cat
    - Dog
    - Person
    (~10,000 parameters)
```

### Non-Linear Decision Boundaries

```
Single Layer:              Multi-Layer:
Can only learn:           Can learn:

  x₂                        x₂
   |                         |
   |  ╱ straight line         |  ╱╲  complex curves
   | ╱                        | ╱  ╲ and boundaries
   |╱______ x₁                |╱____╲_ x₁
   0                          0

Linear separation          Any shape possible!
```

---

## Parameter Distribution

### Small Network Example

```
Architecture: 2 → 4 → 3 → 1

Layer 1: 2 → 4
  Weights: 2 × 4 = 8
  Biases: 4
  Subtotal: 12 parameters (34%)

Layer 2: 4 → 3
  Weights: 4 × 3 = 12
  Biases: 3
  Subtotal: 15 parameters (43%)

Layer 3: 3 → 1
  Weights: 3 × 1 = 3
  Biases: 1
  Subtotal: 4 parameters (11%)

Total: 31 parameters

Distribution:
Layer 1: ████████████ (34%)
Layer 2: █████████████████ (43%)
Layer 3: ████ (11%)
```

### Large Network Example

```
Architecture: 784 → 128 → 64 → 10
(MNIST digit classification)

Layer 1: 784 → 128
  Parameters: (784 × 128) + 128 = 100,480 (92%)

Layer 2: 128 → 64
  Parameters: (128 × 64) + 64 = 8,256 (7.5%)

Layer 3: 64 → 10
  Parameters: (64 × 10) + 10 = 650 (0.6%)

Total: 109,386 parameters

Distribution:
Layer 1: ████████████████████████████████████ (92%)
Layer 2: ███ (7.5%)
Layer 3: ▌ (0.6%)

First layer dominates parameter count!
```

---

## Network Depth vs Width

### Deep Networks (Many Layers)

```
Example: 100 → 100 → 100 → 100 → 10
(5 layers, narrow)

Advantages:
  ✓ Learns hierarchical features
  ✓ More expressive with fewer parameters
  ✓ Better generalization

Challenges:
  ✗ Harder to train (vanishing gradients)
  ✗ More complex architecture
```

### Wide Networks (Few Layers)

```
Example: 100 → 1000 → 10
(2 layers, wide)

Advantages:
  ✓ Easier to train
  ✓ Simpler architecture

Challenges:
  ✗ More parameters needed
  ✗ Less feature hierarchy
  ✗ May overfit
```

---

## Real-World Network Sizes

### Historical Progression

| Network | Year | Layers | Parameters | Task |
|---------|------|--------|-----------|------|
| Perceptron | 1958 | 1 | ~10 | Logic gates |
| LeNet-5 | 1998 | 7 | 60K | Handwriting |
| AlexNet | 2012 | 8 | 60M | Image classification |
| VGG-16 | 2014 | 16 | 138M | Image classification |
| ResNet-152 | 2015 | 152 | 60M | Image classification |
| BERT-Base | 2018 | 12 | 110M | Language |
| GPT-2 | 2019 | 48 | 1.5B | Language |
| GPT-3 | 2020 | 96 | 175B | Language |

### Trend: Deeper and Larger

```
1950s-1980s:  Single layer networks
1990s-2000s:  3-10 layers
2010s:        10-100 layers
2020s:        100+ layers with billions of parameters
```

---

## GPT-3 Architecture

### Layer-by-Layer Structure

```
GPT-3: 96 Transformer Layers

┌────────────────────────────────────┐
│ Token Embedding                    │
│ Vocab: 50,257                      │
│ Dimensions: 12,288                 │
│ Parameters: ~617 million           │
└────────────────────────────────────┘
          ↓
┌────────────────────────────────────┐
│ Transformer Layer 1                │
│ - Multi-head attention             │
│ - Feedforward network              │
│ - Layer normalization              │
│ Parameters: ~1.8 billion           │
└────────────────────────────────────┘
          ↓
┌────────────────────────────────────┐
│ Transformer Layer 2                │
│ Parameters: ~1.8 billion           │
└────────────────────────────────────┘
          ↓
    ... (94 more layers)
          ↓
┌────────────────────────────────────┐
│ Transformer Layer 96               │
│ Parameters: ~1.8 billion           │
└────────────────────────────────────┘
          ↓
┌────────────────────────────────────┐
│ Output Layer                       │
│ Parameters: ~617 million           │
└────────────────────────────────────┘

═══════════════════════════════════
Total: ~175 billion parameters

Distribution:
  Embedding: 0.35%
  Layers 1-96: 99.3%
  Output: 0.35%
═══════════════════════════════════
```

---

## Comparing Network Scales

### Visual Scale Comparison

```
Single Perceptron:
  3 parameters
  ●

Small Network (2→4→3→1):
  31 parameters
  ████

MNIST Network (784→128→64→10):
  109,386 parameters
  ████████████████████████████

ResNet-50:
  25 million parameters
  ████████████████████████████████████████████████

BERT-Base:
  110 million parameters
  ████████████████████████████████████████████████
  ████████████████████████████████████████████████
  ████████████████████████████████████████████████

GPT-3:
  175 billion parameters
  [Scale too large to visualize meaningfully]
  ~1,500,000x larger than single perceptron
```

---

## Training Multi-Layer Networks

### Backpropagation

Training multi-layer networks requires **backpropagation**:

```
Forward Pass:
  Input → Layer1 → Layer2 → Layer3 → Output → Error

Backward Pass (adjusting parameters):
  Error → Layer3 ← Layer2 ← Layer1
  
Each layer's parameters are adjusted based on:
  1. Its contribution to the error
  2. Learning rate
  3. Gradient of the loss function
```

### Training Challenges

| Challenge | Description | Solution |
|-----------|-------------|----------|
| **Vanishing gradients** | Deep layers receive tiny updates | ReLU activation, skip connections |
| **Overfitting** | Network memorizes training data | Dropout, regularization |
| **Slow convergence** | Takes long to train | Better optimizers (Adam, RMSprop) |
| **Local minima** | Gets stuck in suboptimal solutions | Momentum, learning rate schedules |

---

## Network Design Principles

### Rule of Thumb

```
Input size > Hidden layer size > Output size

Example (good):
  784 → 128 → 64 → 10
  (funnel shape)

Example (acceptable):
  784 → 256 → 256 → 10
  (wide middle)

Example (unusual):
  784 → 32 → 256 → 10
  (bottleneck then expand)
```

### Choosing Architecture

| Task Complexity | Suggested Architecture |
|----------------|------------------------|
| Very simple (OR gate) | 1 layer, few neurons |
| Simple (digit recognition) | 2-3 layers, 100-500 neurons |
| Medium (image classification) | 10-50 layers, millions of params |
| Complex (language understanding) | 50-100+ layers, billions of params |

---

## Activation Functions in Multi-Layer Networks

### Why Non-Linear Activations Matter

```
Without non-linear activations:
  Multiple layers collapse to single layer
  No benefit from depth!

With non-linear activations:
  Each layer adds expressive power
  Depth enables complex patterns
```

### Common Activations

```
ReLU (most popular):
  f(x) = max(0, x)
  Fast, effective for deep networks

Sigmoid:
  f(x) = 1 / (1 + e^(-x))
  Output between 0 and 1
  Used in output layers

Tanh:
  f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
  Output between -1 and 1
  Centers data around 0
```

---

## Practical Example: MNIST

### Network Architecture

```
Problem: Classify handwritten digits (0-9)

Architecture: 784 → 128 → 64 → 10

Input: 28×28 pixel image = 784 values
    ↓
Hidden 1: 128 neurons
  - Learns edge detectors
  - Parameters: 100,480
    ↓
Hidden 2: 64 neurons
  - Learns digit parts
  - Parameters: 8,256
    ↓
Output: 10 neurons (one per digit)
  - Outputs probabilities
  - Parameters: 650

Total: 109,386 parameters
Accuracy: ~98% on test data
```

---

## Key Takeaways

**Multi-Layer Advantages:**
- ✓ Learn non-linear patterns
- ✓ Hierarchical feature extraction
- ✓ Can approximate any function (universal approximators)
- ✓ Handle complex real-world tasks

**Design Considerations:**
- More layers = more expressiveness (but harder to train)
- More neurons = more capacity (but more prone to overfitting)
- First layers typically have most parameters
- Balance between depth and width

**Parameter Distribution:**
- Parameters spread across all layers
- Each layer contributes to total count
- Early layers often dominate parameter count
- GPT-3: 175B parameters across 96 layers

---

## Common Questions

**Q: How many layers should I use?**  
A: Start with 2-3 hidden layers. Add more if underfitting.

**Q: How many neurons per layer?**  
A: Start with similar size to input, gradually decrease toward output.

**Q: Why does GPT-3 need 96 layers?**  
A: Language is extremely complex. Each layer learns different linguistic patterns.

**Q: Can a single layer do everything?**  
A: Theoretically yes (with infinite neurons), practically no.

---

## Next Steps

**Continue Learning:**  
Proceed to **[Lesson 6: How GPT-3 Works - Architecture](06_gpt3_architecture.md)** to see how these concepts scale to modern AI systems.

**Deepen Understanding:**
- Study backpropagation algorithm
- Learn about different network architectures (CNN, RNN, Transformer)
- Explore activation functions in detail

**Hands-On:**
- Implement a 2-layer network from scratch
- Experiment with different architectures on MNIST
- Visualize what each layer learns

