# Lesson 2: The Perceptron - Your First Neural Network

**Learning Objectives:**
- Understand the perceptron architecture
- Learn how a perceptron makes predictions
- See how training adjusts weights and biases
- Build intuition through examples

---

## What is a Perceptron?

The **perceptron** is the simplest neural network - a single artificial neuron. Invented by Frank Rosenblatt in 1958, it's the foundation for understanding modern deep learning.

### Basic Idea

A perceptron:
- Takes multiple inputs (numbers)
- Multiplies each by a weight
- Sums everything with a bias
- Outputs 0 or 1 based on a threshold

---

## Perceptron Architecture

### Visual Representation

```
        Inputs                    Output
          ↓                         ↓
    
    x₁ ────×w₁────┐
                   │
    x₂ ────×w₂────┼──→ Σ + b ──→ Activation ──→ y
                   │    (sum)     (threshold)
    x₃ ────×w₃────┘
    
    Bias ─────────┘
```

### Components

| Component | Symbol | Description | Example |
|-----------|--------|-------------|---------|
| **Inputs** | x₁, x₂, x₃ | Values fed into neuron | [0.5, 1.0, 0.3] |
| **Weights** | w₁, w₂, w₃ | Learned parameters | [0.8, 0.6, -0.2] |
| **Bias** | b | Threshold adjustment | -0.5 |
| **Weighted Sum** | z | x₁w₁ + x₂w₂ + x₃w₃ + b | 0.9 |
| **Activation** | f(z) | Threshold function | 1 if z ≥ 0 else 0 |
| **Output** | y | Final prediction | 1 |

---

## How It Works: Step-by-Step

### Example: 2-Input Perceptron

Given:
- Inputs: x₁ = 0.5, x₂ = 1.0
- Weights: w₁ = 0.8, w₂ = 0.6
- Bias: b = -0.3

**Step 1: Calculate Weighted Sum**
```
z = (x₁ × w₁) + (x₂ × w₂) + b
z = (0.5 × 0.8) + (1.0 × 0.6) + (-0.3)
z = 0.4 + 0.6 - 0.3
z = 0.7
```

**Step 2: Apply Activation Function**
```
f(z) = 1 if z ≥ 0 else 0
f(0.7) = 1  (because 0.7 ≥ 0)
```

**Step 3: Output**
```
y = 1
```

---

## Activation Functions

### Step Function (Binary Threshold)

```
Output
  1 |         ┌─────────
    |         │
    |         │
  0 |─────────┘
    └─────────┼─────── Input
              0

f(z) = {  1  if z ≥ 0
       {  0  if z < 0
```

Most common for basic perceptrons. Produces binary output.

### Other Activation Functions (Preview)

**Sigmoid:** Smooth curve (0 to 1)
**ReLU:** Max(0, z) - popular in deep learning
**Tanh:** Smooth curve (-1 to 1)

For this lesson, we focus on the step function.

---

## Understanding Weights

Weights control how much each input influences the output.

### Weight Effects

```
Example: OR gate approximation
w₁ = 1.0, w₂ = 1.0, b = -0.5

Scenario 1: x₁ = 0, x₂ = 0
z = (0 × 1.0) + (0 × 1.0) + (-0.5) = -0.5 → Output: 0

Scenario 2: x₁ = 0, x₂ = 1
z = (0 × 1.0) + (1 × 1.0) + (-0.5) = 0.5 → Output: 1

Scenario 3: x₁ = 1, x₂ = 0
z = (1 × 1.0) + (0 × 1.0) + (-0.5) = 0.5 → Output: 1

Scenario 4: x₁ = 1, x₂ = 1
z = (1 × 1.0) + (1 × 1.0) + (-0.5) = 1.5 → Output: 1
```

This perceptron implements an OR gate!

### Weight Interpretation

| Weight Value | Meaning | Effect |
|--------------|---------|--------|
| w > 0 | Positive influence | Higher input → more likely to fire |
| w < 0 | Negative influence | Higher input → less likely to fire |
| w ≈ 0 | No influence | Input ignored |
| \|w\| large | Strong influence | Input very important |
| \|w\| small | Weak influence | Input less important |

---

## Understanding Bias

Bias shifts the decision boundary without depending on inputs.

### Bias as Threshold

```
Without bias:          With bias b = -0.5:
z = x₁w₁ + x₂w₂       z = x₁w₁ + x₂w₂ - 0.5
Fires when z ≥ 0      Fires when z ≥ 0
                      Equivalent to x₁w₁ + x₂w₂ ≥ 0.5
```

### Geometric View (2D Input Space)

```
Decision Boundary: Line where perceptron switches from 0 to 1

No Bias (b = 0):          With Bias (b = -0.5):
  x₂                        x₂
   |                         |
   |  ╱ boundary             |    ╱ boundary
   | ╱                       |   ╱ (shifted)
   |╱______ x₁               |  ╱_______ x₁
   0                         | ╱
                            |╱
                            0

Bias shifts the line away from origin
```

---

## Learning Algorithm (Training)

The perceptron learns by adjusting weights and bias when it makes mistakes.

### Perceptron Learning Rule

```
For each training example:
  1. Make prediction with current weights
  2. Calculate error = expected - predicted
  3. Update each weight: wᵢ = wᵢ + η × error × xᵢ
  4. Update bias: b = b + η × error
  
Where η (eta) = learning rate (typically 0.1)
```

### Training Example: Learning OR Gate

**Initial State:**
- Weights: w₁ = 0.2, w₂ = -0.3 (random)
- Bias: b = 0.1 (random)
- Learning rate: η = 0.1

**Training Data:**
```
[0, 0] → 0  (0 OR 0 = 0)
[0, 1] → 1  (0 OR 1 = 1)
[1, 0] → 1  (1 OR 0 = 1)
[1, 1] → 1  (1 OR 1 = 1)
```

**Epoch 1, Example 1: [0, 0] → 0**
```
Prediction: z = 0×0.2 + 0×(-0.3) + 0.1 = 0.1 → y = 1
Error: 0 - 1 = -1
Updates:
  w₁ = 0.2 + 0.1×(-1)×0 = 0.2
  w₂ = -0.3 + 0.1×(-1)×0 = -0.3
  b = 0.1 + 0.1×(-1) = 0.0
```

**Epoch 1, Example 2: [0, 1] → 1**
```
Prediction: z = 0×0.2 + 1×(-0.3) + 0.0 = -0.3 → y = 0
Error: 1 - 0 = 1
Updates:
  w₁ = 0.2 + 0.1×1×0 = 0.2
  w₂ = -0.3 + 0.1×1×1 = -0.2
  b = 0.0 + 0.1×1 = 0.1
```

**Continue for all examples...**

After several epochs, weights converge to values that correctly classify all examples.

---

## What Can a Perceptron Learn?

### Linearly Separable Problems ✓

Problems where a straight line can separate classes:

**OR Gate:**
```
  x₂
1 | [1,1]✓  [0,1]✓
  |
0 | [0,0]✗  [1,0]✓
  └─────────────── x₁
    0       1

A line can separate ✓ from ✗
Perceptron CAN learn this!
```

**AND Gate:**
```
  x₂
1 | [1,1]✓  [0,1]✗
  |
0 | [0,0]✗  [1,0]✗
  └─────────────── x₁
    0       1

A line can separate ✓ from ✗
Perceptron CAN learn this!
```

### Non-Linearly Separable Problems ✗

**XOR Gate:**
```
  x₂
1 | [1,1]✗  [0,1]✓
  |
0 | [0,0]✗  [1,0]✓
  └─────────────── x₁
    0       1

No single line can separate ✓ from ✗
Perceptron CANNOT learn this!
```

This limitation led to "AI winter" in the 1970s. Multi-layer networks solve this problem (Lesson 5).

---

## Implementation Overview

### Python Structure

```python
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize random weights and bias
        self.weights = [random() for _ in range(input_size)]
        self.bias = random()
        self.learning_rate = learning_rate
    
    def predict(self, inputs):
        # Calculate weighted sum
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
        weighted_sum += self.bias
        
        # Apply activation
        return 1 if weighted_sum >= 0 else 0
    
    def train(self, training_data, epochs):
        for epoch in range(epochs):
            for inputs, expected in training_data:
                # Predict
                predicted = self.predict(inputs)
                
                # Calculate error
                error = expected - predicted
                
                # Update weights and bias
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * error * inputs[i]
                self.bias += self.learning_rate * error
```

---

## Practical Examples

### Example 1: OR Gate

```
Training Data:
┌───────┬───────┬──────────┐
│  x₁   │  x₂   │  Output  │
├───────┼───────┼──────────┤
│  0    │  0    │    0     │
│  0    │  1    │    1     │
│  1    │  0    │    1     │
│  1    │  1    │    1     │
└───────┴───────┴──────────┘

After Training:
Weights: [0.9, 0.9]
Bias: -0.4

Test: All examples classified correctly ✓
```

### Example 2: AND Gate

```
Training Data:
┌───────┬───────┬──────────┐
│  x₁   │  x₂   │  Output  │
├───────┼───────┼──────────┤
│  0    │  0    │    0     │
│  0    │  1    │    0     │
│  1    │  0    │    0     │
│  1    │  1    │    1     │
└───────┴───────┴──────────┘

After Training:
Weights: [0.7, 0.7]
Bias: -1.0

Test: All examples classified correctly ✓
```

### Example 3: Custom Classification

```
Problem: Classify if sum of inputs > 1.5

Training Data (3 inputs):
┌──────┬──────┬──────┬──────────┬────────┐
│  x₁  │  x₂  │  x₃  │   Sum    │ Output │
├──────┼──────┼──────┼──────────┼────────┤
│  0   │  0   │  0   │   0.0    │   0    │
│  1   │  0   │  0   │   1.0    │   0    │
│  0   │  1   │  0   │   1.0    │   0    │
│  1   │  1   │  0   │   2.0    │   1    │
│  1   │  0   │  1   │   2.0    │   1    │
│  0   │  1   │  1   │   2.0    │   1    │
│  1   │  1   │  1   │   3.0    │   1    │
└──────┴──────┴──────┴──────────┴────────┘

Perceptron learns to classify correctly!
```

---

## Training Visualization

### Learning Progress

```
Epoch 01: errors=4, weights=[0.3, 0.2], bias=-0.1
Epoch 02: errors=2, weights=[0.5, 0.4], bias=-0.2
Epoch 03: errors=1, weights=[0.7, 0.6], bias=-0.3
Epoch 04: errors=0, weights=[0.8, 0.7], bias=-0.4
Training completed.

Error Rate Over Time:
Errors
  4 | ●
  3 |
  2 |   ●
  1 |     ●
  0 |       ●───●───●
    └───────────────── Epochs
      1 2 3 4 5 6
```

---

## Key Takeaways

### Perceptron Strengths
- ✓ Simple and interpretable
- ✓ Fast training
- ✓ Guaranteed convergence for linearly separable data
- ✓ Foundation for understanding deep learning

### Perceptron Limitations
- ✗ Only learns linear boundaries
- ✗ Cannot solve XOR and similar problems
- ✗ Binary output only (0 or 1)
- ✗ Single layer limits complexity

These limitations are addressed by multi-layer networks (covered in Lesson 5).

---

## Hands-On Practice

**Try the implementation:**
1. Navigate to `/perceptron` directory
2. Open `main.py` to see examples
3. Run the code: `python main.py`
4. Modify training data to experiment
5. Try creating your own classification problems

**Exercises:**
1. Train a perceptron for a NAND gate
2. Create a perceptron for: output 1 if x₁ > 0.5, else 0
3. Experiment with different learning rates (0.01, 0.1, 1.0)
4. Try to train on XOR - observe that it fails

---

## Common Questions

**Q: How many parameters does a perceptron have?**  
A: For n inputs: n weights + 1 bias = n+1 parameters total

**Q: What if the data isn't linearly separable?**  
A: The perceptron won't converge. Need multi-layer networks.

**Q: How do I choose the learning rate?**  
A: Start with 0.1. Too high = unstable, too low = slow learning.

**Q: Why use bias instead of just adjusting weights?**  
A: Bias allows the boundary to shift away from the origin.

---

## Next Steps

**Continue Learning:**  
Proceed to **[Lesson 3: Understanding Parameters](03_parameters.md)** to learn about the values that neural networks learn.

**Deepen Understanding:**  
- Read the [full perceptron implementation](../perceptron/perceptron.py)
- Study the [training algorithm code](../perceptron/main.py)
- Draw your own diagrams of perceptron computation

**Explore Further:**
- Research: Perceptron convergence theorem
- Read: Original perceptron paper (Rosenblatt, 1958)
- Experiment: Build a perceptron in your favorite language

