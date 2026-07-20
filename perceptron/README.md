# Perceptron

A Python implementation of the perceptron algorithm - the fundamental building block of neural networks and deep learning.

> **Learning Resources**: New to neural networks? Check out the [Neural Networks Course](../some_reading/README.md) for structured lessons from basics to GPT-3!

> **Viewing Diagrams**: This README contains ASCII diagrams optimized for PyCharm. For interactive Mermaid diagrams, see the [diagrams/](./diagrams/) directory or view on GitHub.

## Table of Contents
- [What is a Perceptron?](#what-is-a-perceptron)
- [How Does It Work?](#how-does-it-work)
- [Architecture Diagram](#architecture-diagram)
- [Mathematical Model](#mathematical-model)
- [The Learning Algorithm](#the-learning-algorithm)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Limitations](#limitations)

## What is a Perceptron?

The **perceptron** is a type of artificial neuron invented by Frank Rosenblatt in 1957. It's a binary linear classifier that learns to separate data into two classes by finding a decision boundary (a hyperplane in multi-dimensional space).

Think of it as a simple decision-maker that:
1. Takes multiple inputs
2. Weighs them based on their importance
3. Sums them up
4. Makes a binary decision (0 or 1)

## How Does It Work?

The perceptron operates in two phases:

### 1. **Forward Pass (Prediction)**
   - Takes input features
   - Multiplies each input by its corresponding weight
   - Adds all weighted inputs together
   - Adds a bias term
   - Applies an activation function to produce output (0 or 1)

### 2. **Training (Learning)**
   - Compares prediction with expected output
   - Calculates the error
   - Updates weights and bias to reduce error
   - Repeats until data is correctly classified

## Architecture Diagram

> **View detailed diagrams**: [diagrams/](./diagrams/) directory contains all diagrams in multiple formats

### ASCII Diagram (PyCharm-friendly)

```
┌─────────┐
│ Input x₁│───┐
└─────────┘   │ ×w₁
              │
┌─────────┐   │    ┌──────────────────┐     ┌─────────────┐     ┌──────────┐
│ Input x₂│───┼───→│  Σ (Weighted     │────→│ Activation  │────→│ Output ŷ │
└─────────┘   │ ×w₂│    Sum + Bias)   │     │  Function   │     │  (0 or 1)│
              │    └──────────────────┘     │  Step(x)    │     └──────────┘
┌─────────┐   │                             └─────────────┘
│ Input x₃│───┘ ×w₃
└─────────┘   
              ↑
┌─────────┐   │
│ Input xₙ│───┘ ×wₙ
└─────────┘
              ↑
        ┌──────────┐
        │ Bias (b) │
        └──────────┘
```

**Components:**
- **Inputs (x₁, x₂, ..., xₙ)**: Feature values
- **Weights (w₁, w₂, ..., wₙ)**: Learned parameters
- **Σ (Summation)**: Computes weighted sum + bias
- **Activation**: Step function (outputs 0 or 1)
- **Output (ŷ)**: Final prediction

[View full diagram with Mermaid version](./diagrams/perceptron-architecture.md)

### Visual Flow

```
        ┌─────────────┐
        │   START     │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │   Receive   │
        │   Inputs    │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  Multiply   │
        │  Each Input │
        │  by Weight  │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │   Sum All   │
        │   Weighted  │
        │   Inputs    │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  Add Bias   │
        │    Term     │
        └──────┬──────┘
               │
               ▼
        ╔═════════════╗
        ║  Weighted   ║
        ║  Sum ≥ 0?   ║
        ╚══════╦══════╝
               │
       ┌───────┴───────┐
       │               │
     Yes              No
       │               │
       ▼               ▼
┌────────────┐  ┌────────────┐
│ Output = 1 │  │ Output = 0 │
└──────┬─────┘  └──────┬─────┘
       │               │
       └───────┬───────┘
               │
               ▼
        ┌─────────────┐
        │     END     │
        └─────────────┘
```

[View full diagram with Mermaid version](./diagrams/prediction-flow.md)

## Mathematical Model

### Forward Propagation

The perceptron computes its output using the following formula:

```
ŷ = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

Where:
- **xᵢ** = input features
- **wᵢ** = weights (learned parameters)
- **b** = bias term (learned parameter)
- **f** = activation function (step function)
- **ŷ** = predicted output

### Activation Function (Step Function)

```
f(z) = {
  1  if z ≥ 0
  0  if z < 0
}
```

### Weight Update Rule

During training, weights are updated using the perceptron learning rule:

```
wᵢ ← wᵢ + η × (y - ŷ) × xᵢ
b ← b + η × (y - ŷ)
```

Where:
- **η** (eta) = learning rate (controls step size)
- **y** = expected output
- **ŷ** = predicted output
- **(y - ŷ)** = error

## The Learning Algorithm

### Training Algorithm Flow (Simplified)

```
        ┌──────────────────┐
        │  Start Training  │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │   Initialize     │
        │  Random Weights  │
        │    and Bias      │
        └────────┬─────────┘
                 │
                 ▼
     ╔═══════════════════╗
     ║  For Each Epoch   ║◄────────────┐
     ╚═════════┬═════════╝             │
               │                        │
               ▼                        │
     ╔═══════════════════╗             │
     ║ For Each Sample   ║◄─────┐      │
     ╚═════════┬═════════╝      │      │
               │                 │      │
               ▼                 │      │
        ┌──────────────────┐    │      │
        │  Make Prediction │    │      │
        └────────┬─────────┘    │      │
                 │                │      │
                 ▼                │      │
        ┌──────────────────┐    │      │
        │ Calculate Error  │    │      │
        └────────┬─────────┘    │      │
                 │                │      │
                 ▼                │      │
          ╔══════════╗           │      │
          ║ Error≠0? ║           │      │
          ╚════╦═════╝           │      │
               │                  │      │
        ┌──────┴─────┐           │      │
       Yes           No           │      │
        │              │           │      │
        ▼              │           │      │
 ┌─────────────┐      │           │      │
 │   Update    │      │           │      │
 │   Weights   │      │           │      │
 └──────┬──────┘      │           │      │
        │              │           │      │
        └──────┬───────┘           │      │
               │                    │      │
               ▼                    │      │
        ╔══════════════╗           │      │
        ║ More Samples?║───Yes────┘      │
        ╚══════╦═══════╝                  │
               │No                         │
               ▼                          │
        ╔══════════════╗                 │
        ║Total Errors  ║                 │
        ║    = 0?      ║                 │
        ╚══════╦═══════╝                 │
               │                          │
        ┌──────┴─────┐                   │
       No            Yes                  │
        │              │                   │
        ▼              ▼                   │
 ╔═════════════╗  ┌────────────┐         │
 ║More Epochs? ║  │  Training  │         │
 ╚══════╦══════╝  │  Complete! │         │
        │Yes      └────────────┘         │
        └──────────────────────────────┘
```

[View full training algorithm diagram](./diagrams/training-algorithm.md)

## Installation

No special installation required! Just Python 3.8+ with the built-in `random` module.

```bash
# Clone or download the repository
cd perceptron/

# Run the examples
python main.py
```

## Usage

### Basic Usage

```python
from perceptron import Perceptron

# Create a perceptron with 2 inputs
perceptron = Perceptron(input_size=2, learning_rate=0.1)

# Define training data: list of (inputs, expected_output) tuples
training_data = [
    ([0.0, 0.0], 0),
    ([0.0, 1.0], 1),
    ([1.0, 0.0], 1),
    ([1.0, 1.0], 1),
]

# Train the perceptron
perceptron.train(training_data=training_data, epochs=10)

# Make predictions
prediction = perceptron.predict([1.0, 0.0])
print(f"Prediction: {prediction}")  # Output: 1
```

### Training Data Structure

The `training_data` parameter is a list of tuples:

```python
training_data: list[tuple[list[float], int]]
```

Each tuple contains:
- **First element**: List of input features `[x₁, x₂, ..., xₙ]`
- **Second element**: Expected output (0 or 1)

Example:
```python
training_data = [
    ([0.5, 0.8, 1.0], 1),  # Three inputs, expected output: 1
    ([0.2, 0.1, 0.3], 0),  # Three inputs, expected output: 0
]
```

## Examples

See `main.py` for complete working examples:

### Example 1: OR Logic Gate
```python
or_gate_data = [
    ([0.0, 0.0], 0),  # 0 OR 0 = 0
    ([0.0, 1.0], 1),  # 0 OR 1 = 1
    ([1.0, 0.0], 1),  # 1 OR 0 = 1
    ([1.0, 1.0], 1),  # 1 OR 1 = 1
]
```

### Example 2: AND Logic Gate
```python
and_gate_data = [
    ([0.0, 0.0], 0),  # 0 AND 0 = 0
    ([0.0, 1.0], 0),  # 0 AND 1 = 0
    ([1.0, 0.0], 0),  # 1 AND 0 = 0
    ([1.0, 1.0], 1),  # 1 AND 1 = 1
]
```

### Example 3: Custom Classification (3 inputs)
```python
custom_data = [
    ([0.0, 0.0, 0.0], 0),
    ([1.0, 1.0, 0.0], 1),
    ([1.0, 0.0, 1.0], 1),
    # ... more examples
]
```

## Limitations

### What Perceptrons CAN Learn
✅ **Linearly separable patterns**:
- OR gate
- AND gate
- NOT gate
- Simple linear classification problems

### What Perceptrons CANNOT Learn
❌ **Non-linearly separable patterns**:
- XOR gate (exclusive OR)
- Any problem requiring a curved decision boundary

### Visual Comparison

**Linearly Separable (CAN solve):**
```
    Class A         │        Class B
                    │
       ●            │            ○
         ●          │          ○
       ●   ●        │        ○   ○
         ●          │          ○
       ●            │            ○
                    │
    ────────────────┼────────────────
                    │  ← Single straight line
                    │     separates classes!
```

**Non-Linearly Separable (CANNOT solve):**
```
    XOR Problem:
    
         ○            ●
                           
                           
    
         ●            ○
    
    No single straight line works!
    Need curved/multiple boundaries.
```

[View detailed separability diagrams](./diagrams/linear-separability.md)

**Solution**: For non-linear problems, use multi-layer perceptrons (MLPs) or neural networks, which stack multiple perceptrons in layers.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Weights** | Parameters that determine importance of each input |
| **Bias** | Shifts the decision boundary (like intercept in linear equation) |
| **Learning Rate** | Controls how much weights change during training |
| **Epoch** | One complete pass through all training data |
| **Activation Function** | Converts weighted sum to binary output (0 or 1) |
| **Decision Boundary** | The line/hyperplane that separates the two classes |

## References

- Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
- Minsky, M., & Papert, S. (1969). "Perceptrons: An Introduction to Computational Geometry"

## License

This is an educational implementation for learning purposes.

---

**Happy Learning!**

