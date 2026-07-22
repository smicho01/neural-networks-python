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
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Dataset Format](#dataset-format)
- [Examples](#examples)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

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

Because both `y` and `ŷ` are binary, the error is always `-1`, `0`, or `1`. Its
sign gives the direction of the correction, and a value of zero makes the whole
update vanish. That is why the code needs no explicit "only update when wrong"
check - the arithmetic handles it.

The bias update has no `xᵢ` factor because bias behaves like a weight on an input
permanently fixed at `1.0`.

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

**Early stopping**: training breaks out as soon as an epoch completes with zero
errors, since every subsequent update would be multiplied by zero and change
nothing.

## Installation

No external dependencies, just the standard library (`random`, `csv`, `pathlib`).
Requires Python 3.10 or newer for the union type syntax (`str | Path`) used in
the data loader.

```bash
# Clone or download the repository
cd perceptron/

# Run the examples
python main.py
```

## Project Structure

```
perceptron/
├── README.md          ← This file
├── perceptron.py      ← Perceptron class: predict and train
├── data_loader.py     ← Reads datasets from CSV files
├── main.py            ← Runs the three examples
├── data/              ← Datasets, one train/test pair per example
│   ├── or_gate_train.csv
│   ├── or_gate_test.csv
│   ├── and_gate_train.csv
│   ├── and_gate_test.csv
│   ├── threshold_sum_train.csv
│   └── threshold_sum_test.csv
└── diagrams/          ← Visual diagrams
```

Datasets live outside the code, so changing what the perceptron learns means
editing a CSV rather than editing Python.

## Usage

### Running the Examples

```bash
python main.py
```

Output ends with a summary:

```
============================================================
Summary
============================================================
  OR Gate              100.00%
  AND Gate             100.00%
  Threshold Sum        100.00%
```

Epoch counts vary between runs because weights start at random values.

### Loading Data from CSV

```python
from data_loader import load_dataset, get_input_size
from perceptron import Perceptron

training_data = load_dataset("data/or_gate_train.csv")
test_data = load_dataset("data/or_gate_test.csv")

perceptron = Perceptron(
    input_size=get_input_size(training_data),
    learning_rate=0.1,
)
perceptron.train(training_data=training_data, epochs=40)

prediction = perceptron.predict([1.0, 0.0])
print(f"Prediction: {prediction}")  # Output: 1
```

`get_input_size` reads the feature count from the data, so the same code works
for 2-input and 3-input datasets without changes.

### Building Data In Memory

The CSV loader is a convenience, not a requirement. `train` accepts any list of
tuples:

```python
from perceptron import Perceptron

training_data = [
    ([0.0, 0.0], 0),
    ([0.0, 1.0], 1),
    ([1.0, 0.0], 1),
    ([1.0, 1.0], 1),
]

perceptron = Perceptron(input_size=2, learning_rate=0.1)
perceptron.train(training_data=training_data, epochs=10)
```

### Training Data Structure

Whether it comes from a CSV or a literal, the in-memory type is:

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

### Adding Your Own Dataset

Write the two CSV files, then add one call in `main()`:

```python
"My Gate": run_example(
    title="Example 4: My Gate",
    train_file="data/my_gate_train.csv",
    test_file="data/my_gate_test.csv",
    epochs=50,
),
```

## Dataset Format

Every CSV has a header row. All columns except the last are input features, the
last column is the label (`0` or `1`):

```csv
x1,x2,label
0.0,0.0,0
0.0,1.0,1
1.0,0.0,1
1.0,1.0,1
```

Feature count is inferred from the file, so a 3-input dataset just needs a third
column before `label`. Nothing in the code changes.

### Train and Test Sets

Each example uses two files. The training set is what the perceptron learns from.
The test set contains examples it never sees during training, so accuracy on it
measures generalisation rather than memorisation. A model scoring 100% on its
training data and 60% on unseen data has memorised, not learned.

## Examples

### Example 1: OR Logic Gate

`data/or_gate_train.csv`
```csv
x1,x2,label
0.0,0.0,0
0.0,1.0,1
1.0,0.0,1
1.0,1.0,1
0.1,0.8,1
0.9,0.1,1
0.2,0.3,1
0.1,0.1,0
0.05,0.05,0
0.15,0.1,0
```

Labels follow `x1 + x2 >= ~0.4`, a straight line, so the perceptron converges.
Typically 7 epochs.

### Example 2: AND Logic Gate

`data/and_gate_train.csv`
```csv
x1,x2,label
0.0,0.0,0
0.0,1.0,0
1.0,0.0,0
0.2,0.8,0
0.4,0.9,0
0.5,0.8,0
0.3,0.7,0
0.6,0.6,0
1.0,1.0,1
0.8,0.9,1
0.9,0.7,1
0.85,0.8,1
0.95,0.9,1
0.7,0.9,1
```

Labels follow `x1 + x2 >= ~1.5`. On binary inputs this behaves exactly like AND,
and on continuous inputs it stays linearly separable. Typically 13 epochs.

See [Troubleshooting](#troubleshooting) for why the more intuitive rule "both
inputs at least 0.5" would break this example.

### Example 3: Threshold Sum (3 inputs)

`data/threshold_sum_train.csv`
```csv
x1,x2,x3,label
0.0,0.0,0.0,0
1.0,0.0,0.0,0
0.0,1.0,0.0,0
0.0,0.0,1.0,0
1.0,1.0,0.0,1
1.0,0.0,1.0,1
0.0,1.0,1.0,1
1.0,1.0,1.0,1
0.5,0.5,0.5,0
0.8,0.4,0.5,1
0.3,0.6,0.3,0
0.6,0.7,0.4,1
0.2,0.3,0.4,0
0.9,0.8,0.1,1
```

The rule is `sum(inputs) > 1.5`, which in three dimensions is a plane. This is
the easiest of the three because the target rule matches exactly what the
architecture can express. Typically 4 epochs.

## Limitations

### What Perceptrons CAN Learn
✅ **Linearly separable patterns**:
- OR gate
- AND gate
- NOT gate
- Any rule expressible as `w₁x₁ + w₂x₂ + ... + b ≥ 0`

### What Perceptrons CANNOT Learn
❌ **Non-linearly separable patterns**:
- XOR gate (exclusive OR)
- Any rule combining several thresholds with AND
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

### Solution Quality

Even on separable data, the perceptron stops at the *first* line that classifies
everything correctly, not the best one. An SVM by contrast maximises the margin
between classes, which usually generalises better to unseen data.

## Troubleshooting

### Training never reaches zero errors

If `total_errors` bounces around instead of settling, the dataset is almost
certainly not linearly separable. More epochs will not help. More examples of the
same kind will not help either. Check the labelling rule.

**Quick test:** if two rows have the same weighted sum but different labels, no
line can separate them.

**A real example of this trap.** It is tempting to label an AND dataset with the
rule "output 1 when both inputs are at least 0.5". That rule is not linearly
separable. It describes a square corner in input space, and a corner needs two
lines:

```
  x₂
 1.0 ┤ · · · │ ✓ ✓ ✓      The region "x₁ ≥ 0.5 AND x₂ ≥ 0.5"
     │ · · · │ ✓ ✓ ✓      is a square, not a half-plane.
 0.5 ┤───────┼───────      Its boundary turns a right angle,
     │ · · · │ · · ·       which one straight line cannot trace.
     │ · · · │ · · ·
 0.0 ┴───────┴───────  x₁
    0.0     0.5     1.0
```

The contradiction is visible in the data itself. Under that rule `[0.5, 0.5]`
would be labelled 1 while `[0.2, 0.8]` would be labelled 0, yet both have a sum
of 1.0, so no line `w₁x₁ + w₂x₂ + b = 0` can put them on opposite sides.

The fix is to label by the sum rather than by a conjunction of thresholds, which
is what `data/and_gate_train.csv` does.

### Error count goes up between epochs

Normal, even on solvable data. A correction that fixes one example often breaks
another. The convergence theorem guarantees a finite number of steps for
separable data, not steady improvement.

### Accuracy differs between runs

Weights start random, so the number of epochs to converge varies. Final accuracy
should not vary on separable data. If it does, the dataset is probably not
separable.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Weights** | Parameters that determine importance of each input |
| **Bias** | Shifts the decision boundary (like intercept in linear equation) |
| **Learning Rate** | Controls how much weights change during training |
| **Epoch** | One complete pass through all training data |
| **Activation Function** | Converts weighted sum to binary output (0 or 1) |
| **Decision Boundary** | The line/hyperplane that separates the two classes |
| **Training Set** | Examples the model learns from |
| **Test Set** | Held-out examples used to measure generalisation |

## Next Steps

Replacing the step activation with a sigmoid makes the output differentiable,
which opens the door to gradient descent and backpropagation, and from there to
multi-layer networks that can handle XOR.

## References

- Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
- Minsky, M., & Papert, S. (1969). "Perceptrons: An Introduction to Computational Geometry"

## License

This is an educational implementation for learning purposes.

---

**Happy Learning!**