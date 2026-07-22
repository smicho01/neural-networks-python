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
- [Where the Learning Happens](#where-the-learning-happens)
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

## Where the Learning Happens

Everything above is illustration. This is the code, from `perceptron.py`:

```python
for inputs, expected_output in training_data:
    predicted_output = self.predict(inputs)
    error = expected_output - predicted_output

    self.weights = [
        weight + self.learning_rate * error * input_value
        for weight, input_value in zip(self.weights, inputs)
    ]
    self.bias += self.learning_rate * error

    total_errors += abs(error)
```

Five lines. Everything the perceptron knows lives in `self.weights` and
`self.bias`, and this is the only place either one changes.

### Reading the update rule

```python
weight + self.learning_rate * error * input_value
```

Three factors, each doing a different job:

| Factor | Role |
|---|---|
| `error` | **Direction.** `+1` means the output was too low, so increase. `-1` means too high, so decrease. `0` means correct, so the whole product is zero and nothing moves. |
| `input_value` | **Blame.** An input of `0.0` contributed nothing to the wrong answer, so its weight is left alone. An input of `1.0` contributed fully and takes the full correction. |
| `learning_rate` | **Step size.** Too large and the weights overshoot and oscillate. Too small and convergence crawls. |

There is no `if error != 0` anywhere, and none is needed. A correct prediction
multiplies the update by zero.

### Why the bias update looks different

```python
self.bias += self.learning_rate * error
```

No `input_value`. Bias behaves like a weight on an input permanently fixed at
`1.0`, so `learning_rate * error * 1.0` simplifies to `learning_rate * error`.

### Why `abs(error)`

```python
total_errors += abs(error)
```

The count is of mistakes, not of their direction. Without `abs`, a `+1` and a
`-1` in the same epoch would cancel out and two wrong predictions would look
like a perfect pass.

### What each parameter controls geometrically

The decision boundary is the line (or plane) where `w₁x₁ + w₂x₂ + ... + b = 0`.

- **Weights** set its angle. Changing them rotates the boundary.
- **Bias** sets its distance from the origin. Changing it slides the boundary
  without turning it.

Training is those two adjustments repeated until every training example falls on
the correct side.

### Watching it happen

Each epoch prints the current parameters. From a real AND gate run:

```
Epoch 01, errors: 5, weights: [0.538, -0.535], bias: 0.26
Epoch 05, errors: 2, weights: [0.508, -0.125], bias: -0.04
Epoch 09, errors: 1, weights: [0.298,  0.094], bias: -0.24
Epoch 13, errors: 0, weights: [0.278,  0.184], bias: -0.34
```

The second weight starts negative and has to cross zero before the boundary
points the right way. That crossing accounts for most of the 13 epochs. A luckier
random start would have finished in three or four.

### A note for anyone continuing into deep learning

This update rule does **not** generalise. It was derived geometrically by
Rosenblatt in 1958, before backpropagation existed, and it works only because the
step function's output is binary. Every network after this one trains by gradient
descent, where the update includes the derivative of the activation function:

```
gradient descent:  w ← w + η × error × f'(z) × x
perceptron rule:   w ← w + η × error × x
```

The missing `f'(z)` is not an omission, it is the reason the perceptron is a
special case. What does carry forward is the forward pass and the epoch loop
structure, not the arithmetic above.

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
├── main.py            ← Runs the examples
├── data/              ← Datasets, one train/test pair per example
│   ├── or_gate_train.csv
│   ├── or_gate_test.csv
│   ├── and_gate_train.csv
│   ├── and_gate_test.csv
│   ├── threshold_sum_train.csv
│   ├── threshold_sum_test.csv
│   ├── 10_inputs_train.csv
│   └── 10_inputs_test.csv
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
  Sum of 3 Inputs      100.00%
  Sum of 10 Inputs     100.00%
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
for 2-input, 3-input, and 10-input datasets without changes.

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
    title="Example 5: My Gate",
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

### Leaving a Margin

When writing a dataset, keep examples away from the boundary itself. If the rule
is `sum >= 5.0`, avoid rows summing to 4.99 or 5.01. Points sitting exactly on
the boundary force the perceptron to find one of very few valid solutions, and
whether it succeeds depends on where random initialisation happened to leave it.

The 10-input dataset was generated with a deliberate gap: no row sums to within
0.4 of the threshold. It converges reliably as a result.

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
Typically 2 to 7 epochs.

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

### Example 3: Sum of 3 Inputs

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

The rule is `sum(inputs) > 1.5`, which in three dimensions is a plane. Not a
logic gate, just a cutoff on a sum, included to show the perceptron is not
limited to two inputs. Typically 4 to 11 epochs.

**This example does not always reach 100%.** Both the training and test sets
contain rows summing to exactly 1.5, the boundary value. With almost no margin,
whether the final plane classifies them correctly depends on random
initialisation. Runs alternate between 87.50% and 100%. That is not a bug, it is
the margin problem made visible, and it is worth showing rather than hiding.

### Example 4: Sum of 10 Inputs

`data/10_inputs_train.csv` (40 rows, 20 of each class)

The rule is `x1 + x2 + ... + x10 >= 5.0`. Ten features between 0.0 and 1.0, so
sums range from 0 to 10 and the cutoff sits in the middle.

Generated with a 0.4 margin around the threshold, so label-0 rows sum between
2.31 and 4.54 while label-1 rows sum between 5.48 and 6.86. That gap is why this
example converges reliably where Example 3 does not.

Needs more epochs than the smaller examples, typically 16 to 26, because ten
weights take longer to arrange than three. The parameter count is 11: ten weights
plus one bias.

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
between classes, which usually generalises better to unseen data. Example 3 shows
the consequence directly.

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

Weights start random, so the number of epochs to converge varies.

If final *accuracy* also varies, the dataset has examples sitting too close to
the boundary. Training stops at the first solution that fits the training rows,
and when the margin is thin, some of those solutions misclassify nearby unseen
points while others do not. Example 3 does this. The fix is a wider gap between
the classes, as in Example 4.

### Training reaches zero errors but test accuracy is poor

Zero training error means the boundary separates the *training* rows, nothing
more. If unseen data lands on the wrong side, the boundary is probably sitting
right up against one class instead of running between them. Add a margin, or add
more examples near the boundary on both sides.

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
| **Margin** | Distance between the boundary and the nearest examples |

## Exercises

1. Write `data/nand_gate_train.csv` and `data/nand_gate_test.csv`, then train on
   them. NAND is separable, so this should converge.
2. Build a dataset for "output 1 if x₁ > 0.5, else 0", ignoring x₂ entirely. What
   should the trained weight for x₂ look like?
3. Run the same example with learning rates of 0.01, 0.1, and 1.0. Compare epoch
   counts and watch for oscillation at the high end.
4. Write an XOR dataset and watch training fail:
   ```csv
   x1,x2,label
   0.0,0.0,0
   0.0,1.0,1
   1.0,0.0,1
   1.0,1.0,0
   ```
   The error count will never reach zero and accuracy tops out at 75%. No number
   of epochs fixes it.
5. Take the 10-input dataset and relabel it with one feature weighted heavily,
   for example `2·x1 + x2 + ... + x10 >= 5.5`. After training, check whether the
   first weight came out noticeably larger than the rest.

## Next Steps

Replacing the step activation with a sigmoid makes the output differentiable,
which opens the door to gradient descent and backpropagation, and from there to
multi-layer networks that can handle XOR.

Note that swapping the activation alone is not enough. The training method has to
change with it, since the perceptron rule assumes a binary output. See the note
at the end of [Where the Learning Happens](#where-the-learning-happens).

## References

- Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
- Minsky, M., & Papert, S. (1969). "Perceptrons: An Introduction to Computational Geometry"

## License

This is an educational implementation for learning purposes.

---

**Happy Learning!**