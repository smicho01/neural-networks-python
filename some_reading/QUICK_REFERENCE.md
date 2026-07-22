# Neural Networks Course - Quick Reference

## Course Map

```
┌──────────────────────────────────────────────────────────────────┐
│                  NEURAL NETWORKS FUNDAMENTALS                    │
│             From Perceptron to 175 Billion Parameters            │
└──────────────────────────────────────────────────────────────────┘

Part 1: FOUNDATIONS
═══════════════════════════════════════════════════════════════════

[Lesson 1] Introduction to Neural Networks
   │
   ├─ What are neural networks?
   ├─ Biological vs artificial neurons
   ├─ Applications and importance
   └─ Course roadmap
   
[Lesson 2] The Perceptron - Your First Neural Network
   │
   ├─ Architecture: inputs, weights, bias, activation
   ├─ How predictions work
   ├─ Training algorithm
   └─ Hands-on: OR gate, AND gate
   
   PRACTICE: Run /perceptron/main.py (datasets load from /perceptron/data/*.csv)


Part 2: CORE CONCEPTS
═══════════════════════════════════════════════════════════════════

[Lesson 3] Understanding Parameters
   │
   ├─ Parameters vs inputs
   ├─ Counting parameters
   ├─ Storage and scale
   └─ Why parameter count matters
   
[Lesson 4] Understanding Bias
   │
   ├─ Role of bias in neurons
   ├─ Single neuron: 1 bias
   ├─ Multi-neuron: 1 bias per neuron
   └─ How bias enables learning


Part 3: SCALING UP
═══════════════════════════════════════════════════════════════════

[Lesson 5] Multi-Layer Neural Networks
   │
   ├─ From single to multi-layer
   ├─ Why depth matters
   ├─ Parameter distribution
   └─ Network architecture design


Part 4: REAL-WORLD AI
═══════════════════════════════════════════════════════════════════

[Lesson 6] How GPT-3 Works - Architecture
   │
   ├─ 96 layers, 175 billion parameters
   ├─ Tokenization and embeddings
   ├─ Attention mechanism
   └─ Input/output processing
   
[Lesson 7] How GPT-3 Was Trained
   │
   ├─ Training data: 300 billion tokens
   ├─ Training process
   ├─ Scale and cost
   └─ What was learned

═══════════════════════════════════════════════════════════════════
                        COURSE COMPLETE
═══════════════════════════════════════════════════════════════════
```

---

## Progress Tracking

Mark your progress as you complete each lesson:

- [ ] Lesson 1: Introduction
- [ ] Lesson 2: Perceptron Basics
- [ ] Lesson 3: Parameters
- [ ] Lesson 4: Bias
- [ ] Lesson 5: Multi-Layer Networks
- [ ] Lesson 6: GPT-3 Architecture
- [ ] Lesson 7: GPT-3 Training

**Hands-On:**
- [ ] Run perceptron OR gate example
- [ ] Run perceptron AND gate example
- [ ] Add your own dataset as a CSV pair and run it

---

## Key Concepts by Lesson

### Lesson 1: Introduction
- Neural networks learn from data
- Inspired by biological neurons
- Applications: language, vision, games

### Lesson 2: Perceptron
- Simplest neural network (1 neuron)
- Components: inputs, weights, bias, activation
- Can learn linear patterns

### Lesson 3: Parameters
- Parameters = weights + biases
- Parameters ≠ inputs
- Parameters are learned during training

### Lesson 4: Bias
- One bias per neuron
- Shifts activation threshold
- Enables non-origin boundaries

### Lesson 5: Multi-Layer
- Multiple layers enable complexity
- Hierarchical feature learning
- Parameters distributed across layers

### Lesson 6: GPT-3 Architecture
- 96 transformer layers
- 175 billion parameters
- Text → Tokens → Embeddings → Layers → Output

### Lesson 7: GPT-3 Training
- Trained on 300 billion tokens
- Next token prediction task
- 2-3 months, $4-12 million cost

---

## Quick Reference Tables

### Network Sizes

| Network | Parameters | Storage |
|---------|-----------|---------|
| Perceptron | 3 | 12 bytes |
| MNIST Net | 109K | ~437 KB |
| ResNet-50 | 25M | ~100 MB |
| BERT-Base | 110M | ~440 MB |
| GPT-3 | 175B | ~700 GB |

### Parameter Formula

```
Single Layer:
  Parameters = (inputs × outputs) + outputs
             = (weights)          + (biases)

Multi-Layer:
  Sum parameters across all layers
```

### Training Comparison

| Model | Training Time | Training Cost |
|-------|--------------|---------------|
| Perceptron | <1 second | $0 |
| AlexNet | 5-6 days | ~$1,000 |
| BERT | 4 days | ~$7,000 |
| GPT-3 | 2-3 months | $4-12M |

---

## File Structure

```
neural-networks/
├── some_reading/           ← Course materials (you are here)
│   ├── README.md           ← Course index
│   ├── QUICK_REFERENCE.md  ← This file
│   ├── 01_introduction.md
│   ├── 02_perceptron_basics.md
│   ├── 03_parameters.md
│   ├── 04_bias.md
│   ├── 05_multi_layer_networks.md
│   ├── 06_gpt3_architecture.md
│   └── 07_gpt3_training.md
│
└── perceptron/            ← Hands-on implementation
    ├── README.md          ← Usage guide
    ├── perceptron.py      ← Perceptron class
    ├── data_loader.py     ← Reads datasets from CSV
    ├── main.py            ← Examples (OR, AND, threshold sum)
    ├── data/              ← Training and test datasets
    │   ├── or_gate_train.csv
    │   ├── or_gate_test.csv
    │   ├── and_gate_train.csv
    │   ├── and_gate_test.csv
    │   ├── threshold_sum_train.csv
    │   └── threshold_sum_test.csv
    └── diagrams/          ← Visual diagrams
```

### Dataset Format

Every CSV has a header row. All columns except the last are input features,
the last column is the label:

```csv
x1,x2,label
0.0,0.0,0
0.0,1.0,1
1.0,0.0,1
1.0,1.0,1
```

The loader infers the number of features from the header, so the same code
handles 2-input and 3-input datasets without changes.

---

## Recommended Learning Paths

### Path 1: Complete Beginner
```
01 → 02 → [Practice] → 03 → 04 → 05 → 06 → 07
Estimated time: 4-5 hours
```

### Path 2: Some Programming Background
```
01 (skim) → 02 → [Practice] → 03 → 04 → 05 → 06 → 07
Estimated time: 3-4 hours
```

### Path 3: Focused on Modern AI (GPT)
```
01 (skim) → 02 (skim) → 03 → 05 → 06 → 07
Estimated time: 2-3 hours
```

---

## Key Formulas

### Perceptron Calculation
```
z = (x₁ × w₁) + (x₂ × w₂) + ... + (xₙ × wₙ) + b
output = activation(z)
```

### Parameter Count
```
Layer: (num_inputs × num_neurons) + num_neurons
```

### Training Update
```
For each weight: wᵢ = wᵢ + η × error × xᵢ
For bias: b = b + η × error
Where η = learning rate
```

---

## Common Terms

| Term | Simple Definition |
|------|-------------------|
| **Neuron** | Processing unit (takes inputs, produces output) |
| **Weight** | Number controlling input importance |
| **Bias** | Threshold adjustment value |
| **Activation** | Function deciding if neuron fires |
| **Layer** | Collection of neurons |
| **Parameter** | Weight or bias (learned from data) |
| **Training** | Adjusting parameters from examples |
| **Epoch** | One pass through all training data |
| **Token** | Piece of text (word or subword) |
| **Embedding** | Converting token to number vector |

---

## Next Steps After Course

### Implement
- [ ] Build 2-layer network from scratch
- [ ] Train on MNIST dataset
- [ ] Experiment with architectures

### Learn
- [ ] Backpropagation mathematics
- [ ] Optimization algorithms
- [ ] CNNs for image processing
- [ ] Transformers in detail

### Explore
- [ ] PyTorch or TensorFlow
- [ ] Kaggle competitions
- [ ] Research papers

### Stay Current
- [ ] Follow AI news
- [ ] Try new models
- [ ] Join AI communities

---

## Helpful Resources

**Foundations:**
- 3Blue1Brown: Neural Networks series (YouTube)
- Fast.ai: Practical Deep Learning course
- Stanford CS231n: Convolutional Neural Networks

**Frameworks:**
- PyTorch Tutorials: pytorch.org/tutorials
- TensorFlow Guide: tensorflow.org/guide
- Keras Documentation: keras.io

**Research:**
- ArXiv: arxiv.org (preprints)
- Papers With Code: paperswithcode.com
- Distill.pub: Interactive explanations

**Communities:**
- r/MachineLearning (Reddit)
- Hugging Face Forums
- AI Discord servers

---

## Tips for Success

**Learning:**
1. Don't rush - understand each concept before moving on
2. Draw diagrams - visualize networks on paper
3. Code along - modify examples, break things
4. Ask questions - research confusing topics
5. Be patient - concepts click gradually

**Practice:**
1. Start simple - single perceptron first
2. Gradually increase complexity
3. Experiment with parameters
4. Visualize what you're learning
5. Build real projects

**Remember:**
- Everyone starts as a beginner
- Understanding basics is crucial
- Scale doesn't change fundamentals
- Practice makes perfect
  Have fun learning!

---

**Quick Links:**
- [Course Index](README.md)
- [Start Learning](01_introduction.md)
- [Perceptron Code](../perceptron/README.md)