# Neural Networks Fundamentals - Learning Course

A beginner-friendly introduction to neural networks, from basic concepts to understanding modern AI systems like GPT-3.

---

## Course Overview

This course takes you from the simplest neural network (a single perceptron) to understanding how large language models like GPT-3 are built and trained. Each lesson builds upon the previous one.

**Prerequisites:** Basic programming knowledge (Python helpful but not required)

**Time to Complete:** 3-4 hours of reading and practice

---

## Course Structure

### Part 1: Foundations (Start Here)

**Lesson 1: [Introduction to Neural Networks](01_introduction.md)**
- What is a neural network?
- Why neural networks matter
- From biological neurons to artificial neurons
- Course roadmap

**Lesson 2: [The Perceptron - Your First Neural Network](02_perceptron_basics.md)**
- Understanding the simplest neural network
- How a perceptron makes decisions
- Architecture and components
- Hands-on: Building an OR gate

### Part 2: Core Concepts

**Lesson 3: [Understanding Parameters](03_parameters.md)**
- What are parameters?
- Weights and biases explained
- How parameters store knowledge
- Parameter counting

**Lesson 4: [Understanding Bias](04_bias.md)**
- The role of bias in neural networks
- Single perceptron: one bias
- Multi-layer networks: bias per neuron
- Why bias matters

### Part 3: Scaling Up

**Lesson 5: [Multi-Layer Neural Networks](05_multi_layer_networks.md)**
- From single perceptron to deep networks
- Why multiple layers?
- Parameter distribution across layers
- Network architecture examples

### Part 4: Real-World AI Systems

**Lesson 6: [How GPT-3 Works - Architecture](06_gpt3_architecture.md)**
- GPT-3 architecture overview
- Input processing and tokenization
- 175 billion parameters explained
- Context windows and limitations

**Lesson 7: [How GPT-3 Was Trained](07_gpt3_training.md)**
- Training data and sources
- The training process
- Training scale and cost
- From internet text to language understanding

---

## Hands-On Practice

The `/perceptron` directory contains a complete implementation of a perceptron:
- `perceptron.py` - Perceptron class implementation
- `data_loader.py` - Loads training and test sets from CSV files
- `main.py` - Examples: OR gate, AND gate, and threshold-sum classification
- `data/` - Datasets as CSV files, one training and one test file per example
- `README.md` - Usage guide

Datasets live outside the code, so you can change what the perceptron learns by
editing a CSV rather than editing Python. Each CSV has a header row where all
columns except the last are input features and the last column is the label.

Run it with:

```bash
cd perceptron
python main.py
```

**Recommended:** After completing Lesson 2, run the perceptron code to see it in action.

---

## Learning Path

### For Complete Beginners
Follow the course in order from Lesson 1 → Lesson 7

### For Programmers New to AI
- Start with Lesson 2 (Perceptron Basics)
- Reference Lesson 1 if concepts are unclear
- Continue through to Lesson 7

### For Understanding Modern AI (GPT, etc.)
- Skim Lessons 1-3 for background
- Focus on Lessons 5-7
- Reference earlier lessons for fundamentals

---

## Key Concepts Covered

| Concept | Lesson | Difficulty |
|---------|--------|------------|
| Perceptron | 2 | ⭐ Beginner |
| Weights & Biases | 3 | ⭐ Beginner |
| Parameters | 3 | ⭐⭐ Intermediate |
| Multi-layer Networks | 5 | ⭐⭐ Intermediate |
| Training Process | 2, 7 | ⭐⭐ Intermediate |
| GPT Architecture | 6, 7 | ⭐⭐⭐ Advanced |

---

## What You'll Learn

After completing this course, you will understand:

✓ How neural networks make decisions  
✓ What "175 billion parameters" means  
✓ The difference between inputs and parameters  
✓ How neural networks learn from data  
✓ Why deep networks are more powerful than shallow ones  
✓ How modern AI systems like GPT-3 are built and trained  

---

## How to Use This Course

1. **Read sequentially** - Each lesson builds on previous ones
2. **Run the code** - Experiment with the perceptron implementation
3. **Take notes** - Concepts build upon each other
4. **Ask questions** - Research topics that interest you further
5. **Practice** - Try modifying the perceptron code

---

## Additional Resources

**After This Course:**
- Implement a multi-layer neural network
- Learn about backpropagation in detail
- Explore convolutional neural networks (CNNs)
- Study recurrent neural networks (RNNs)
- Deep dive into transformer architecture

**Recommended Next Steps:**
- Write your own CSV datasets and see which ones the perceptron can learn
- Build a neural network from scratch for MNIST digit recognition
- Study the math behind backpropagation
- Learn about different activation functions
- Explore PyTorch or TensorFlow frameworks

---

## Course Notes

- All examples use simple language and avoid unnecessary jargon
- Diagrams use basic ASCII art for universal compatibility
- Code examples are in Python for clarity
- Concepts are explained with real-world analogies

---

## Getting Started

**Begin with Lesson 1:** [Introduction to Neural Networks](01_introduction.md)

Or jump directly to the [Perceptron Implementation](../perceptron/README.md) if you prefer learning by doing.