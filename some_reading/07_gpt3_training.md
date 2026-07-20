# Lesson 7: How GPT-3 Was Trained

**Learning Objectives:**
- Understand the training process for large language models
- Learn what data GPT-3 was trained on
- See how training works at massive scale
- Understand the cost and complexity involved

---

## Training Overview

### What is Training?

Training is the process of adjusting a neural network's parameters (weights and biases) so it can perform a task accurately.

```
Before Training:
  Parameters: Random values
  Output: Nonsense

Training Process:
  Feed examples → Calculate errors → Adjust parameters
  Repeat billions of times

After Training:
  Parameters: Learned values
  Output: Coherent, useful text
```

---

## The Training Task

### Next Token Prediction

GPT-3 was trained on a simple task: **predict the next token**

```
Training Example 1:
  Input:  "The cat sat on the"
  Target: "mat"
  
Training Example 2:
  Input:  "Machine learning is a subset of"
  Target: "artificial"
  
Training Example 3:
  Input:  "def factorial(n):"
  Target: "if"

This simple task teaches language understanding!
```

### Why Next Token Prediction Works

```
By learning to predict next tokens, GPT-3 learns:
  ✓ Grammar and syntax
  ✓ Word meanings and relationships
  ✓ Factual knowledge
  ✓ Reasoning patterns
  ✓ Common sense
  ✓ Code structure
  ✓ Conversational patterns

All from one simple objective!
```

---

## Training Data

### Data Sources

GPT-3 was trained on approximately **570 GB of text**:

```
┌─────────────────────────────────────────────┐
│ Common Crawl (web scraping)                 │
│ Size: 410 GB                                │
│ Percentage: 60%                             │
│ Content: Web pages from across internet     │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ WebText2 (curated web content)              │
│ Size: 19 GB                                 │
│ Percentage: 22%                             │
│ Content: Outbound links from Reddit         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Books1 (internet books)                     │
│ Size: 12 GB                                 │
│ Percentage: 8%                              │
│ Content: Internet-based books corpus        │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Books2 (books corpus)                       │
│ Size: 55 GB                                 │
│ Percentage: 8%                              │
│ Content: Additional books corpus            │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Wikipedia (encyclopedia)                    │
│ Size: 3 GB                                  │
│ Percentage: 3%                              │
│ Content: English Wikipedia                  │
└─────────────────────────────────────────────┘

═════════════════════════════════════════════
TOTAL: ~570 GB = ~300 billion tokens
═════════════════════════════════════════════
```

### Data Characteristics

```
Content types:
  ✓ News articles
  ✓ Scientific papers
  ✓ Books and literature
  ✓ Discussions and forums
  ✓ Wikipedia articles
  ✓ Code repositories
  ✓ Educational content

Languages:
  → Primarily English
  → Some multilingual content

Time period:
  → Data collected before 2020
  → No knowledge of events after training
```

---

## The Training Process

### Step-by-Step Training

```
STEP 1: Data Preparation
  ↓
  Raw text: "Artificial intelligence is transforming the world."
  ↓
STEP 2: Create Training Examples
  ↓
  Example 1:
    Input:  "Artificial"
    Target: "intelligence"
  
  Example 2:
    Input:  "Artificial intelligence"
    Target: "is"
  
  Example 3:
    Input:  "Artificial intelligence is"
    Target: "transforming"
  
  ... and so on
  ↓
STEP 3: Tokenize
  ↓
  Convert text to token IDs
  "Artificial" → 38293
  "intelligence" → 4430
  ↓
STEP 4: Forward Pass
  ↓
  Feed input through 96 layers
  Get prediction probabilities
  ↓
STEP 5: Calculate Loss
  ↓
  Compare prediction with target
  Measure how wrong the prediction was
  ↓
STEP 6: Backward Pass (Backpropagation)
  ↓
  Calculate gradients for all 175B parameters
  Determine how each parameter contributed to error
  ↓
STEP 7: Update Parameters
  ↓
  Adjust all 175B parameters slightly
  Parameters move toward better predictions
  ↓
STEP 8: Repeat
  ↓
  Process next training example
  Do this billions of times!
```

---

## Training at Scale

### Training Statistics

| Metric | Value |
|--------|-------|
| **Total tokens** | ~300 billion |
| **Epochs** | 1-2 (estimated) |
| **Batch size** | ~3.2 million tokens |
| **Training steps** | ~hundreds of billions |
| **Training time** | 2-3 months |
| **GPUs** | ~10,000 NVIDIA V100 |
| **Cost** | $4-12 million |
| **Energy** | ~1,287 MWh |
| **CO₂ emissions** | ~552 tons |

### Why So Few Epochs?

```
Traditional approach:
  └─ Small dataset, many epochs
     Example: 10,000 examples × 100 epochs

GPT-3 approach:
  └─ Massive dataset, few epochs
     Example: 300 billion tokens × 1-2 epochs

Benefits:
  ✓ More diverse examples
  ✓ Better generalization
  ✓ Less overfitting
  ✓ Learns from variety, not repetition
```

---

## Training Example (Detailed)

### Processing a Single Sentence

```
Training Data:
  "Machine learning is a subset of artificial intelligence."

══════════════════════════════════════════════

Training Step 1:
  Input tokens:  "Machine learning is a subset of"
  Target token:  "artificial"
  
  ┌─ Tokenize input
  │  [38395, 4673, 318, 257, 24637, 286]
  ↓
  ┌─ Embed tokens
  │  6 × 12,288 = 73,728 numbers
  ↓
  ┌─ Process through 96 transformer layers
  │  Using 175 billion parameters
  ↓
  ┌─ Output probabilities for all 50,257 tokens
  │  "artificial"   → 0.12 (12%)
  │  "AI"          → 0.08 (8%)
  │  "computer"    → 0.05 (5%)
  │  ... (other 50,254 tokens)
  ↓
  ┌─ Calculate loss
  │  Target was "artificial" but probability only 12%
  │  Loss = -log(0.12) = 2.12 (high error!)
  ↓
  ┌─ Backpropagation
  │  Calculate gradients for all 175B parameters
  ↓
  ┌─ Update parameters
  │  For each of 175 billion parameters:
  │    param = param - learning_rate × gradient
  │  
  │  Example updates:
  │    w₁ = 0.234 - 0.001 × 0.05 = 0.23395
  │    w₂ = -0.567 - 0.001 × (-0.03) = -0.56697
  │    ... (175 billion more updates!)
  ↓
Move to next training example

══════════════════════════════════════════════

After billions of such steps:
  Model learns that "artificial" commonly follows
  "subset of" in this context!
```

---

## Training vs Inference

### Key Differences

| Aspect | Training | Inference (Using GPT-3) |
|--------|----------|-------------------------|
| **Data** | 300B tokens from internet | Your prompt |
| **Parameters** | Being adjusted | Fixed (learned) |
| **Duration** | 2-3 months | 1-5 seconds |
| **Hardware** | 10,000 GPUs | Few GPUs |
| **Cost** | $4-12 million | $0.01-0.10 per query |
| **Purpose** | Learn language patterns | Generate text |
| **Happens** | Once (2020) | Every time you use it |

### Timeline

```
Phase 1: Training (2020)
  ├─ Collect 570 GB of text
  ├─ Process 300 billion tokens
  ├─ Adjust 175 billion parameters
  ├─ Run for 2-3 months
  └─ Cost: millions of dollars

Phase 2: Deployment (2020-present)
  ├─ Save trained parameters (700 GB)
  ├─ Load into GPU clusters
  ├─ Serve to users via API
  └─ Parameters stay fixed

Phase 3: Your usage (anytime)
  ├─ Send prompt to API
  ├─ GPT-3 processes with learned parameters
  ├─ Receive generated text
  └─ Parameters unchanged
```

---

## Comparing Training Scales

### Your Perceptron vs GPT-3

```
Perceptron (from Lesson 2):
┌──────────────────────────────────────┐
│ Training data: 4 examples            │
│ Parameters: 3                        │
│ Epochs: 10                           │
│ Time: <1 second                      │
│ Hardware: Your laptop CPU            │
│ Cost: $0.00                          │
│ Total updates: 40 (4 × 10)           │
└──────────────────────────────────────┘

GPT-3:
┌──────────────────────────────────────┐
│ Training data: 300 billion tokens    │
│ Parameters: 175 billion              │
│ Epochs: 1-2                          │
│ Time: 2-3 months                     │
│ Hardware: 10,000 GPUs                │
│ Cost: $4-12 million                  │
│ Total updates: hundreds of billions  │
└──────────────────────────────────────┘

Scale factor:
  Data: 75,000,000,000x more
  Parameters: 58,000,000,000x more
  Cost: ∞ (perceptron is free)
  Time: 7,000,000x longer
```

---

## The Training Loop (Conceptual)

### Simplified Code

```python
# Initialize GPT-3 with 175 billion random parameters
model = GPT3(layers=96, params=175_000_000_000)

# Load training data
training_data = load_internet_text()  # 300 billion tokens

# Training hyperparameters
learning_rate = 0.00001  # Very small for stable training
batch_size = 3_200_000   # Process multiple examples together

# Training loop
for epoch in range(2):  # 1-2 epochs
    print(f"Starting epoch {epoch + 1}")
    
    for batch in get_batches(training_data, batch_size):
        # 1. Forward pass
        inputs, targets = batch
        predictions = model.forward(inputs)
        # Processes through 96 layers using 175B parameters
        
        # 2. Calculate loss
        loss = cross_entropy_loss(predictions, targets)
        
        # 3. Backward pass
        gradients = model.backward(loss)
        # Calculates gradients for all 175B parameters
        
        # 4. Update parameters
        for i in range(175_000_000_000):
            model.parameters[i] -= learning_rate * gradients[i]
        
        # 5. Log progress
        if step % 1000 == 0:
            print(f"Step {step}, Loss: {loss}")
    
    print(f"Epoch {epoch + 1} complete")

# Save trained model
model.save("gpt3_175b.pt")  # 700 GB file!
print("Training complete!")
```

---

## Training Challenges

### Technical Challenges

```
1. Memory Requirements
   Problem: 175B parameters won't fit on one GPU
   Solution: Model parallelism across thousands of GPUs

2. Gradient Instability
   Problem: Gradients can explode or vanish in deep networks
   Solution: Careful initialization, gradient clipping, layer norm

3. Training Time
   Problem: Would take years on single GPU
   Solution: Massive parallelization (10,000 GPUs)

4. Data Quality
   Problem: Internet text has errors, biases, toxic content
   Solution: Filtering, cleaning, quality scoring

5. Convergence
   Problem: Ensuring model actually learns
   Solution: Careful monitoring, learning rate schedules
```

### Hardware Infrastructure

```
GPT-3 Training Cluster (estimated):

Compute:
  └─ ~10,000 NVIDIA V100 GPUs
     └─ 32 GB memory each
     └─ Connected via high-speed network

Storage:
  └─ Petabytes for training data
  └─ Fast SSDs for data loading

Networking:
  └─ High-bandwidth interconnects
  └─ Low-latency communication

Power:
  └─ Several megawatts
  └─ Industrial cooling systems
```

---

## What GPT-3 Learned

### Knowledge Acquired

Through training, GPT-3 learned:

```
Language Understanding:
  ✓ Grammar and syntax
  ✓ Word relationships
  ✓ Context and meaning
  ✓ Multiple languages (primarily English)

World Knowledge:
  ✓ Historical facts
  ✓ Scientific concepts
  ✓ Geography
  ✓ Common sense reasoning
  ✓ Cultural references

Skills:
  ✓ Text summarization
  ✓ Question answering
  ✓ Code generation
  ✓ Translation
  ✓ Creative writing
  ✓ Logical reasoning

All without explicit programming for each task!
```

### Emergent Abilities

Unexpected capabilities that emerged from training:

```
Few-shot learning:
  └─ Can learn from a few examples in the prompt

Arithmetic:
  └─ Can perform basic math (not 100% reliable)

Code understanding:
  └─ Can write and debug code

Analogical reasoning:
  └─ Can draw comparisons and analogies

Task adaptation:
  └─ Can switch between different tasks

These weren't explicitly trained!
They emerged from scale + data.
```

---

## Training Progress Visualization

### Learning Over Time

```
Training Progress (simplified):

Week 1: (0-10% of data)
  Predictions: Random gibberish
  Loss: Very high
  Output: "xkj qpz rlm..."

Week 4: (10-30% of data)
  Predictions: Some word-like patterns
  Loss: High
  Output: "the cat sat... qpz"

Week 8: (30-60% of data)
  Predictions: Grammatical sentences
  Loss: Medium
  Output: "The cat sat on the..."

Week 12: (60-100% of data)
  Predictions: Coherent, contextual
  Loss: Lower
  Output: "The cat sat on the mat and purred."

End of Training:
  Predictions: High-quality text
  Loss: Low
  Output: [Coherent, contextual, often correct]
```

---

## Post-Training Fine-Tuning

### Additional Training Steps

```
Phase 1: Pre-training (what we discussed)
  └─ 300 billion tokens
  └─ Next token prediction
  └─ Result: GPT-3 base model

Phase 2: Fine-tuning (optional)
  └─ Smaller, curated datasets
  └─ Specific tasks or behaviors
  └─ Result: GPT-3.5, ChatGPT

Phase 3: RLHF (Reinforcement Learning from Human Feedback)
  └─ Human preferences
  └─ Reward modeling
  └─ Result: More helpful, harmless outputs
```

---

## Comparing Model Training

### Historical Progression

```
1958: Perceptron
  Training data: ~10 examples
  Time: Seconds
  Cost: $0

2012: AlexNet
  Training data: 1.2 million images
  Time: 5-6 days
  Cost: ~$1,000

2018: BERT
  Training data: 3.3 billion words
  Time: 4 days (64 TPUs)
  Cost: ~$7,000

2020: GPT-3
  Training data: 300 billion tokens
  Time: 2-3 months
  Cost: $4-12 million

2023: GPT-4 (estimated)
  Training data: Undisclosed (larger)
  Time: Several months
  Cost: $50-100 million (estimated)

Trend: Exponential growth in scale and cost
```

---

## Environmental Impact

### Energy and Emissions

```
GPT-3 Training:
  Electricity: ~1,287 MWh
  CO₂ emissions: ~552 tons
  Equivalent to:
    - 120 cars driving for a year
    - 550 transatlantic flights
    - Average US home for 47 years

For perspective:
  - One training run, not per use
  - Inference (using GPT-3) much more efficient
  - Ongoing research to reduce costs
```

---

## Key Takeaways

**Training Process:**
- Simple objective: predict next token
- Massive dataset: 300 billion tokens from internet
- Few epochs: 1-2 passes through data
- Billions of parameter updates

**Training Scale:**
- 175 billion parameters adjusted
- 2-3 months of computation
- 10,000 GPUs working in parallel
- $4-12 million in compute costs

**What Was Learned:**
- Language understanding from data patterns
- World knowledge from internet text
- Skills emerge from scale + data
- No explicit programming for specific tasks

**After Training:**
- Parameters are fixed (700 GB file)
- Deployed for public use via API
- No further learning during use
- Same model serves millions of requests

---

## Common Questions

**Q: Could we train GPT-3 on a personal computer?**  
A: Theoretically yes, but it would take ~355 years on a consumer GPU!

**Q: Why not train for more epochs?**  
A: Diminishing returns + risk of overfitting. More data > more epochs.

**Q: Does GPT-3 learn from my conversations?**  
A: No. Parameters are fixed after training. Your conversations don't update the model.

**Q: How often is GPT-3 retrained?**  
A: Rarely. New models (GPT-3.5, GPT-4) are trained separately.

**Q: Why such a large dataset?**  
A: More diverse data = better generalization = more capable model.

---

## Course Completion

### Congratulations!

You've completed the Neural Networks Fundamentals course!

### Journey Summary

```
Lesson 1: Introduction
  └─ Learned what neural networks are

Lesson 2: Perceptron
  └─ Built first neural network (3 parameters)

Lesson 3: Parameters
  └─ Understood weights and biases

Lesson 4: Bias
  └─ Learned role of bias in neurons

Lesson 5: Multi-Layer Networks
  └─ Saw how layers stack for complexity

Lesson 6: GPT-3 Architecture
  └─ Explored 175B parameter model

Lesson 7: GPT-3 Training (You are here)
  └─ Understood training at massive scale

From 3 parameters to 175 billion!
```

---

## What's Next?

### Continue Your Learning

**Implement More:**
- Build a 2-layer network from scratch
- Train on MNIST dataset
- Experiment with different architectures

**Learn Advanced Topics:**
- Backpropagation mathematics
- Optimization algorithms (Adam, RMSprop)
- Regularization techniques
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformer architecture in depth

**Explore Frameworks:**
- PyTorch tutorials
- TensorFlow/Keras
- JAX

**Stay Current:**
- Follow AI research papers
- Try new models as they release
- Experiment with APIs (OpenAI, Anthropic, etc.)

---

## Final Thoughts

Neural networks have evolved from simple perceptrons to models with hundreds of billions of parameters. The fundamental concepts remain the same:

- **Neurons** process inputs with weights and biases
- **Layers** stack to create depth
- **Training** adjusts parameters from data
- **Scale** enables incredible capabilities

Understanding these basics provides a foundation for exploring any neural network, from the simplest perceptron to the most advanced AI systems.

---

**End of Course**

Thank you for learning about neural networks!

