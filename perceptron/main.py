"""
Example usage of the Perceptron class.

This file demonstrates how to use the Perceptron class to learn
different logical gates (OR, AND) and more complex classifications.
"""

from perceptron import Perceptron


def calculate_accuracy(perceptron: Perceptron, test_data: list[tuple[list[float], int]]) -> float:
    """Calculate the accuracy of predictions on test data."""
    correct = 0
    total = len(test_data)
    for inputs, expected in test_data:
        prediction = perceptron.predict(inputs)
        if prediction == expected:
            correct += 1
    return (correct / total) * 100 if total > 0 else 0.0


# ============================================================================
# Example 1: Training a perceptron to learn the OR logic gate
# ============================================================================

print("=" * 60)
print("Example 1: OR Gate")
print("=" * 60)

# Training data structure: list[tuple[list[float], int]]
# Each tuple contains:
#   - list[float]: input features (e.g., [0.0, 1.0])
#   - int: expected output (0 or 1)
# Extended training data with more variations including intermediate values
# IMPORTANT: Added boundary cases with very small values to improve accuracy
or_gate_training_data = [
    ([0.0, 0.0], 0),  # 0 OR 0 = 0
    ([0.0, 1.0], 1),  # 0 OR 1 = 1
    ([1.0, 0.0], 1),  # 1 OR 0 = 1
    ([1.0, 1.0], 1),  # 1 OR 1 = 1
    ([0.0, 0.0], 0),  # Duplicate for reinforcement
    ([0.1, 0.8], 1),  # Near 0 OR Near 1 = 1
    ([0.9, 0.1], 1),  # Near 1 OR Near 0 = 1
    ([0.2, 0.3], 1),  # Small values OR = 1 (if either > threshold)
    ([0.1, 0.1], 0),  # Very small values = 0 (critical boundary case)
    ([0.05, 0.05], 0), # Reinforcing low boundary
]

# Separate test data (unseen during training)
or_gate_test_data = [
    ([0.0, 0.0], 0),  # Edge case: both zero
    ([0.5, 0.5], 1),  # Mid values
    ([0.3, 0.7], 1),  # Different mid values
    ([1.0, 1.0], 1),  # Edge case: both one
    ([0.8, 0.2], 1),  # Mixed values
    ([0.1, 0.1], 0),  # Very small values
]

# Create a perceptron with 2 inputs and learning rate of 0.1
or_perceptron = Perceptron(input_size=2, learning_rate=0.1)

# Train the perceptron
or_perceptron.train(training_data=or_gate_training_data, epochs=40)

# Test the trained perceptron on NEW data
print("\nTesting OR gate on unseen data:")
for inputs, expected in or_gate_test_data:
    prediction = or_perceptron.predict(inputs)
    status = "✓" if prediction == expected else "✗"
    print(f"  {status} Inputs: {inputs} -> Predicted: {prediction}, Expected: {expected}")

# Calculate and display accuracy
accuracy = calculate_accuracy(or_perceptron, or_gate_test_data)
print(f"\n>>> Prediction Accuracy: {accuracy:.2f}%")


# ============================================================================
# Example 2: Training a perceptron to learn the AND logic gate
# ============================================================================

print("\n" + "=" * 60)
print("Example 2: AND Gate")
print("=" * 60)

# The AND gate returns 1 only if BOTH inputs are >= 0.5
# This creates a clear, learnable threshold for the perceptron
# Rule: output = 1 if (input1 >= 0.5 AND input2 >= 0.5), else 0
and_gate_training_data = [
    # Clear negative examples (at least one input < 0.5)
    ([0.0, 0.0], 0),  # Both low
    ([0.0, 1.0], 0),  # One low
    ([1.0, 0.0], 0),  # One low
    ([0.1, 0.9], 0),  # One low
    ([0.9, 0.2], 0),  # One low
    ([0.4, 0.9], 0),  # One below threshold
    ([0.3, 0.9], 0),  # One below threshold
    ([0.2, 0.8], 0),  # One below threshold
    # Clear positive examples (both inputs >= 0.5)
    ([1.0, 1.0], 1),  # Both high
    ([0.8, 0.9], 1),  # Both high
    ([0.7, 0.8], 1),  # Both above threshold
    ([0.6, 0.6], 1),  # Both above threshold
    ([0.6, 0.7], 1),  # Both above threshold
    ([0.5, 0.5], 1),  # Boundary case (both exactly at threshold)
    ([0.5, 0.9], 1),  # Both above/at threshold
]

# Separate test data
and_gate_test_data = [
    ([0.0, 0.0], 0),  # Both zero
    ([0.5, 0.0], 0),  # One zero
    ([0.0, 0.5], 0),  # One zero
    ([1.0, 1.0], 1),  # Both one
    ([0.7, 0.8], 1),  # Both high
    ([0.6, 0.6], 1),  # Both medium-high
    ([0.3, 0.9], 0),  # One low, one high
]

and_perceptron = Perceptron(input_size=2, learning_rate=0.1)
and_perceptron.train(training_data=and_gate_training_data, epochs=50)

print("\nTesting AND gate on unseen data:")
for inputs, expected in and_gate_test_data:
    prediction = and_perceptron.predict(inputs)
    status = "✓" if prediction == expected else "✗"
    print(f"  {status} Inputs: {inputs} -> Predicted: {prediction}, Expected: {expected}")

# Calculate and display accuracy
accuracy = calculate_accuracy(and_perceptron, and_gate_test_data)
print(f"\n>>> Prediction Accuracy: {accuracy:.2f}%")


# ============================================================================
# Example 3: More complex data with 3 inputs
# ============================================================================

print("\n" + "=" * 60)
print("Example 3: Custom classification with 3 inputs")
print("=" * 60)

# More complex training data with 3 input features
# Returns 1 if sum of inputs > 1.5
# Extended with more training examples
complex_training_data = [
    ([0.0, 0.0, 0.0], 0),  # sum = 0
    ([1.0, 0.0, 0.0], 0),  # sum = 1
    ([0.0, 1.0, 0.0], 0),  # sum = 1
    ([0.0, 0.0, 1.0], 0),  # sum = 1
    ([1.0, 1.0, 0.0], 1),  # sum = 2
    ([1.0, 0.0, 1.0], 1),  # sum = 2
    ([0.0, 1.0, 1.0], 1),  # sum = 2
    ([1.0, 1.0, 1.0], 1),  # sum = 3
    ([0.5, 0.5, 0.5], 0),  # sum = 1.5 (boundary)
    ([0.8, 0.4, 0.5], 1),  # sum = 1.7
    ([0.3, 0.6, 0.3], 0),  # sum = 1.2
    ([0.6, 0.7, 0.4], 1),  # sum = 1.7
    ([0.2, 0.3, 0.4], 0),  # sum = 0.9
    ([0.9, 0.8, 0.1], 1),  # sum = 1.8
]

# Separate test data (unseen during training)
complex_test_data = [
    ([0.1, 0.2, 0.3], 0),  # sum = 0.6 -> expect 0
    ([0.7, 0.5, 0.4], 1),  # sum = 1.6 -> expect 1
    ([1.0, 0.5, 0.0], 0),  # sum = 1.5 -> boundary, expect 0
    ([0.6, 0.6, 0.6], 1),  # sum = 1.8 -> expect 1
    ([0.4, 0.4, 0.4], 0),  # sum = 1.2 -> expect 0
    ([1.0, 1.0, 0.5], 1),  # sum = 2.5 -> expect 1
    ([0.2, 0.2, 0.2], 0),  # sum = 0.6 -> expect 0
    ([0.8, 0.9, 0.2], 1),  # sum = 1.9 -> expect 1
]

complex_perceptron = Perceptron(input_size=3, learning_rate=0.1)
complex_perceptron.train(training_data=complex_training_data, epochs=20)

print("\nTesting custom classification on unseen data:")
for inputs, expected in complex_test_data:
    prediction = complex_perceptron.predict(inputs)
    status = "✓" if prediction == expected else "✗"
    input_sum = sum(inputs)
    print(f"  {status} Inputs: {inputs} (sum={input_sum:.1f}) -> Predicted: {prediction}, Expected: {expected}")

# Calculate and display accuracy
accuracy = calculate_accuracy(complex_perceptron, complex_test_data)
print(f"\n>>> Prediction Accuracy: {accuracy:.2f}%")
