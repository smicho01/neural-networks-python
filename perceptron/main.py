"""
Example usage of the Perceptron class.

This file demonstrates how to use the Perceptron class to learn
different logical gates (OR, AND) and more complex classifications.
"""

from perceptron import Perceptron


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
or_gate_data = [
    ([0.0, 0.0], 0),  # 0 OR 0 = 0
    ([0.0, 1.0], 1),  # 0 OR 1 = 1
    ([1.0, 0.0], 1),  # 1 OR 0 = 1
    ([1.0, 1.0], 1),  # 1 OR 1 = 1
]

# Create a perceptron with 2 inputs and learning rate of 0.1
or_perceptron = Perceptron(input_size=2, learning_rate=0.1)

# Train the perceptron
or_perceptron.train(training_data=or_gate_data, epochs=10)

# Test the trained perceptron
print("\nTesting OR gate:")
for inputs, expected in or_gate_data:
    prediction = or_perceptron.predict(inputs)
    print(f"  Inputs: {inputs} -> Predicted: {prediction}, Expected: {expected}")


# ============================================================================
# Example 2: Training a perceptron to learn the AND logic gate
# ============================================================================

print("\n" + "=" * 60)
print("Example 2: AND Gate")
print("=" * 60)

# The AND gate returns 1 only if both inputs are 1
and_gate_data = [
    ([0.0, 0.0], 0),  # 0 AND 0 = 0
    ([0.0, 1.0], 0),  # 0 AND 1 = 0
    ([1.0, 0.0], 0),  # 1 AND 0 = 0
    ([1.0, 1.0], 1),  # 1 AND 1 = 1
]

and_perceptron = Perceptron(input_size=2, learning_rate=0.1)
and_perceptron.train(training_data=and_gate_data, epochs=10)

print("\nTesting AND gate:")
for inputs, expected in and_gate_data:
    prediction = and_perceptron.predict(inputs)
    print(f"  Inputs: {inputs} -> Predicted: {prediction}, Expected: {expected}")


# ============================================================================
# Example 3: More complex data with 3 inputs
# ============================================================================

print("\n" + "=" * 60)
print("Example 3: Custom classification with 3 inputs")
print("=" * 60)

# More complex training data with 3 input features
# Returns 1 if sum of inputs > 1.5
complex_data = [
    ([0.0, 0.0, 0.0], 0),
    ([1.0, 0.0, 0.0], 0),
    ([0.0, 1.0, 0.0], 0),
    ([0.0, 0.0, 1.0], 0),
    ([1.0, 1.0, 0.0], 1),
    ([1.0, 0.0, 1.0], 1),
    ([0.0, 1.0, 1.0], 1),
    ([1.0, 1.0, 1.0], 1),
]

complex_perceptron = Perceptron(input_size=3, learning_rate=0.1)
complex_perceptron.train(training_data=complex_data, epochs=20)

print("\nTesting custom classification:")
for inputs, expected in complex_data:
    prediction = complex_perceptron.predict(inputs)
    print(f"  Inputs: {inputs} -> Predicted: {prediction}, Expected: {expected}")

