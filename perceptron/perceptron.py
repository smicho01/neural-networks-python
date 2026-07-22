import random

class Perceptron:
    """
    A simple perceptron - the fundamental building block of neural networks.
    The perceptron is a linear binary classifier that learns to separate
    data points into two classes by adjusting weights and bias through training.
    """

    def __init__(self, input_size: int, learning_rate: float = 0.1):
        """
        Initialize the perceptron with random weights and bias.
        Args:
            input_size: Number of input features the perceptron will receive
            learning_rate: Step size for weight updates during training (default: 0.1)
        """
        # Initialize weights randomly between -1.0 and 1.0 for each input feature
        self.weights = [
            random.uniform(-1.0, 1.0) for _ in range(input_size)
        ]
        # Initialize bias term randomly between -1.0 and 1.0
        # The bias allows the decision boundary to be shifted
        self.bias = random.uniform(-1.0, 1.0)
        # Store the learning rate for use during weight updates
        self.learning_rate = learning_rate

    def calculate_weighted_sum(self, inputs: list[float]) -> float:
        """
        Calculate the weighted sum of inputs (dot product of inputs and weights).
        This is the core computation: sum(input_i * weight_i) for all i
        Args:
            inputs: List of input feature values
        Returns:
            The weighted sum of all inputs
        """
        # Compute the dot product: multiply each input by its corresponding weight and sum
        weighted_sum = sum(
            input_value * weight for input_value, weight in zip(inputs, self.weights)
        )
        return weighted_sum + self.bias

    def activation(self, weighted_sum: float) -> int:
        """
        Apply the step activation function to produce binary output.

        This is a simple threshold function that outputs:
        - 1 if weighted_sum >= 0
        - 0 if weighted_sum < 0

        Args:
            weighted_sum: The weighted sum of inputs plus bias

        Returns:
            Binary output (0 or 1)
        """
        return 1 if weighted_sum >= 0 else 0

    def predict(self, inputs: list[float]) -> int:
        """
        Make a prediction for the given inputs.

        This method combines the weighted sum calculation and activation
        to produce a final binary classification.

        Args:
            inputs: List of input feature values

        Returns:
            Predicted class (0 or 1)
        """
        weighted_sum = self.calculate_weighted_sum(inputs)
        return self.activation(weighted_sum)

    def train(
        self,
        training_data: list[tuple[list[float], int]],
        epochs: int,
    ) -> None:
        """
        Train the perceptron using the perceptron learning algorithm.

        The algorithm iterates through the training data multiple times (epochs),
        adjusting weights and bias based on prediction errors using the rule:
            weight_i = weight_i + learning_rate * error * input_i
            bias = bias + learning_rate * error

        Args:
            training_data: List of training examples, where each example is a tuple
                          containing (inputs, expected_output).
                          Example structure:
                          [
                              ([0.0, 0.0], 0),  # inputs=[0.0, 0.0], expected=0
                              ([0.0, 1.0], 1),  # inputs=[0.0, 1.0], expected=1
                              ([1.0, 0.0], 1),  # inputs=[1.0, 0.0], expected=1
                              ([1.0, 1.0], 1),  # inputs=[1.0, 1.0], expected=1
                          ]
                          This example represents an OR gate with 2 inputs.
            epochs: Maximum number of complete passes through the training data
        """
        for epoch in range(epochs):
            # Track the total number of errors in this epoch
            total_errors = 0

            # Process each training example
            for inputs, expected_output in training_data:
                # Make a prediction with current weights
                predicted_output = self.predict(inputs)

                # Calculate the error (difference between expected and predicted)
                # error will be: -1, 0, or 1
                error = expected_output - predicted_output

                # Update each weight using the perceptron learning rule
                # Only updates when error != 0 (i.e., when prediction is wrong)
                for index in range(len(self.weights)):
                    self.weights[index] += (
                        self.learning_rate
                        * error
                        * inputs[index]
                    )

                # Update the bias term
                # Bias acts like a weight for an input that's always 1
                self.bias += self.learning_rate * error

                # Accumulate total errors for this epoch
                total_errors += abs(error)

            # Print progress after each epoch
            print(
                f"Epoch {epoch + 1:02d}, "
                f"errors: {total_errors}, "
                f"weights: {self.weights}, "
                f"bias: {self.bias:.2f}"
            )

            # If no errors occurred, the data is perfectly classified - stop training
            if total_errors == 0:
                print("Training completed. total_errors = 0")
                break
