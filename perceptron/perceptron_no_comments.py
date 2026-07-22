import random

class Perceptron:
    def __init__(self, input_size: int, learning_rate: float = 0.1):
        self.weights = [
            random.uniform(-1.0, 1.0) for _ in range(input_size)
        ]
        self.bias = random.uniform(-1.0, 1.0)
        self.learning_rate = learning_rate

    def calculate_weighted_sum(self, inputs: list[float]) -> float:
        weighted_sum = sum(
            input_value * weight for input_value, weight in zip(inputs, self.weights)
        )
        return weighted_sum + self.bias

    def activation(self, weighted_sum: float) -> int:
        return 1 if weighted_sum >= 0 else 0

    def predict(self, inputs: list[float]) -> int:
        weighted_sum = self.calculate_weighted_sum(inputs)
        return self.activation(weighted_sum)

    def train(
        self,
        training_data: list[tuple[list[float], int]],
        epochs: int,
    ) -> None:
        for epoch in range(epochs):
            total_errors = 0

            for inputs, expected_output in training_data:
                predicted_output = self.predict(inputs)
                error = expected_output - predicted_output

                self.weights = [
                    weight + self.learning_rate * error * input_value
                    for weight, input_value in zip(self.weights, inputs)
                ]
                self.bias += self.learning_rate * error

                total_errors += abs(error)

            print(
                f"Epoch {epoch + 1:02d}, "
                f"errors: {total_errors}, "
                f"weights: {self.weights}, "
                f"bias: {self.bias:.2f}"
            )

            if total_errors == 0:
                print("Training completed. total_errors = 0")
                break