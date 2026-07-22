"""
Example usage of the Perceptron class.

Each example loads its dataset from CSV, trains a perceptron, and reports
accuracy on a separate test set the model never saw during training.
"""

from data_loader import TrainingExample, get_input_size, load_dataset
from perceptron import Perceptron


def calculate_accuracy(perceptron: Perceptron, test_data: list[TrainingExample]) -> float:
    """
    Calculate the percentage of correct predictions on a test set.

    Args:
        perceptron: A trained perceptron.
        test_data: Examples the perceptron was not trained on.

    Returns:
        Accuracy as a percentage between 0.0 and 100.0.

    Raises:
        ValueError: If test_data is empty.
    """
    if not test_data:
        raise ValueError("cannot calculate accuracy on an empty test set")

    correct = sum(
        1 for inputs, expected in test_data if perceptron.predict(inputs) == expected
    )
    return correct / len(test_data) * 100


def run_example(
    title: str,
    train_file: str,
    test_file: str,
    epochs: int,
    learning_rate: float = 0.1,
) -> float:
    """
    Train a perceptron on one dataset and report its accuracy on another.

    Args:
        title: Heading printed above the results.
        train_file: Path to the training CSV.
        test_file: Path to the test CSV.
        epochs: Maximum number of passes through the training data.
        learning_rate: Step size for weight updates.

    Returns:
        Accuracy on the test set, so callers can summarise across examples.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    training_data = load_dataset(train_file)
    test_data = load_dataset(test_file)

    perceptron = Perceptron(
        input_size=get_input_size(training_data),
        learning_rate=learning_rate,
    )
    perceptron.train(training_data=training_data, epochs=epochs)

    print("\nTesting on unseen data:")
    for inputs, expected in test_data:
        prediction = perceptron.predict(inputs)
        status = "✓" if prediction == expected else "✗"
        print(f"  {status} {inputs} -> predicted: {prediction}, expected: {expected}")

    accuracy = calculate_accuracy(perceptron, test_data)
    print(f"\n>>> Prediction Accuracy: {accuracy:.2f}%")
    return accuracy


def main() -> None:
    """Run every example and print a summary of the results."""
    results = {
        "OR Gate": run_example(
            title="Example 1: OR Gate",
            train_file="data/or_gate_train.csv",
            test_file="data/or_gate_test.csv",
            epochs=40,
        ),
        "AND Gate": run_example(
            title="Example 2: AND Gate",
            train_file="data/and_gate_train.csv",
            test_file="data/and_gate_test.csv",
            epochs=50,
        ),
        "Threshold Sum": run_example(
            title="Example 3: Threshold sum with 3 inputs",
            train_file="data/threshold_sum_train.csv",
            test_file="data/threshold_sum_test.csv",
            epochs=20,
        ),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, accuracy in results.items():
        print(f"  {name:<20} {accuracy:6.2f}%")


if __name__ == "__main__":
    main()