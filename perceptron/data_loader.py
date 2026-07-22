"""
Loading training and test data for the perceptron from CSV files.

Expected CSV format: a header row, then one example per line.
All columns except the last one are input features; the last column is the label.
"""

import csv
from pathlib import Path

TrainingExample = tuple[list[float], int]


def load_dataset(file_path: str | Path) -> list[TrainingExample]:
    """
    Load a dataset from a CSV file.

    Args:
        file_path: Path to the CSV file. The last column is treated as the label.

    Returns:
        A list of (features, label) tuples ready to feed into Perceptron.train().

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty, has no feature columns, or contains
                    a row whose length does not match the header.
    """
    path = Path(file_path)

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)

        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path} is empty")

        if len(header) < 2:
            raise ValueError(
                f"{path} needs at least one feature column and one label column, "
                f"found columns: {header}"
            )

        dataset: list[TrainingExample] = []

        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue

            if len(row) != len(header):
                raise ValueError(
                    f"{path} line {line_number}: expected {len(header)} values, "
                    f"got {len(row)}"
                )

            *feature_strings, label_string = row
            features = [float(value) for value in feature_strings]
            dataset.append((features, int(label_string)))

    if not dataset:
        raise ValueError(f"{path} contains a header but no data rows")

    return dataset


def get_input_size(dataset: list[TrainingExample]) -> int:
    """
    Determine how many input features a dataset has.

    Useful for constructing a Perceptron without hardcoding the input size.

    Args:
        dataset: A dataset as returned by load_dataset().

    Returns:
        The number of features in the first example.
    """
    return len(dataset[0][0])