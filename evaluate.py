"""Train and evaluate the Hypothesis Forge candidate model."""

import json
from pathlib import Path

from candidate import predict, train

DATA_FILE = Path(__file__).parent / "data" / "cases.json"


def load_data():
    """Load the fixed training and validation datasets."""
    with DATA_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def count_labels(examples):
    """Count how many examples belong to each category."""
    counts = {}
    for example in examples:
        label = example["label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def evaluate_classifier(examples):
    """Evaluate the trained candidate and return its correct predictions."""
    correct_count = 0

    for example in examples:
        text = example["text"]
        expected_label = example["label"]
        predicted_label = predict(text)

        is_correct = predicted_label == expected_label

        if is_correct:
            status = "PASS"
            correct_count += 1
        else:
            status = "FAIL"

        print(
            f"[{status}] expected={expected_label}, "
            f"predicted={predicted_label}"
        )

    return correct_count


def main():
    data = load_data()
    training_examples = data["train"]
    validation_examples = data["validation"]

    print()
    print("Training model...")
    train(training_examples)

    print(f"Training examples: {len(training_examples)}")
    print(f"Validation examples: {len(validation_examples)}")
    print(f"Training labels: {count_labels(training_examples)}")
    print(f"Validation labels: {count_labels(validation_examples)}")

    print("Validation results")
    print("-" * 50)

    correct_count = evaluate_classifier(validation_examples)
    total_count = len(validation_examples)
    accuracy = correct_count / total_count * 100

    print()
    print("Summary")
    print(f"Correct: {correct_count}/{total_count}")
    print(f"Accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    main()
