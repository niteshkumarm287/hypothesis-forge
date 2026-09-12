"""Run the trained classifier against the holdout test set."""

import json
from pathlib import Path

from candidate import train
from evaluate import evaluate_classifier, load_data

TEST_FILE = Path(__file__).parent / "data" / "test_cases.json"


def load_test_data():
    """Load the holdout test examples."""
    with TEST_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def main():
    data = load_data()

    training_examples = data["train"]
    test_examples = load_test_data()

    print("Training model...")
    train(training_examples)

    print()
    print("Holdout test results")
    print("-" * 50)

    correct_count = evaluate_classifier(test_examples)
    total_count = len(test_examples)
    accuracy = correct_count / total_count * 100

    print()
    print("Holdout summary")
    print(f"Correct: {correct_count}/{total_count}")
    print(f"Accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    main()
