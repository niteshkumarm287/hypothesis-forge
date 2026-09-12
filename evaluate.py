"""
Evaluation script for the mini-autoresearch project.
"""

import json
from pathlib import Path

from candidate import predict

DATA_FILE = Path(__file__).parent / "data" / "cases.json"

def load_data():
    with open(DATA_FILE, "r") as file:
        return json.load(file)

def count_labels(examples):
    counts = {}
    for example in examples:
        label = example["label"]
        counts[label] = counts.get(label, 0) + 1
    return counts

def evaluate_classifier(examples):
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