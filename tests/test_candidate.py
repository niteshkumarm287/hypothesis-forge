import unittest
from collections import Counter

from candidate import predict, train
from evaluate import count_labels, evaluate_classifier, load_data


class ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        train(cls.data["train"])

    def test_validation_regression(self):
        validation = self.data["validation"]
        self.assertTrue(validation)
        self.assertEqual(evaluate_classifier(validation), len(validation))

    def test_prediction_contract(self):
        self.assertIn(
            predict("database unavailable"), {"incident", "maintenance", "non_incident"}
        )

    def test_training_and_validation_are_disjoint(self):
        training = {item["text"] for item in self.data["train"]}
        validation = {item["text"] for item in self.data["validation"]}
        self.assertFalse(training & validation)

    def test_label_counts(self):
        examples = self.data["train"]
        self.assertEqual(
            count_labels(examples), dict(Counter(item["label"] for item in examples))
        )
