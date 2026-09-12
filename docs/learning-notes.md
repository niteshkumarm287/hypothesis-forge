# Learning Notes

These notes capture the concepts explored while building Hypothesis Forge V1.

## Logistic regression

Logistic regression is a machine-learning algorithm that learns which input features are
associated with each category. Unlike a hand-written rule such as:

```python
if "planned" in text:
    return "maintenance"
```

the model learns feature weights from the labeled examples in `data/cases.json`.

It was selected because it is fast on a Mac, effective for small text-classification problems,
supports multiple categories, and exposes influential features for inspection. It does not deeply
understand language and may struggle with sarcasm, negation, unseen vocabulary, long discussions,
or an unrepresentative dataset.

Alternatives include Naive Bayes, support vector machines, neural networks, and large language
models. Each introduces different tradeoffs in complexity, data requirements, cost, and
interpretability.

## Generalization and dataset boundaries

Training and holdout messages can express the same idea with different wording:

```text
Training: TLS certificates will be rotated
Holdout:  Network certificates are being renewed

Training: Incident review meeting
Holdout:  Incident retrospective notes

Training: Who owns the checkout service?
Holdout:  Share the ownership page
```

This tests whether the model learned broader relationships rather than exact sentences. Once a
holdout result influences model changes, it is no longer a truly untouched final exam.

## Accuracy saturation

V1 reaches 100% validation accuracy. That does not mean every correct model is equally good:

```text
Model A: correct-category probability = 51%
Model B: correct-category probability = 96%
```

Accuracy treats both predictions equally. A future version can add log loss, which rewards
correct, confident predictions and penalizes confident mistakes.

## Keep only with evidence

The Naive Bayes experiment matched the best validation accuracy but was discarded because it did
not demonstrate a meaningful improvement over logistic regression. Recording discarded ideas
prevents repeated work and keeps the research branch from accumulating unsupported changes.
