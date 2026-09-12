"""Train and run the current operational-message classifier."""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline(
    [
        ("text_features", CountVectorizer(stop_words="english")),
        ("classifier", LogisticRegression()),
    ]
)


def train(examples: list[dict[str, str]]) -> None:
    """Fit the classifier using labeled training examples."""
    texts = []
    labels = []

    for example in examples:
        texts.append(example["text"])
        labels.append(example["label"])

    model.fit(texts, labels)


def predict(text: str) -> str:
    """Predict the category for one operational message."""
    predictions = model.predict([text])
    return str(predictions[0])
