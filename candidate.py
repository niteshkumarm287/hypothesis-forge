"""Current experimental message classifier."""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    # ("text_features", TfidfVectorizer()), # convert text into numerical word features
    ("text_features", CountVectorizer(stop_words="english")), # plain word-presence counts, no IDF weighting
    ("classifier", LogisticRegression()), # learns how those features relate to labels
])

def train(examples):
    texts = []
    labels = []

    for example in examples:
        texts.append(example["text"])
        labels.append(example["label"])

    model.fit(texts, labels) # ths line performs the learning

def predict(text):
    predictions = model.predict([text])
    return predictions[0]