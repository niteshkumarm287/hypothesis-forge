"""Current experimental message classifier."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    # ("text_features", TfidfVectorizer()), # convert text into numerical word features
    ("text_features", TfidfVectorizer(stop_words="english")), # Words such as the, is, be, and at are called stop words. They are common in English but often carry little classification information.
    ("classifier", LogisticRegression(max_iter=1000)), # learns how those features relate to labels
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