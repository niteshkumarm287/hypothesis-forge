"""Display the words that most influence each model category."""

from candidate import model, train
from evaluate import load_data

def main():
    data = load_data()
    train(data["train"])

    vectorizer = model.named_steps["text_features"]
    classifier = model.named_steps["classifier"]

    words = vectorizer.get_feature_names_out()

    for category, weights in zip(classifier.classes_, classifier.coef_):
        strongest_positions = weights.argsort()[-8:][::-1]

        print()
        print(f"Strongest words for category '{category}':")

        for position in strongest_positions:
            word = words[position]
            weight = weights[position]
            print(f" {word:<15} {weight:.3f}")

if __name__ == "__main__":
    main()