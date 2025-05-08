# intent_classifier.py
import os
import json
import pickle
import torch
import pandas as pd
import spacy
import re
spacy.cli.download("en_core_web_sm")
from model import vocabulary, models, LSTMClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# Domain-specific stopwords
domain_stopwords = set([
    "flight", "flights", "airline", "airlines", "airport", "ticket", "tickets",
    "transportation", "service", "services", "travel", "arrival", "departures"
])

# Cities to preserve (multi-word cities handled later)
cities = ["san francisco", "new york", "los angeles", "las vegas", "philadelphia", "boston", "houston", "atlanta", "dallas", "pittsburgh"]

def clean_text(text):
    """
    Full cleaning pipeline:
    - Lowercasing
    - City name protection
    - Punctuation removal
    - Domain-specific stopwords removal
    - Lemmatization
    """
    # Lowercase
    text = text.lower()

    # Protect city names (merge with underscore before tokenizing)
    for city in cities:
        city_ = city.replace(" ", "_")
        text = text.replace(city, city_)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Remove domain-specific stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in domain_stopwords]

    # Lemmatize
    doc = nlp(" ".join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]

    # Final clean text
    cleaned_text = " ".join(lemmatized_tokens)

    return cleaned_text


def load_atis_data(file_path: str) -> tuple[list, list]:
    """
    Load ATIS dataset from a TSV file.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        tuple: List of texts and list of labels.
    """
    df = pd.read_csv(file_path, sep="\t", header=None, names=["text", "label"])
    return df["text"].tolist(), df["label"].tolist()


class IntentClassifier:
    """
    IntentClassifier to load, train, and manage models (ZeroShot, LSTM, Transformer).
    """

    def __init__(self):
        self._ready = False

    def load(
        self,
        file_path: str = "",
        train: bool = False,
        lstm_model_path: str = "",
        transformer_model_path: str = "",
        batch_size: int = 32
    ) -> None:
        """
        Load training data, vocabulary, models, and optionally train models.

        Args:
            file_path (str): Path to training data.
            train (bool): Whether to train models or load existing.
            lstm_model_path (str): Path to save/load LSTM model.
            transformer_model_path (str): Path to save/load Transformer model.
            batch_size (int): Batch size for LSTM training.
        """

        # Load texts and labels -> training the model
        texts, labels = load_atis_data(file_path)

        # Clean the data
        texts = list(map(clean_text, texts))

        # Paths for saving vocab and labels
        labels_path = os.path.join(lstm_model_path, "labels.json")
        vocab_path = os.path.join(lstm_model_path, "vocab.pkl")
        os.makedirs(lstm_model_path, exist_ok=True)

        # Build or Load Vocabulary
        if os.path.exists(vocab_path) and not train:
            with open(vocab_path, "rb") as f:
                vocab = pickle.load(f)
        else:
            vocab = vocabulary(min_freq=1)
            vocab.build_vocab(texts)
            with open(vocab_path, "wb") as f:
                pickle.dump(vocab, f)

        # Build Label Mappings
        label_set = sorted(set(labels))
        label2idx = {label: idx for idx, label in enumerate(label_set)}
        idx2label = {idx: label for label, idx in label2idx.items()}

        # Check label consistency
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                previous_label2idx = json.load(f)
                if previous_label2idx != label2idx:
                    train = True  # Force retraining if labels mismatch

        # Save current label mapping
        with open(labels_path, "w") as f:
            json.dump(label2idx, f)

        vocab_size = len(vocab.word2idx)
        num_classes = len(label2idx)

        # Instantiate model manager
        model_wrapper = models(texts, labels, vocab, label2idx, idx2label)

        # LSTM Model Setup
        lstm_model = LSTMClassifier(vocab_size, embed_dim=64, hidden_dim=64, num_classes=num_classes)
        lstm_model_file = os.path.join(lstm_model_path, "lstm_model.pt")

        # train if required
        if train:
            if os.path.exists(lstm_model_file) :
                lstm_model.load_state_dict(torch.load(lstm_model_file))
                lstm_model.to(DEVICE)
                #print("LSTM model loaded")
            else:
                lstm_model = model_wrapper.train_lstm(
                    lstm_model=lstm_model,
                    batch_size=batch_size,
                    model_dir=lstm_model_file
                )

            # Transformer Model Setup
            model_wrapper.train_transformer(model_dir=os.path.join(transformer_model_path, "bert_transformer"))

        self._ready = True

    def is_ready(self):
        return self._ready

    def predict(
            self,
            text: str,
            lstm_model_path: str,
            transformer_model_path: str,
            method: str = "lstm"
    ) -> str:
        """
        Predict intent for a given text.

        Args:
            text (str): Input text.
            lstm_model_path (str): Path to LSTM model directory (vocab and model).
            transformer_model_path (str): Path to Transformer model directory.
            method (str): Method to use ("lstm", "transformer", "zeroshot", "voting").

        Returns:
            str: Predicted intent label.
        """
        # CLean the texts
        text = clean_text(text)

        # Load Vocabulary
        vocab_path = os.path.join(lstm_model_path, "vocab.pkl")
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        # Load label mappings
        labels_path = os.path.join(lstm_model_path, "labels.json")
        with open(labels_path, "r") as f:
            label2idx = json.load(f)
        idx2label = {idx: label for label, idx in label2idx.items()}

        # Load LSTM Model
        lstm_model_file = os.path.join(lstm_model_path, "lstm_model.pt")
        vocab_size = len(vocab.word2idx)
        num_classes = len(label2idx)

        lstm_model = LSTMClassifier(vocab_size, embed_dim=64, hidden_dim=64, num_classes=num_classes)
        lstm_model.load_state_dict(torch.load(lstm_model_file))
        lstm_model.to(DEVICE)
        lstm_model.eval()

        # Create models object
        model_wrapper = models(["dummy", "dummy"], ["0","1"], vocab, label2idx, idx2label)  # just to initialise

        if method == "lstm":
            preds = model_wrapper.lstm_predict(text, vocab, idx2label, lstm_model, top_k=3)
            return preds

        elif method == "transformer":
            transformer_path = os.path.join(transformer_model_path, "bert_transformer")
            return model_wrapper.predict_transformer(model_dir=transformer_path, text=text)

        elif method == "zeroshot":
            candidate_labels = list(label2idx.keys())
            return model_wrapper.predict_zero_shot_classifier(text, candidate_labels)

        elif method == "voting":
            transformer_path = os.path.join(transformer_model_path, "bert_transformer")
            return model_wrapper.simple_voting_classifier(text, lstm_model, transformer_path)

        else:
            raise ValueError(f"Unknown method {method}. Choose 'lstm', 'transformer', 'zeroshot', or 'voting'.")

# --- CLI mode ---
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--file_path", type=str, default="data/atis/train.tsv", help="Path to ATIS training data")
#     parser.add_argument("--lstm_model_path", type=str, default="saved_model", help="Directory to save/load LSTM model")
#     parser.add_argument("--transformer_model_path", type=str, default="saved_model", help="Directory to save/load Transformer model")
#     parser.add_argument("--train", type=bool, default=True, help="Train models if True")
#     args = parser.parse_args()
#
#     classifier = IntentClassifier()
#     classifier.load(
#         file_path=args.file_path,
#         train=args.train,
#         lstm_model_path=args.lstm_model_path,
#         transformer_model_path=args.transformer_model_path,
#         batch_size=32
#     )
