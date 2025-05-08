# import libs
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
from datasets import Dataset as HFDataset
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, \
    AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import logging
from transformers import pipeline

logging.set_verbosity_error()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set up device

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_predictions(y_true, y_pred, label_set, model_name="Model"):
    """
    Evaluates predictions and returns two DataFrames:
    1. overall_df - overall accuracy, AUC, macro-F1
    2. classwise_df - class-wise precision, recall, f1, support
    """

    # Classification Report
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    # Binarize for ROC-AUC calculation
    y_true_bin = label_binarize(y_true, classes=label_set)
    y_pred_bin = label_binarize(y_pred, classes=label_set)

    try:
        auc = roc_auc_score(y_true_bin, y_pred_bin, average='macro', multi_class='ovr')
    except Exception as e:
        print(f"AUC couldn't be calculated properly: {e}")
        auc = None

    # Overall Metrics
    overall_metrics = {
        "Model": [model_name],
        "Accuracy": [acc],
        #"AUC": [auc],
        "Macro_F1": [report_dict['macro avg']['f1-score']],
        "Macro_Precision": [report_dict['macro avg']['precision']],
        "Macro_Recall": [report_dict['macro avg']['recall']]
    }
    overall_df = pd.DataFrame(overall_metrics)

    # Class-wise Metrics
    classwise_metrics = []
    for label in label_set:
        if label in report_dict:
            classwise_metrics.append({
                "Model": model_name,
                "Class": label,
                "Precision": report_dict[label]['precision'],
                "Recall": report_dict[label]['recall'],
                "F1-Score": report_dict[label]['f1-score'],
                # "Support": report_dict[label]['support']
            })

    classwise_df = pd.DataFrame(classwise_metrics)
    return overall_df, classwise_df


class vocabulary:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build_vocab(self, texts):
        counter = Counter(word for text in texts for word in text.split())
        for word, freq in counter.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text):
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in text.split()]

    def decode(self, indices):
        return [self.idx2word.get(idx, "<UNK>") for idx in indices]


class IntentDataset(torch_data.Dataset):
    def __init__(self, texts, labels, vocab, label2idx):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label2idx = label2idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            # idx is a batch of indices
            texts = [self.texts[i] for i in idx]
            labels = [self.label2idx[self.labels[i]] for i in idx]
            encoded = [torch.tensor(self.vocab.encode(t)) for t in texts]
            return encoded, labels
        else:
            # idx is a single index
            encoded = self.vocab.encode(self.texts[idx])
            label = self.label2idx[self.labels[idx]]
            return torch.tensor(encoded), label


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / len(dataloader), correct / total


def collate_batch(batch):
    texts, labels = zip(*batch)
    lengths = [len(t) for t in texts]
    max_len = max(lengths)
    padded = [F.pad(t, (0, max_len - len(t)), value=0) for t in texts]
    return torch.stack(padded), torch.tensor(labels)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.fc(hidden)
        return out


class models:
    """
    Main model management class.
    Provides methods to train and predict using:
    - Zero-shot classification
    - LSTM model
    - Transformer model
    - Random Forest model
    """

    def __init__(self, texts, labels, vocab, label2idx, idx2label):
        """
        Initialize Models class by splitting data into training and validation.
        Args:
            texts (list): List of text samples.
            labels (list): List of labels corresponding to texts.
            vocab (Vocabulary): Vocabulary object for encoding.
            label2idx (dict): Mapping from label to index.
            idx2label (dict): Mapping from index to label.
        """
        self.train_texts, self.val_texts, self.train_labels, self.val_labels = train_test_split(
            texts, labels, test_size=0.15, random_state=42
        )
        self.vocab = vocab
        self.label2idx = label2idx
        self.idx2label = idx2label

    @staticmethod
    def predict_zero_shot_classifier(texts: list, candidate_labels: list, top_k = 3) -> list:
        """
        Perform zero-shot classification using a pretrained model.
        Supports batch prediction for speed improvement.

        Args:
            texts (list[str]): List of input texts.
            candidate_labels (list[str]): List of possible labels.

        Returns:
            list[list[dict]]: Each entry in the outer list corresponds to one input text.
                              Each inner list contains top 3 dicts with label and confidence.
        """
        if not isinstance(texts, list):
            texts = [texts]

        zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1,
            batch_size=8
        )

        results = zero_shot_classifier(
            texts,
            candidate_labels,
            multi_label=True
        )

        outputs = []
        if isinstance(results, list):
            for result in results:
                scores = result['scores'][:top_k]
                labels = result['labels'][:top_k]
                total_score = sum(scores)
                top_preds = [
                    {"label": label, "confidence": round(score / total_score, 4)}
                    for label, score in zip(labels, scores)
                ]
                outputs.append(top_preds)
        else:
            scores = results['scores'][:top_k]
            labels = results['labels'][:top_k]
            total_score = sum(scores)
            top_preds = [
                {"label": label, "confidence": round(score / total_score, 4)}
                for label, score in zip(labels, scores)
            ]
            outputs.append(top_preds)

        return outputs

    def train_lstm(self, lstm_model, model_dir="", batch_size=32, epochs=10):
        """
        Train an LSTM model with class imbalance handling.
        Args:
            lstm_model (nn.Module): Initialized LSTM model.
            model_dir (str): Path to save the model.
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs.
        Returns:
            Trained LSTM model.
        """
        # print("Initializing LSTM model...")
        model = lstm_model.to(DEVICE)

        # Compute class weights
        classes = np.unique(self.train_labels)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=self.train_labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

        # Prepare datasets and loaders
        train_dataset = IntentDataset(self.train_texts, self.train_labels, self.vocab, self.label2idx)
        val_dataset = IntentDataset(self.val_texts, self.val_labels, self.vocab, self.label2idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

        # print("Starting LSTM training...")
        best_val_acc = 0.0
        for epoch in range(1, epochs + 1):
            train_loss = train(model, train_loader, optimizer, criterion)
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_dir)
        # print("LSTM Model training complete and saved!")
        return model

    def lstm_predict(self, text, vocab, idx2label, model, top_k=3):
        """
        Predict using a trained LSTM model.
        Args:
            text (str): Input text.
            vocab (Vocabulary): Vocabulary object.
            idx2label (dict): Mapping from index to label.
            model (nn.Module): Trained LSTM model.
            top_k (int): Number of top predictions to return.
        Returns:
            list: Top-k predicted labels and their normalized confidence scores.
        """
        encoded = torch.tensor([vocab.encode(text)], device=DEVICE)
        with torch.no_grad():
            logits = model(encoded)
            probs = F.softmax(logits, dim=1).squeeze()
            top_probs, top_indices = torch.topk(probs, top_k)

        total = top_probs.sum().item()
        return [
            {"label": idx2label[idx.item()], "confidence": round(prob.item() / total, 4)}
            for prob, idx in zip(top_probs, top_indices)
        ]

    def train_transformer(self, model_name=None, model_dir="transformer_model"):
        """
        Train a Transformer model for text classification.
        Args:
            model_name (str): Pretrained model name.
            model_dir (str): Directory to save trained model.
        """
        if model_name is None:
            model_name = "bert-base-uncased"

        # print("Initializing Transformer model...")

        # Map labels to integers
        train_encodings = [{"text": text, "label": self.label2idx[label]} for text, label in
                           zip(self.train_texts, self.train_labels)]
        val_encodings = [{"text": text, "label": self.label2idx[label]} for text, label in
                         zip(self.val_texts, self.val_labels)]

        # Create Datasets
        train_dataset = HFDataset.from_list(train_encodings)
        val_dataset = HFDataset.from_list(val_encodings)

        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.transformer_model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=len(self.label2idx)
        )

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=32)

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        # Prepare dataset
        train_dataset = train_dataset.rename_column("label", "labels")
        val_dataset = val_dataset.rename_column("label", "labels")
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Training setup
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        training_args = TrainingArguments(
            output_dir=model_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=1,
            fp16=True,
            weight_decay=0.01,
            save_strategy="no",
            logging_steps=10
        )

        trainer = Trainer(
            model=self.transformer_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # print("Starting Transformer training...")
        trainer.train()

        # print(f"Saving Transformer model to {model_dir}...")
        self.transformer_model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        # print("Transformer Model saved successfully.")

    def predict_transformer(self, model_dir, text, top_k = 3):
        """
        Predict using a trained Transformer model.
        Args:
            model_dir (str): Directory where model is saved.
            text (str): Input text to predict.
        Returns:
            list: Top-3 predicted labels with normalized confidence scores.
        """
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(DEVICE)
        model.eval()

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze()
            top_probs, top_indices = torch.topk(probs, k=top_k)

        total = top_probs.sum().item()
        return [
            {"label": self.idx2label[idx.item()], "confidence": round(prob.item() / total, 4)}
            for prob, idx in zip(top_probs, top_indices)
        ]

    def simple_voting_classifier(self, text, lstm_model, transformer_model_path,top_k =3):
        """
        Simple Voting Classifier combining Transformer, LSTM, and ZeroShot outputs.
        Args:
            text (str): Input text to predict.
            lstm_model (nn.Module): Trained LSTM model.
            transformer_model_path (str): Path where transformer model is saved.
        Returns:
            str: Final predicted label after voting.
        """

        # --- Transformer output ---
        tokenizer = BertTokenizer.from_pretrained(transformer_model_path)
        transformer_model = AutoModelForSequenceClassification.from_pretrained(transformer_model_path)
        transformer_model.to(DEVICE)
        transformer_model.eval()

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)

        with torch.no_grad():
            outputs = transformer_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        # Normalize the probabilities to sum to 1
        total_prob = probs.sum()
        normalized_probs = probs / total_prob if total_prob > 0 else probs  # Avoid division by zero

        # Convert to label-confidence mapping
        idx2label = {int(k): v for k, v in self.idx2label.items()}
        transformer_label_scores = {
            idx2label[i]: normalized_probs[i] for i in range(len(normalized_probs))
        }

        # --- Zero-Shot output ---
        zero_shot_preds  = self.predict_zero_shot_classifier(text, list(self.label2idx.keys()))
        zero_shot_label_scores = {
            entry["label"]: entry["confidence"]
            for entry in zero_shot_preds[0]
        }

        # --- LSTM output ---
        encoded = torch.tensor([self.vocab.encode(text)], device=DEVICE)
        with torch.no_grad():
            logits = lstm_model(encoded)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        # Normalize the probabilities
        total_prob = probs.sum()
        normalized_probs = probs / total_prob if total_prob > 0 else probs  # Avoid division by zero

        # Convert to label-confidence mapping
        lstm_label_scores = {
            self.idx2label[i]: normalized_probs[i] for i in range(len(normalized_probs))
        }

        # --- Combine all model scores ---
        total_scores = {}
        for label in self.label2idx.keys():
            total_scores[label] = (
                    float(transformer_label_scores.get(label, 0.0)) +
                    float(zero_shot_label_scores.get(label, 0.0)) +
                    float(lstm_label_scores.get(label, 0.0))
            )

        # Normalize the total scores
        # Step 1: Sort and select top-k
        sorted_labels = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Step 2: Normalize top-k scores
        score_sum = sum(score for _, score in sorted_labels)
        return [
            {"label": label, "confidence": round(score / score_sum, 2) if score_sum > 0 else 0.0}
            for label, score in sorted_labels
        ]


