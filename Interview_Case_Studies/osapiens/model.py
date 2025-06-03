import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report,  accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # get the device

class BERTLSTMClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_dim=1028, num_classes=3, dropout=0.3, bidirectional=True, idx2label=None):
        super(BERTLSTMClassifier, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

        bert_output_dim = self.bert.config.hidden_size
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.lstm = nn.LSTM(
            input_size=bert_output_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, num_classes)
        )

        self.idx2label = idx2label

    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
        tokens = {key: val.to(DEVICE) for key, val in tokens.items()}

        with torch.no_grad():
            outputs = self.bert(**tokens)
            bert_embeddings = outputs.last_hidden_state

        lstm_out, _ = self.lstm(bert_embeddings)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden)

    def predict(self, texts, top_k=3):
        self.eval()
        if isinstance(texts, str):
            texts = [texts]
        with torch.no_grad():
            logits = self.forward(texts)
            probs = torch.softmax(logits, dim=1)

        top_probs, top_indices = torch.topk(probs, k=top_k)
        predictions = []
        for i in range(len(texts)):
            pred = []
            total = top_probs[i].sum().item()
            for prob, idx in zip(top_probs[i], top_indices[i]):
                label = self.idx2label[idx.item()] if self.idx2label else idx.item()
                pred.append({"label": label, "confidence": round(prob.item() / total, 4)})
            predictions.append(pred)
        return predictions if len(predictions) > 1 else predictions[0]

    def train_model(self, train_texts, train_labels, epochs=3, batch_size=32, learning_rate=1e-4):
        """
        Trains the model using raw texts and label lists (no DataLoader).
        Includes class weighting and F1-score tracking.
        """
        self.train()

        # Compute class weights
        classes = np.unique(train_labels)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
        weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        label_tensor = torch.tensor(train_labels, dtype=torch.long).to(DEVICE)

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            y_true_epoch = []
            y_pred_epoch = []

            for i in range(0, len(train_texts), batch_size):
                batch_texts = train_texts[i:i + batch_size]
                batch_labels = label_tensor[i:i + batch_size]

                outputs = self.forward(batch_texts)
                loss = criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                y_true_epoch.extend(batch_labels.tolist())
                y_pred_epoch.extend(preds.tolist())

            avg_loss = total_loss / (len(train_texts) // batch_size + 1)
            f1 = f1_score(y_true_epoch, y_pred_epoch, average='macro')
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - F1 (macro): {f1:.4f}")



class TransformerClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=3, idx2label=None):
        super(TransformerClassifier, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )
        self.idx2label = idx2label

    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        return self.transformer(**tokens).logits

    def predict(self, texts, top_k=3):
        self.eval()
        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            logits = self.forward(texts)
            probs = torch.softmax(logits, dim=1)

        top_probs, top_indices = torch.topk(probs, k=top_k)
        predictions = []
        for i in range(len(texts)):
            pred = []
            total = top_probs[i].sum().item()
            for prob, idx in zip(top_probs[i], top_indices[i]):
                label = self.idx2label[idx.item()] if self.idx2label else idx.item()
                pred.append({"label": label, "confidence": round(prob.item() / total, 4)})
            predictions.append(pred)

        return predictions if len(predictions) > 1 else predictions[0]

    def train_model(self, train_texts, train_labels, epochs=3, batch_size=32, learning_rate=2e-5):
        self.train()

        # Class weights
        classes = np.unique(train_labels)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
        weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        label_tensor = torch.tensor(train_labels, dtype=torch.long).to(DEVICE)

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            y_true_epoch = []
            y_pred_epoch = []

            for i in range(0, len(train_texts), batch_size):
                batch_texts = train_texts[i:i + batch_size]
                batch_labels = label_tensor[i:i + batch_size]

                logits = self.forward(batch_texts)
                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                y_true_epoch.extend(batch_labels.tolist())
                y_pred_epoch.extend(preds.tolist())

            avg_loss = total_loss / (len(train_texts) // batch_size + 1)
            f1 = f1_score(y_true_epoch, y_pred_epoch, average='macro')
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - F1 (macro): {f1:.4f}")

def evaluate_classification_metrics_to_df(y_true, y_pred, y_score=None, class_labels=None, plot_auc=False):
    """
    Evaluate metrics including class-wise AUC and handles NaNs safely.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, roc_curve, auc
    )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if class_labels is None:
        class_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    metrics_data = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision (Macro)': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Precision (Weighted)': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall (Macro)': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall (Weighted)': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score (Macro)': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'F1 Score (Weighted)': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    # AUC Calculation
    if y_score is not None:
        y_true_bin = label_binarize(y_true, classes=list(range(len(class_labels))))
        auc_scores = []
        roc_auc_per_class = {}

        for i in range(len(class_labels)):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                auc_scores.append(roc_auc)
                roc_auc_per_class[class_labels[i]] = roc_auc
            except ValueError:
                auc_scores.append(np.nan)
                roc_auc_per_class[class_labels[i]] = np.nan

        auc_macro = np.nanmean(auc_scores)
        auc_micro = roc_auc_score(y_true_bin, y_score, average='micro', multi_class='ovr')

        metrics_data['AUC (Macro)'] = auc_macro
        metrics_data['AUC (Micro)'] = auc_micro

        if plot_auc:
            plt.figure(figsize=(8, 6))
            for i in range(len(class_labels)):
                if not np.isnan(roc_auc_per_class[class_labels[i]]):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                    plt.plot(fpr, tpr, label=f"{class_labels[i]} (AUC={roc_auc_per_class[class_labels[i]]:.2f})")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves (One-vs-Rest)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    return pd.DataFrame([metrics_data])

def plot_single_label_roc(y_true, y_probs, label_name, class_names):
    """
    Plot ROC curve for a single label.
    
    Parameters:
    - y_true: array-like (n_samples, n_classes)
    - y_probs: array-like (n_samples, n_classes)
    - label_name: string, name of the label to plot
    - class_names: list of strings, class names in the same order as columns
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    if label_name not in class_names:
        raise ValueError(f"Label '{label_name}' not found in class_names.")
    
    i = class_names.index(label_name)
    
    fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
    auc_score = roc_auc_score(y_true[:, i], y_probs[:, i])

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'{label_name} (AUC = {auc_score:.2f})', color='green')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for '{label_name}'")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


