import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

# Ensure GPU/MPS is used if available
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

# Load tokenizer and model
model_path = "./bertweet_topic_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# Load test dataset
dataset = load_dataset("cardiffnlp/tweet_topic_single", split="test_2021")
label_names = dataset.features["label_name"].names

# Tokenize
def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize_fn, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# For saving raw text later
raw_texts = dataset["text"]
true_labels = dataset["label"]

# DataLoader
class TweetDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["label"]
        }

test_loader = DataLoader(TweetDataset(dataset), batch_size=32)

# Inference loop
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("F1 (macro):", f1_score(all_labels, all_preds, average="macro"))
print("F1 (weighted):", f1_score(all_labels, all_preds, average="weighted"))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=label_names))

# Save predictions to CSV
df = pd.DataFrame({
    "text": raw_texts,
    "true_label": [label_names[i] for i in true_labels],
    "predicted_label": [label_names[i] for i in all_preds]
})
df.to_csv("bert_predictions_test2021.csv", index=False)
print("Predictions saved to bert_predictions_test2021.csv")
