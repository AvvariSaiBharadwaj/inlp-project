from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score

dataset = load_dataset("cardiffnlp/tweet_topic_single", split="test_2021")
texts = dataset["text"]
labels = dataset["label"]

model_dir = "./bertweet_topic_model"  # or wherever you saved it during training
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)

encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model.to(device)
encodings = {k: v.to(device) for k, v in encodings.items()}

with torch.no_grad():
    outputs = model(**encodings)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).cpu().numpy()

true_labels = labels
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score (weighted):", f1)
print("Classification Report:\n", classification_report(true_labels, predictions))
