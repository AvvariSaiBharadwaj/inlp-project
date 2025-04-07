import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import classification_report

def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS (Apple GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU)")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

dataset = load_dataset("cardiffnlp/tweet_topic_single")
train_dataset = dataset["train_2020"]
test_dataset = dataset["test_2020"]

model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, normalization=True)

label_list = train_dataset.features["label"].names
num_labels = len(label_list)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

encoded_train = train_dataset.map(preprocess_function, batched=True)
encoded_test = test_dataset.map(preprocess_function, batched=True)

encoded_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
encoded_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == labels)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_test,
    compute_metrics=compute_metrics,
)

device = get_device()
model.to(device)

print("\nStarting training... This may take a while... or forever if you forget to hydrate.")
trainer.train()

print("\nEvaluating model...")
eval_results = trainer.evaluate()
print(eval_results)

preds = trainer.predict(encoded_test)
pred_labels = np.argmax(preds.predictions, axis=1)
true_labels = preds.label_ids
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=label_list))

model.save_pretrained('./bertweet_topic_model')
tokenizer.save_pretrained('./bertweet_topic_model')

