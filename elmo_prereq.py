import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from datasets import load_dataset
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logs

# Load ELMo model from TF Hub
elmo = hub.load("https://tfhub.dev/google/elmo/3")

# Load dataset
dataset = load_dataset("cardiffnlp/tweet_topic_single")

splits = ["train_2020", "validation_2020", "test_2020"]

def elmo_embed(texts):
    embeddings = elmo.signatures["default"](tf.constant(texts))["elmo"]
    return embeddings.numpy().mean(axis=1)  # mean-pooled sentence embeddings

for split in splits:
    print(f"Processing {split}...")
    texts = dataset[split]["text"]
    labels = dataset[split]["label"]

    # Batch processing
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = elmo_embed(batch)
        all_embeddings.append(emb)
    all_embeddings = np.vstack(all_embeddings)

    # Save to disk
    np.savez_compressed(f"{split}_elmo_embeddings.npz", X=all_embeddings, y=np.array(labels))

print("✅ All embeddings saved.")
