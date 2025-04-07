# Mid Evaluation Report – NLP Project

## Team Name: **BAM**

### Team Members:
- **Sai Bharadwaj Avvari** (2024901003)  
- **Aravind Iddamsetty** (2024901005)  
- **Mohan Kumar Kanaka** (2024901009)

---

## 📌 Brief Overview

We want to present both **like** and **opposite** views of people on Twitter.  
Based on a user's most recent tweets, our system aims to:

- Show tweets sharing **similar sentiments** on the same topic.
- Show tweets expressing **opposite sentiments** on the same topic.

In essence, we want to give users an overview of **both sides of the coin** for any tweet they post.

---

## 🔧 Breakdown of NLP Tasks

To achieve the above goal, we break the problem into the following subtasks:

1. **Classify** the user's last tweet into one of several **pre-defined categories** (learned from training data).
2. **Identify the sentiment** of the user's tweet.
3. From the dataset, **select tweets** in the **same category**.
4. From step 3:
   - Select tweets with **similar sentiment** and summarize them.
   - Select tweets with **opposite sentiment** and summarize them.
5. Present the **similar** and **opposite** summaries back to the user.

---

## 🧪 Step 1: Tweet Classification

We used the [`cardiffnlp/tweet_topic_single`](https://huggingface.co/datasets/cardiffnlp/tweet_topic_single) dataset for classification experiments.

### ✳️ ELMO for Classification

- Used **ELMO model** pre-trained on the **1 Billion Word Benchmark** (from Kaggle).
- Trained on the `train_2020` split of the dataset.
- Generated ELMO embeddings (see `elmo_prereq.py`).
- Trained an **MLP classifier** (1 hidden layer, 256 neurons) using these embeddings.
- Achieved **65% accuracy** on `test_2020`.

### ⚡ BERT for Classification

- Used the `vinai/bertweet-base` model.
- Fine-tuned on the `train_2020` split.
- Tested on the `test_2020` split.
- Achieved **89% accuracy**.

> 🧠 Note: We intentionally experimented with both ELMO and BERT for a **deeper understanding of course concepts**, even though BERT was expected to perform better. 😊

---

## 🔭 Future Work

- Implement **sentiment detection** within topic-aligned tweets.
- Summarize **similar** and **opposite** sentiment tweets.
- Present a **unified view** to the end user — both sides of the conversation.

---

