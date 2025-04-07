import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

# Load ELMo embeddings
train = np.load("train_2020_elmo_embeddings.npz")
val = np.load("validation_2020_elmo_embeddings.npz")
test = np.load("test_2020_elmo_embeddings.npz")

X_train, y_train = torch.tensor(train["X"]).float(), torch.tensor(train["y"])
X_val, y_val = torch.tensor(val["X"]).float(), torch.tensor(val["y"])
X_test, y_test = torch.tensor(test["X"]).float(), torch.tensor(test["y"])

num_classes = len(set(y_train.tolist()))

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

model = MLP(input_dim=1024, hidden_dim=256, output_dim=num_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_preds = torch.argmax(val_logits, dim=1)
        val_acc = (val_preds == y_val).float().mean().item()
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")

# Final evaluation
model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    test_preds = torch.argmax(test_logits, dim=1)
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_preds))
