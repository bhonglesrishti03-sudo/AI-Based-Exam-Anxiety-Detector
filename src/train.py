# Cell 1
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from torch.optim import AdamW
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# Tokenizer
MAX_LEN = 128
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset class
class AnxietyDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Map string labels to integers
        self.label2id = {label: idx for idx, label in enumerate(sorted(self.df['status'].unique()))}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.loc[idx, "text"])
        label_str = self.df.loc[idx, "status"]
        label = self.label2id[label_str]  # map string to integer

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Load dataset
DATA_PATH = "/content/drive/MyDrive/AI-Anxiety-Data/mental_health_combined_test.csv"
dataset = AnxietyDataset(DATA_PATH, tokenizer, MAX_LEN)

# Train/validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)


# Cell 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

num_classes = len(dataset.label2id)  # number of unique labels

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)


# Cell 3
EPOCHS = 2

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")


# Cell 4
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print("Validation Accuracy:", correct / total)


# Instead of saving to Drive
MODEL_PATH = "model/bert_anxiety_model.pt"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")