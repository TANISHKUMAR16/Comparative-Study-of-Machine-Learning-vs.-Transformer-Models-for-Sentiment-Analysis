# =========================
# IMPORTS
# =========================
import pandas as pd
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# =========================
# LOAD IMDB DATASET
# =========================
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")

train_df = pd.DataFrame(dataset['train']).sample(5000, random_state=42)
test_df = pd.DataFrame(dataset['test']).sample(2000, random_state=42)

print("Dataset loaded!")

# =========================
# TOKENIZER
# =========================
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(
    train_df['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    test_df['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

# =========================
# DATASET CLASS
# =========================
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_df['label'].tolist())
test_dataset = Dataset(test_encodings, test_df['label'].tolist())

# =========================
# MODEL
# =========================
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# =========================
# TRAINING CONFIG
# =========================
training_args = TrainingArguments(
    output_dir='./results_transformer',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=100,
    save_strategy="no"
)

# =========================
# TRAIN
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

print("\nTraining DistilBERT...")
trainer.train()

# =========================
# EVALUATE ON IMDB
# =========================
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(axis=1)

acc_imdb = accuracy_score(test_df['label'], y_pred)
f1_imdb = f1_score(test_df['label'], y_pred)

print("\nDistilBERT (IMDB):")
print(f"Accuracy: {acc_imdb:.4f}")
print(f"F1 Score: {f1_imdb:.4f}")

# =========================
# LOAD TWITTER DATASET
# =========================
print("\nLoading Twitter dataset...")

cols = ['target', 'id', 'date', 'flag', 'user', 'text']
twitter_df = pd.read_csv(
    'training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    names=cols,
    engine='python'
)

twitter_df = twitter_df[['text', 'target']]
twitter_df['target'] = twitter_df['target'].map({0: 0, 4: 1})

twitter_df = twitter_df.sample(1000, random_state=42)

# =========================
# TOKENIZE TWITTER
# =========================
twitter_encodings = tokenizer(
    twitter_df['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

twitter_dataset = Dataset(
    twitter_encodings,
    twitter_df['target'].tolist()
)

# =========================
# EVALUATE ON TWITTER
# =========================
predictions = trainer.predict(twitter_dataset)
y_pred = predictions.predictions.argmax(axis=1)

acc_twitter = accuracy_score(twitter_df['target'], y_pred)
f1_twitter = f1_score(twitter_df['target'], y_pred)

print("\nDistilBERT (Twitter):")
print(f"Accuracy: {acc_twitter:.4f}")
print(f"F1 Score: {f1_twitter:.4f}")

# =========================
# SAVE RESULTS
# =========================
df = pd.DataFrame({
    "Dataset": ["IMDB", "Twitter"],
    "Accuracy": [acc_imdb, acc_twitter],
    "F1": [f1_imdb, f1_twitter]
})

df.to_csv("transformer_results.csv", index=False)

print("\nTransformer results saved")

# =========================
# GRAPH (IMDB vs TWITTER)
# =========================
labels = ["IMDB", "Twitter"]
accuracy_vals = [acc_imdb, acc_twitter]
f1_vals = [f1_imdb, f1_twitter]

x = range(len(labels))

plt.figure()
plt.bar(x, accuracy_vals, width=0.4, label="Accuracy")
plt.bar([i + 0.4 for i in x], f1_vals, width=0.4, label="F1 Score")

plt.xticks([i + 0.2 for i in x], labels)
plt.title("DistilBERT Performance (IMDB vs Twitter)")
plt.legend()
plt.tight_layout()
plt.savefig("transformer_comparison.png")

print("\nTransformer graph saved as transformer_comparison.png")

print("\n✅ TRANSFORMER PIPELINE COMPLETED")