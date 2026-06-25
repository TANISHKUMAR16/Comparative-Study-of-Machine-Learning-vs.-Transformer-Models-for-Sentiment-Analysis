# =========================
# IMPORTS
# =========================
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# =========================
# CLEAN FUNCTION
# =========================
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# =========================
# LOAD IMDB
# =========================
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")

train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print("Cleaning text...")
train_df['text'] = train_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)

# =========================
# TF-IDF
# =========================
print("Vectorizing...")
vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

y_train = train_df['label']
y_test = test_df['label']

# =========================
# MODELS
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

imdb_results = {}

# =========================
# TRAIN + EVALUATE (IMDB)
# =========================
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} F1 Score: {f1:.4f}")

    imdb_results[name] = {"Accuracy": acc, "F1": f1}

# =========================
# IMDB GRAPH
# =========================
labels = list(imdb_results.keys())
acc_vals = [imdb_results[m]["Accuracy"] for m in labels]
f1_vals = [imdb_results[m]["F1"] for m in labels]

x = range(len(labels))

plt.figure()
plt.bar(x, acc_vals, width=0.4, label='Accuracy')
plt.bar([i + 0.4 for i in x], f1_vals, width=0.4, label='F1 Score')

plt.xticks([i + 0.2 for i in x], labels, rotation=30)
plt.title("IMDB Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("imdb_comparison.png")

print("\nIMDB graph saved")

# =========================
# TWITTER DATA
# =========================
print("\n------ ENTERING TWITTER SECTION ------")

cols = ['target', 'id', 'date', 'flag', 'user', 'text']
twitter_df = pd.read_csv(
    'training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    names=cols,
    engine='python'
)

twitter_df = twitter_df[['text', 'target']]
twitter_df['target'] = twitter_df['target'].map({0: 0, 4: 1})

twitter_df = twitter_df.sample(3000, random_state=42)

print("Cleaning Twitter data...")
twitter_df['text'] = twitter_df['text'].apply(clean_text)

X_twitter = vectorizer.transform(twitter_df['text'])
y_twitter = twitter_df['target']

# =========================
# TEST ON TWITTER
# =========================
twitter_results = {}

for name, model in models.items():
    print(f"\nTesting {name} on Twitter...")

    y_pred = model.predict(X_twitter)

    acc = accuracy_score(y_twitter, y_pred)
    f1 = f1_score(y_twitter, y_pred)

    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} F1 Score: {f1:.4f}")

    twitter_results[name] = {"Accuracy": acc, "F1": f1}

# =========================
# TWITTER GRAPH
# =========================
labels = list(twitter_results.keys())
acc_vals = [twitter_results[m]["Accuracy"] for m in labels]
f1_vals = [twitter_results[m]["F1"] for m in labels]

x = range(len(labels))

plt.figure()
plt.bar(x, acc_vals, width=0.4, label='Accuracy')
plt.bar([i + 0.4 for i in x], f1_vals, width=0.4, label='F1 Score')

plt.xticks([i + 0.2 for i in x], labels, rotation=30)
plt.title("Twitter Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("twitter_comparison.png")

print("\nTwitter graph saved")

print("\n✅ FINAL PIPELINE COMPLETED SUCCESSFULLY")