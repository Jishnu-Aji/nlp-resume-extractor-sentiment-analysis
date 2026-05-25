# run_sentiment.py
"""
Sentiment Classification Pipeline
Runs the full workflow originally defined in the Jupyter notebook.
It loads the CSV dataset, preprocesses text, vectorizes, trains three models,
 evaluates them, creates visualizations, and saves the models and plots.
"""

import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from textblob import TextBlob
import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "dataset", "sentiment_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure output directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
# Clean column names
df.columns = [c.strip() for c in df.columns]
# Expect columns: text, sentiment
if "text" not in df.columns or "sentiment" not in df.columns:
    raise ValueError("Dataset must contain 'text' and 'sentiment' columns")

# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\n\r]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

df["clean_text"] = df["text"].apply(preprocess)
# Encode sentiment labels to integers
label_mapping = {label: idx for idx, label in enumerate(df["sentiment"].unique())}
df["label"] = df["sentiment"].map(label_mapping)

X_text = df["clean_text"]
y = df["label"]

# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------
vectorizer = CountVectorizer(stop_words="english")
X_counts = vectorizer.fit_transform(X_text)
transformer = TfidfTransformer()
X_tfidf = transformer.fit_transform(X_counts)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
print("Training Naive Bayes...")
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)

print("Training Linear SVM...")
svm_clf = LinearSVC()
svm_clf.fit(X_train, y_train)

print("Training Random Forest...")
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_clf.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    print(f"--- {name} ---")
    print(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    return {"name": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion": cm}

results = []
results.append(evaluate("Naive Bayes", nb_clf, X_test, y_test))
results.append(evaluate("Linear SVM", svm_clf, X_test, y_test))
results.append(evaluate("Random Forest", rf_clf, X_test, y_test))

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
# Accuracy bar chart
plt.figure(figsize=(6, 4))
names = [r["name"] for r in results]
accs = [r["accuracy"] for r in results]
sns.barplot(x=names, y=accs, palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for i, v in enumerate(accs):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
acc_path = os.path.join(OUTPUT_DIR, "accuracy_comparison.png")
plt.savefig(acc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved accuracy chart to {acc_path}")

# Confusion matrices heatmaps
for r in results:
    cm = r["confusion"]
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {r['name']}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(
        OUTPUT_DIR, f"confusion_{r['name'].replace(' ', '_').lower()}.png"
    )
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {r['name']} confusion matrix to {cm_path}")

# ---------------------------------------------------------------------------
# TextBlob baseline
# ---------------------------------------------------------------------------
def textblob_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 0  # assume first label is positive
    elif polarity < -0.1:
        return 1  # assume second label is negative
    else:
        return 2  # neutral (or third label)

# Apply to test set only for fair comparison
blob_preds = df.loc[y_test.index, "clean_text"].apply(textblob_sentiment)
blob_acc = accuracy_score(y_test, blob_preds)
print(f"TextBlob baseline accuracy: {blob_acc:.2f}")

# ---------------------------------------------------------------------------
# Save models
# ---------------------------------------------------------------------------
joblib.dump(nb_clf, os.path.join(MODEL_DIR, "naive_bayes.pkl"))
joblib.dump(svm_clf, os.path.join(MODEL_DIR, "linear_svm.pkl"))
joblib.dump(rf_clf, os.path.join(MODEL_DIR, "random_forest.pkl"))
print(f"Models saved to {MODEL_DIR}")

print("Pipeline completed successfully.")
