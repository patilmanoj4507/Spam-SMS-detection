import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Dataset location
DATASET_PATH = os.path.join("dataset", "SMSSpamCollection")

print("Loading SMS dataset...")

if not os.path.exists(DATASET_PATH):
    print("ERROR: Dataset not found!")
    print("Expected location:", DATASET_PATH)
    raise SystemExit(1)

# Load the dataset (tab-separated)
data = pd.read_csv(
    DATASET_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
    on_bad_lines="skip"
)

data = data.dropna()
data["label"] = data["label"].str.strip().str.lower()
data["message"] = data["message"].astype(str).str.strip()

# Keep only valid labels and non-empty messages
data = data[
    data["label"].isin(["ham", "spam"]) &
    (data["message"] != "")
]

if data.empty:
    print("ERROR: No valid SMS records found.")
    raise SystemExit(1)

# Convert labels into readable names
data["label"] = data["label"].map({
    "ham": "Legitimate",
    "spam": "Spam"
})

print("Total messages:", len(data))
print("Training the model...")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

# TF-IDF + Naive Bayes pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nTraining completed successfully!")
print(f"Model accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save the trained model
with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as spam_model.pkl")