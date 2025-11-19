"""
Train a simple model and create baseline statistics
Run this once to generate the required model files
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


def create_baseline_stats(X_train, y_train):
    """Create baseline statistics from training data"""
    # Calculate statistics - ensure numpy arrays
    feature_means = X_train.mean().values.astype(float)
    feature_stds = X_train.std().values.astype(float)

    # Create text representations for embedding
    texts = X_train.astype(str).agg(" ".join, axis=1).tolist()

    # Calculate baseline embedding
    if len(texts) > 1:
        try:
            vectorizer = TfidfVectorizer(max_features=50)
            embeddings = vectorizer.fit_transform(texts)
            embedding_mean = embeddings.mean(axis=0)
        except Exception as e:
            print(f"⚠️ Warning: Could not create embeddings: {e}")
            embedding_mean = None
    else:
        embedding_mean = None

    return {
        "train_samples": X_train,
        "train_targets": y_train,
        "feature_means": feature_means,  # numpy array
        "feature_stds": feature_stds,  # numpy array
        "embedding_mean": embedding_mean
    }


def main():
    print("🚀 Training model and creating baseline statistics...")

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Load Iris dataset
    print("📊 Loading Iris dataset...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"✅ Training samples: {len(X_train)}")
    print(f"✅ Test samples: {len(X_test)}")

    # Train Random Forest
    print("🌲 Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"✅ Training accuracy: {train_score:.3f}")
    print(f"✅ Test accuracy: {test_score:.3f}")

    # Save model
    model_path = "models/rf_iris.joblib"
    joblib.dump(clf, model_path)
    print(f"💾 Model saved to {model_path}")

    # Create baseline statistics
    print("📈 Creating baseline statistics...")
    baseline = create_baseline_stats(X_train, y_train)

    # Save baseline
    baseline_path = "models/baseline_stats.joblib"
    joblib.dump(baseline, baseline_path)
    print(f"💾 Baseline statistics saved to {baseline_path}")

    print("\n✨ Setup complete! You can now run the Streamlit app.")
    print("Run: streamlit run app.py")


if __name__ == "__main__":
    main()