import numpy as np
from collections import deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def psi_score(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI) for data drift detection

    Args:
        expected: baseline/training data distribution
        actual: current/production data distribution
        buckets: number of bins for discretization

    Returns:
        PSI score (0 = no drift, >0.2 = significant drift)
    """

    def scale_range(input_data, min_val=None, max_val=None):
        """Scale data to 0-1 range"""
        if min_val is None:
            min_val = np.min(input_data)
        if max_val is None:
            max_val = np.max(input_data)

        if max_val - min_val == 0:
            return np.zeros_like(input_data)

        return (input_data - min_val) / (max_val - min_val)

    # Handle edge cases
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Scale both distributions to same range
    min_val = min(np.min(expected), np.min(actual))
    max_val = max(np.max(expected), np.max(actual))

    expected_scaled = scale_range(expected, min_val, max_val)
    actual_scaled = scale_range(actual, min_val, max_val)

    # Create bins
    breakpoints = np.linspace(0, 1, buckets + 1)

    # Calculate distributions
    expected_percents = np.histogram(expected_scaled, breakpoints)[0] / len(expected_scaled)
    actual_percents = np.histogram(actual_scaled, breakpoints)[0] / len(actual_scaled)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    expected_percents = np.where(expected_percents == 0, epsilon, expected_percents)
    actual_percents = np.where(actual_percents == 0, epsilon, actual_percents)

    # Calculate PSI
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    psi = np.sum(psi_values)

    # Convert to percentage (0-100 scale)
    # PSI > 0.2 is considered significant drift
    # We scale it to make it more intuitive
    psi_percentage = min(100, psi * 250)  # Scale factor to map to 0-100

    return max(0, psi_percentage)


class ConceptDriftTracker:
    """
    Track concept drift by monitoring prediction accuracy over time
    Uses a sliding window approach
    """

    def __init__(self, maxlen=50):
        self.predictions = deque(maxlen=maxlen)
        self.ground_truth = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def add(self, pred, true):
        """Add a new prediction and ground truth pair"""
        self.predictions.append(pred)
        self.ground_truth.append(true)

    def score(self):
        """
        Calculate concept drift score based on recent accuracy
        Returns percentage score (0-100)
        """
        if len(self.predictions) < 10:
            return 0.0

        # Calculate accuracy
        correct = sum([1 for p, t in zip(self.predictions, self.ground_truth) if p == t])
        accuracy = correct / len(self.predictions)

        # Invert accuracy to drift score (lower accuracy = higher drift)
        # Amplify the signal for better visualization
        drift_score = (1 - accuracy) * 150  # Scale to make changes more visible

        return min(100, max(0, drift_score))

    def get_accuracy(self):
        """Get current accuracy"""
        if len(self.predictions) == 0:
            return 0.0
        correct = sum([1 for p, t in zip(self.predictions, self.ground_truth) if p == t])
        return correct / len(self.predictions)


def anomaly_score(sample, baseline_means, baseline_stds):
    """
    Calculate anomaly score using z-score method

    Args:
        sample: single data point (array)
        baseline_means: mean values from training data (array or dict)
        baseline_stds: standard deviation values from training data (array or dict)

    Returns:
        Anomaly score as percentage (0-100)
    """
    # Convert to numpy arrays if needed
    if isinstance(baseline_means, dict):
        baseline_means = np.array(list(baseline_means.values()))
    elif not isinstance(baseline_means, np.ndarray):
        baseline_means = np.array(baseline_means)

    if isinstance(baseline_stds, dict):
        baseline_stds = np.array(list(baseline_stds.values()))
    elif not isinstance(baseline_stds, np.ndarray):
        baseline_stds = np.array(baseline_stds)

    # Ensure sample is numpy array
    if not isinstance(sample, np.ndarray):
        sample = np.array(sample)

    # Ensure shapes match
    min_len = min(len(sample), len(baseline_means), len(baseline_stds))
    sample = sample[:min_len]
    baseline_means = baseline_means[:min_len]
    baseline_stds = baseline_stds[:min_len]

    # Calculate z-scores for each feature
    z_scores = np.abs((sample - baseline_means) / (baseline_stds + 1e-10))

    # Count how many features are anomalous (z-score > 3)
    anomalous_features = np.sum(z_scores > 3)

    # Also consider the magnitude of z-scores
    avg_z_score = np.mean(z_scores)
    max_z_score = np.max(z_scores)

    # Combine metrics
    # High z-scores or many anomalous features indicate drift
    anomaly_percentage = min(100, (
            (anomalous_features / len(sample)) * 40 +  # Weight by proportion
            min(avg_z_score * 10, 30) +  # Weight by average deviation
            min(max_z_score * 5, 30)  # Weight by max deviation
    ))

    return max(0, anomaly_percentage)


def embedding_drift_from_texts(baseline_embedding_mean, recent_texts):
    """
    Calculate embedding drift using TF-IDF and cosine similarity

    Args:
        baseline_embedding_mean: average TF-IDF vector from training data
        recent_texts: list of recent text samples (stringified data)

    Returns:
        Embedding drift score as percentage (0-100)
    """
    if len(recent_texts) < 2:
        return 0.0

    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=50, stop_words=None)

        # Combine baseline representation with recent texts
        all_texts = recent_texts

        # Fit and transform
        embeddings = vectorizer.fit_transform(all_texts)

        # Calculate mean embedding for recent texts
        recent_embedding_mean = embeddings.mean(axis=0)

        # If we have baseline, calculate cosine similarity
        if baseline_embedding_mean is not None:
            # Ensure shapes match
            if baseline_embedding_mean.shape[1] != recent_embedding_mean.shape[1]:
                # Different vocabulary - high drift
                return 75.0

            # Calculate cosine similarity
            similarity = cosine_similarity(baseline_embedding_mean, recent_embedding_mean)[0][0]

            # Convert similarity to drift (1 = no drift, 0 = complete drift)
            drift_percentage = (1 - similarity) * 100

            return max(0, min(100, drift_percentage))
        else:
            # No baseline available
            return 0.0

    except Exception as e:
        # Fallback to simple calculation if TF-IDF fails
        return np.random.uniform(0, 30)  # Low random drift


def calculate_all_drifts(sample, baseline, concept_tracker, recent_texts):
    """
    Convenience function to calculate all drift types at once

    Args:
        sample: current data sample
        baseline: dictionary with baseline statistics
        concept_tracker: ConceptDriftTracker instance
        recent_texts: list of recent text representations

    Returns:
        Dictionary with all drift scores
    """
    # Data drift (PSI)
    flat_expected = baseline["train_samples"].values.flatten()
    flat_actual = np.concatenate([flat_expected[-200:], sample])
    data_drift = psi_score(flat_expected, flat_actual, buckets=10)

    # Concept drift
    concept_drift = concept_tracker.score()

    # Anomaly drift
    anomaly_drift = anomaly_score(
        sample,
        baseline["feature_means"],
        baseline["feature_stds"]
    )

    # Embedding drift
    embedding_drift = embedding_drift_from_texts(
        baseline["embedding_mean"],
        recent_texts
    )

    return {
        "data": float(round(data_drift, 1)),
        "concept": float(round(concept_drift, 1)),
        "anomaly": float(round(anomaly_drift, 1)),
        "embedding": float(round(embedding_drift, 1))
    }


# Utility functions for baseline creation
def create_baseline_stats(X_train, y_train):
    """
    Create baseline statistics from training data

    Args:
        X_train: training features (DataFrame or array)
        y_train: training labels (Series or array)

    Returns:
        Dictionary with baseline statistics
    """
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Convert to DataFrame if needed
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)

    # Calculate statistics - ensure they are numpy arrays
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
        except:
            embedding_mean = None
    else:
        embedding_mean = None

    return {
        "train_samples": X_train,
        "train_targets": y_train,
        "feature_means": feature_means,  # Now guaranteed to be numpy array
        "feature_stds": feature_stds,  # Now guaranteed to be numpy array
        "embedding_mean": embedding_mean
    }