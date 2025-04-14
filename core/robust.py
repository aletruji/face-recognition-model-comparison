import numpy as np
import cv2
from sklearn.metrics import accuracy_score, f1_score

# Störung 1: Gaussian Noise
def add_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

# Störung 2: Blur
def add_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

# Störung 3: Abdunkeln
def darken(img, factor=0.7):
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

# Störung anwenden auf Bildarray
def apply_perturbation(X, func):
    return np.array([func(img) for img in X])

# Einzelmetriken berechnen
def evaluate_metrics(model, X, y):
    preds = [model.predict(img)[0] for img in X]
    return {
        "accuracy": accuracy_score(y, preds),
        "f1": f1_score(y, preds, average='weighted', zero_division=0)
    }

# Robustheit berechnen
def evaluate_robustness(model, X_test, y_test):
    metrics_original = evaluate_metrics(model, X_test, y_test)
    metrics_noise = evaluate_metrics(model, apply_perturbation(X_test, add_noise), y_test)
    metrics_blur = evaluate_metrics(model, apply_perturbation(X_test, add_blur), y_test)
    metrics_dark = evaluate_metrics(model, apply_perturbation(X_test, darken), y_test)

    avg_drop = {
        key: metrics_original[key] - np.mean([
            metrics_noise[key],
            metrics_blur[key],
            metrics_dark[key]
        ]) for key in metrics_original
    }

    return {
        "original": metrics_original,
        "noise": metrics_noise,
        "blur": metrics_blur,
        "dark": metrics_dark,
        "avg_drop": avg_drop
    }
