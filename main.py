

from models.lbph import LBPHModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time
import os
import cv2

# Setze Pfad zum Datensatz
dataset_path = 'data/'

# Modell initialisieren
model = LBPHModel()
X, y = model.load_data(dataset_path)

# --- Neuer Split: pro Person ---
def split_per_person(X, y, train_ratio=0.8):
    X_train, y_train, X_test, y_test = [], [], [], []
    classes = np.unique(y)
    for cls in classes:
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)
        split = int(train_ratio * len(idx))
        train_idx, test_idx = idx[:split], idx[split:]
        X_train.extend(X[train_idx])
        y_train.extend(y[train_idx])
        X_test.extend(X[test_idx])
        y_test.extend(y[test_idx])
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# Split anwenden
X_train, y_train, X_test, y_test = split_per_person(X, y)

# Modell trainieren
model.train(X_train, y_train)

# Vorhersagen sammeln
predictions = []
start_time = time.time()
for img in X_test:
    pred_label, _ = model.predict(img)
    predictions.append(pred_label)
end_time = time.time()

# Metriken berechnen
acc = accuracy_score(y_test, predictions)
prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
rec = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
duration = end_time - start_time

# Ergebnisse ausgeben
print(f"Accuracy:      {acc:.4f}")
print(f"Precision:     {prec:.4f}")
print(f"Recall:        {rec:.4f}")
print(f"F1-Score:      {f1:.4f}")
print(f"Inferenzzeit:  {duration:.2f} Sekunden für {len(X_test)} Bilder")


