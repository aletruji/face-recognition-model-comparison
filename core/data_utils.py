import numpy as np

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
