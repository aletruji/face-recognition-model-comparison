from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from core.data_utils import split_per_person
import time
from core.eval_utils import measure_model_size


def run_model(model_class, model_name):
    dataset_path = 'data/'
    model = model_class()
    X, y = model.load_data(dataset_path)
    X_train, y_train, X_test, y_test = split_per_person(X, y)
    model.train(X_train, y_train)

    model_size_kb = measure_model_size(model)




    predictions = []
    start_time = time.time()
    for img in X_test:
        pred_label, _ = model.predict(img)
        predictions.append(pred_label)
    end_time = time.time()

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
    rec = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    duration = end_time - start_time

    print(f"\n--- Ergebnisse für {model_name} ---")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print(f"Inferenzzeit:  {duration:.3f} Sekunden für {len(X_test)} Bilder")
    print(f"Modellgröße:  {model_size_kb:.2f} KB\n")