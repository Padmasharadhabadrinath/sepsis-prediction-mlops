import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, X_test, y_test):

    print("Evaluating model...")

    preds = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1_score": float(f1_score(y_test, preds))
    }

    print(metrics)

    with open("reports/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return preds
