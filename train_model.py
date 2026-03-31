from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import datetime


def train_model(df):

    print("Training multiple models...")

    X = df.drop("SepsisLabel", axis=1)
    y = df["SepsisLabel"]

    X.columns = X.columns.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔥 Models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=100,
            random_state=42
        ),
        "DecisionTree": DecisionTreeClassifier()
    }

    best_model = None
    best_score = 0
    best_model_name = ""

    # 🔁 Train & compare
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

        # ✅ Select best using F1
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_model_name = name

    print(f"\n✅ Best Model: {best_model_name} with F1 Score: {best_score:.4f}")

    # 💾 Save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/best_model_{best_model_name}_{timestamp}.pkl"

    joblib.dump(best_model, model_path)
    joblib.dump(best_model, "models/trained_model.pkl")

    print(f"Best model saved to {model_path}")

    return best_model, X_train, X_test, y_test