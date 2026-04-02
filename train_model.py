import pandas as pd
def train_model(df, scaler):

    print("Training multiple models...")

    X = df.drop("SepsisLabel", axis=1)
    y = df["SepsisLabel"]

    X.columns = X.columns.astype(str)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ Scaling (correct now)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, f1_score

    # ✅ FIXED models dictionary
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),

        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(class_weight='balanced'),
            n_estimators=100,
            random_state=42
        ),

        "DecisionTree": DecisionTreeClassifier(class_weight='balanced')
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print(f"{name} -> Accuracy: {acc}, F1: {f1}")

        if f1 > best_score:
            best_score = f1
            best_model = model

    import joblib
    import os

    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("✅ Model and scaler saved")

    return best_model, X_train, X_test, y_test
