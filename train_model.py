import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

from imblearn.over_sampling import SMOTE


def train_model(df, scaler):

    print("Training optimized models...")

    # -------------------------------
    # 🔥 Reduce dataset size (VERY IMPORTANT)
    # -------------------------------
    df = df.sample(n=200000, random_state=42)

    # -------------------------------
    # Split features & target
    # -------------------------------
    X = df.drop("SepsisLabel", axis=1)
    y = df["SepsisLabel"]

    X.columns = X.columns.astype(str)

    # -------------------------------
    # Train Test Split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # 🔥 Apply SMOTE (handle imbalance)
    # -------------------------------
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("After SMOTE:", X_train.shape)

    # -------------------------------
    # Models (optimized)
    # -------------------------------
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        "DecisionTree": DecisionTreeClassifier(class_weight='balanced')
    }

    best_model = None
    best_score = 0

    # -------------------------------
    # Training loop
    # -------------------------------
    for name, model in models.items():
        print(f"Training {name}...")

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

        if f1 > best_score:
            best_score = f1
            best_model = model

    # -------------------------------
    # Save model & scaler
    # -------------------------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("✅ Model and scaler saved")

    return best_model, X_train, X_test, y_test
