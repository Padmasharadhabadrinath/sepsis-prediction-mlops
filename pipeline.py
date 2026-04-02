from src.data_ingestion import load_data
from src.data_validation import validate_data
from src.feature_engineering import preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.explainability import generate_shap, generate_lime
from src.drift_detection import detect_drift


def run_pipeline():

    print("Loading datasets...")

    # 1️⃣ Data ingestion
    train_df, test_df = load_data(
        "data/raw/data_part1.csv",
        "data/raw/data_part2.csv"
    )

    print("Train Shape:", train_df.shape)
    print("Test Shape:", test_df.shape)

    # 2️⃣ Data validation
    print("Validating dataset...")
    validation_result = validate_data(train_df)

    # 3️⃣ Feature engineering
    print("Starting feature engineering...")
    df_processed, scaler = preprocess_data(train_df)
    # 4️⃣ Train model
    print("Training model...")
    model, X_train, X_test, y_test = train_model(df_processed, scaler)

    # 5️⃣ Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    # 6️⃣ Explainability
    print("Generating SHAP explanations...")
    generate_shap(model, X_test)

    print("Generating LIME explanations...")
    generate_lime(model, X_train)

    # 7️⃣ Drift detection
    print("Checking data drift...")
    detect_drift(train_df, test_df)

    print("Pipeline completed successfully")
