import shap
import matplotlib.pyplot as plt
import pandas as pd

def generate_shap(model, X):

    print("Calculating SHAP values on sample data...")

    # Take small sample
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X_sample = X.sample(1000, random_state=42)

    try:
        if model.__class__.__name__ in ["RandomForestClassifier", "DecisionTreeClassifier"]:

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            print("Generating SHAP summary plot...")

            shap.summary_plot(shap_values, X_sample, show=False)

            plt.savefig("reports/shap_summary.png")
            plt.close()

            print("SHAP report saved to reports/shap_summary.png")

        else:
            print(f" SHAP not supported for {type(model).__name__}, skipping...")

    except Exception as e:
        print(f"SHAP Error: {e}")
from lime.lime_tabular import LimeTabularExplainer
import joblib
import numpy as np

def generate_lime(model, X_train):

    print("Generating LIME explanations...")

    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=['No Sepsis','Sepsis'],
        mode='classification'
    )

    sample = X_train.iloc[0]

    explanation = explainer.explain_instance(
        sample,
        model.predict_proba,
        num_features=10
    )

    explanation.save_to_file("reports/lime_explanations.html")

    print("LIME explanation saved")
