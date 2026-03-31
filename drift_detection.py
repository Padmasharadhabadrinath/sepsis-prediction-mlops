from scipy.stats import ks_2samp
import pandas as pd


def detect_drift(train_df, test_df):

    print("Checking data drift...")

    drift_results = []

    for col in train_df.columns:

        if col in test_df.columns:

            stat, p_value = ks_2samp(train_df[col], test_df[col])

            drift_results.append({
                "feature": col,
                "p_value": p_value
            })

    drift_df = pd.DataFrame(drift_results)

    drift_df.to_csv("reports/drift_report.csv", index=False)

    print("Drift report saved")