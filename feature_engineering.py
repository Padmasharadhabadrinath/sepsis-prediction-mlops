import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):

    print("Starting feature engineering...")
    df['Unit'] = df['Unit1'] + df['Unit2']

    columns_drop = ['Unnamed: 0','SBP','DBP','EtCO2','BaseExcess','HCO3','pH','PaCO2',
                    'Alkalinephos','Calcium','Magnesium','Phosphate','Potassium',
                    'PTT','Fibrinogen','Unit1','Unit2']

    df = df.drop(columns=columns_drop, errors='ignore')

    # patient wise filling
    grouped = df.groupby('Patient_ID')
    df = grouped.apply(lambda x: x.bfill().ffill()).reset_index(drop=True)
    # drop high null columns
    null_cols = ['TroponinI','Bilirubin_direct','AST','Bilirubin_total',
                 'Lactate','SaO2','FiO2','Patient_ID']

    df = df.drop(columns=null_cols, errors='ignore')

    # one hot encoding
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    # log transformation
    cols_log = ['MAP','BUN','Creatinine','Glucose','WBC','Platelets']

    for col in cols_log:
        if col in df.columns:
            df[col] = np.log(df[col] + 1)

    # scaling
    scaler = StandardScaler()

    scale_cols = ['HR','O2Sat','Temp','MAP','Resp','BUN','Chloride',
                  'Creatinine','Glucose','Hct','Hgb','WBC','Platelets']

    scale_cols = [c for c in scale_cols if c in df.columns]

    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    df = df.dropna()

    print("Feature engineering completed")

    return df, scaler
