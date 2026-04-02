from fastapi import FastAPI
import pandas as pd
import joblib

# ✅ CREATE APP (this was missing)
app = FastAPI()

# Load model
model = joblib.load("models/trained_model.pkl")
print("Model expects features:", model.feature_names_in_)

FEATURE_COLUMNS = [
    'Hour','HR','O2Sat','Temp','MAP','Resp','BUN','Chloride',
    'Creatinine','Glucose','Hct','Hgb','WBC','Platelets',
    'Age','HospAdmTime','ICULOS','Unit','Gender_1'
]

def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    for col in FEATURE_COLUMNS:
        if col not in df:
            df[col] = 0

    df = df[FEATURE_COLUMNS]
    return df

# ✅ API ROUTE
@app.post("/predict")
def predict(data: dict):
    try:
        input_data = preprocess_input(data)
        prediction = model.predict(input_data)[0]
        return {"prediction": int(prediction),
                "result": "Sepsis" if prediction == 1 else "No Sepsis"}
    
    except Exception as e:
        return {"error": str(e)}

# ✅ Optional root route
@app.get("/")
def home():
    return {"message": "Sepsis Prediction API Running"}
