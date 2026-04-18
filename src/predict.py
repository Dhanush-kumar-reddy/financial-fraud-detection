import joblib

model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict(data, threshold=0.3):
    if "Class" in data.columns:
        data = data.drop("Class", axis=1)

    # ensure same feature order
    data = data[model.feature_names_in_]

    data[['Amount','Time']] = scaler.transform(data[['Amount','Time']])

    probs = model.predict_proba(data)[:,1]
    preds = (probs > threshold).astype(int)

    return probs, preds