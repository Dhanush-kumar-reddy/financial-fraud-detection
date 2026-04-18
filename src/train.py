import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from src.preprocess import load_data, split_data, scale_data

def train():
    df = load_data("data/raw/creditcard.csv")

    X_train, X_test, y_train, y_test = split_data(df)

    X_train, X_test, scaler = scale_data(X_train, X_test)

    # 🔥 SMOTE (IMPORTANT)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 🔥 MODEL
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_res, y_train_res)

    # Save everything
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("✅ Model trained and saved!")

if __name__ == "__main__":
    train()