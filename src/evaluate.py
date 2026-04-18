import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from src.preprocess import load_data, split_data, scale_data

def evaluate():
    df = load_data("data/raw/creditcard.csv")

    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, scaler = scale_data(X_train, X_test)

    model = joblib.load("models/random_forest.pkl")

    probs = model.predict_proba(X_test)[:,1]

    # 🔥 use tuned threshold
    threshold = 0.3
    preds = (probs > threshold).astype(int)

    print("Classification Report:\n")
    print(classification_report(y_test, preds))

    print("\nROC-AUC:", roc_auc_score(y_test, probs))

if __name__ == "__main__":
    evaluate()