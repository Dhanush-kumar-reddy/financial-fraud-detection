import pandas as pd
from src.predict import predict

df = pd.read_csv("data/raw/creditcard.csv")

# mix normal + fraud
test_df = pd.concat([
    df[df["Class"] == 0].head(5),
    df[df["Class"] == 1].head(5)
])

probs, preds = predict(test_df)

print("Actual:", test_df["Class"].values)
print("Predicted:", preds)
print("Probabilities:", probs)