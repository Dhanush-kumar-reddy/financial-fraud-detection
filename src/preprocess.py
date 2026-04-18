import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def split_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train[['Amount','Time']] = scaler.fit_transform(X_train[['Amount','Time']])
    X_test[['Amount','Time']] = scaler.transform(X_test[['Amount','Time']])

    return X_train, X_test, scaler