import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

DATASET_FILE = "sign_language/dataset_asl.csv"
MODEL_FILE = "sign_language/asl_model.pkl"

def normalize_row(x):
    """
    x: shape (63,) -> (x,y,z)*21
    Normalize:
      - Wrist (landmark 0) merkez olsun (x0,y0,z0 çıkar)
      - Ölçek: wrist(0) ile middle_mcp(9) arasındaki mesafe 1 olsun
    """
    pts = x.reshape(21, 3).astype(np.float32)

    # 1) Merkezleme (wrist)
    wrist = pts[0].copy()
    pts = pts - wrist

    # 2) Ölçekleme (wrist -> middle_mcp)
    # middle_mcp index=9
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1.0
    pts = pts / scale

    return pts.reshape(-1)

def main():
    df = pd.read_csv(DATASET_FILE, header=None)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Normalize (çok önemli)
    Xn = np.vstack([normalize_row(row) for row in X])

    X_train, X_test, y_train, y_test = train_test_split(
        Xn, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_FILE)
    print(f"\nModel kaydedildi: {MODEL_FILE}")

if __name__ == "__main__":
    main()
