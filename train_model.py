"""
Trains the logistic regression heart-risk model and saves heart_model.joblib.

Run with:  python train_model.py
Requires:  heart1.csv in the same folder (or edit DATA_PATH below)
"""
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

DATA_PATH = "heart1.csv"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values
    y = df["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5-fold cross-validated accuracy, reported to the user in the app
    cv_scores = cross_val_score(LogisticRegression(max_iter=1000), X_scaled, y, cv=5)
    cv_accuracy = cv_scores.mean()

    # held-out AUC just for a console sanity check
    Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    probe = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    test_auc = roc_auc_score(yte, probe.predict_proba(Xte)[:, 1])

    # final model trained on all available data for deployment
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    bundle = {
        "model": model,
        "scaler": scaler,
        "features": FEATURES,
        "cv_accuracy": cv_accuracy,
        "test_auc": test_auc,
        "n_patients": len(df),
    }
    joblib.dump(bundle, "heart_model.joblib")

    print(f"Trained on {len(df)} patients")
    print(f"5-fold CV accuracy: {cv_accuracy*100:.1f}%")
    print(f"Held-out AUC: {test_auc:.3f}")
    print("Saved heart_model.joblib")

if __name__ == "__main__":
    main()
