# Heart Risk Screening — Streamlit App

## Files
- `streamlit_app.py` — the app
- `heart_model.joblib` — pre-trained logistic regression model, scaler, and feature list (already built, ready to use)
- `train_model.py` — retrains the model from `heart1.csv` and overwrites `heart_model.joblib`
- `heart1.csv` — training data (1,025 patient records)
- `requirements.txt` — Python dependencies

## Run it
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`.

## Model
Logistic regression, trained on 13 clinical features (age, sex, chest pain type,
resting BP, cholesterol, fasting blood sugar, resting ECG, max heart rate,
exercise-induced angina, ST depression, ST slope, major vessels, thalassemia).

- 5-fold cross-validated accuracy: **84.6%**
- Held-out AUC: **0.93**

## Retrain on new data
Replace `heart1.csv` with your own file (same column names), then:
```bash
python train_model.py
```
This overwrites `heart_model.joblib`. Restart the app to pick up the new model.

## Note
This is a screening tool, not a diagnostic one. The app's own disclaimer says
so — keep it in place if you customize the UI.
