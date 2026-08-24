"""
Heart Risk Screening — Streamlit app
Run with:  streamlit run streamlit_app.py
Requires:  heart_model.joblib and cardiac-dashboard.html in the same folder
"""
import streamlit as st
import numpy as np
import joblib
from datetime import datetime

# ---------------- Page config ----------------
st.set_page_config(page_title="Heart Risk — Predictor", page_icon="❤️", layout="wide")

# ---------------- Theme (matches old HTML UI) ----------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,600;1,9..144,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

  :root{
    --ink:#0F141B; --ink-2:#161D26; --paper:#F3EEE1; --line:#2A333F;
    --coral:#FF6A52; --mint:#4ECDB6; --amber:#F0A93F; --muted:#8B95A3;
    --text:#EDE9DC; --text-dim:#B9C0CB;
  }
  @media (prefers-color-scheme: light){
    :root{
      --ink:#F6F3EA; --ink-2:#FFFFFF; --line:#DCD6C4;
      --text:#20241F; --text-dim:#4A4A42; --muted:#6B6A5E;
    }
  }
  .stApp{ background-color: var(--ink); color: var(--text); font-family:'IBM Plex Sans', sans-serif; }
  h1, h2, h3 { font-family:'Fraunces', serif !important; color: var(--text) !important; }
  p, label, .stMarkdown { color: var(--text) !important; }
  .eyebrow{
    font-family:'IBM Plex Mono', monospace; font-size:12px; letter-spacing:0.18em;
    text-transform:uppercase; color:var(--coral); margin-bottom:6px;
  }
  .hero-title{ font-size:44px; font-weight:500; line-height:1.05; margin-bottom:6px; color: var(--text);}
  .hero-title em{ color:var(--coral); font-style:italic; }
  .hero-sub{ color:var(--text-dim); font-size:15px; max-width:620px; margin-bottom:24px;}

  div[data-testid="stForm"]{
    background: var(--ink-2); border:1px solid var(--line); border-radius:14px; padding:26px;
  }
  .stButton>button{
    background: var(--coral); color: #0F141B; border:none; border-radius:999px;
    font-family:'IBM Plex Mono', monospace; font-weight:600; letter-spacing:0.05em;
    text-transform:uppercase; padding:10px 26px;
  }
  .stButton>button:hover{ background:#ff7f68; color:#0F141B; }

  .dash-btn>button{
    background: transparent; color: var(--text) !important; border:1px solid var(--line);
    border-radius:999px; font-family:'IBM Plex Mono', monospace; font-weight:600;
    letter-spacing:0.05em; text-transform:uppercase; padding:10px 26px;
  }
  .dash-btn>button:hover{ border-color: var(--coral); color: var(--coral) !important; }

  .report-card{
    background: var(--paper); color:#1B1B18; border-radius:14px; padding:34px 36px; margin-top:18px;
  }
  .report-card *{ color:#1B1B18 !important; }
  .report-k{ font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:0.1em; text-transform:uppercase; color:#7A7358 !important;}
  .report-title{ font-family:'Fraunces', serif; font-size:24px; font-weight:500; margin-top:4px;}
  .risk-pct{ font-family:'Fraunces', serif; font-size:48px; font-weight:600; }
  .factor-row{ display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid #E2DAC2; font-size:14px;}
  .tag-good{background:rgba(78,205,182,0.18); color:#1E7A68 !important; padding:3px 9px; border-radius:6px; font-family:'IBM Plex Mono', monospace; font-size:11px; font-weight:600;}
  .tag-watch{background:rgba(240,169,63,0.22); color:#8C5A0B !important; padding:3px 9px; border-radius:6px; font-family:'IBM Plex Mono', monospace; font-size:11px; font-weight:600;}
  .tag-flag{background:rgba(255,106,82,0.2); color:#B3351D !important; padding:3px 9px; border-radius:6px; font-family:'IBM Plex Mono', monospace; font-size:11px; font-weight:600;}
  .disclaimer{
    margin-top:20px; font-family:'IBM Plex Mono', monospace; font-size:11px; color:#8A836A !important;
    background:#E9E2CB; padding:14px 16px; border-radius:8px; line-height:1.6;
  }
  .suggest-box{
    margin-top:22px; background:#EFE8D2; border-left:4px solid var(--coral);
    border-radius:8px; padding:16px 18px;
  }
  .suggest-box h4{ margin:0 0 10px 0; font-family:'Fraunces', serif; font-size:17px; color:#1B1B18 !important;}
  .suggest-item{ font-size:13.5px; line-height:1.6; margin-bottom:8px; color:#1B1B18 !important; }
  .suggest-item b{ color:#1B1B18 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- Load trained model ----------------
@st.cache_resource
def load_model():
    return joblib.load("heart_model.joblib")

bundle = load_model()
model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]
cv_accuracy = bundle.get("cv_accuracy")
n_patients = bundle.get("n_patients")

# ---------------- Hero ----------------
st.markdown('<div class="eyebrow">● Cardiac Risk Screening · Trained ML model</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Know your heart<br><em>before it tells you.</em></div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="hero-sub">Enter your clinical readings below — a logistic regression model trained on '
    f'{n_patients} patient records ({cv_accuracy*100:.1f}% cross-validated accuracy) scores them instantly.</div>',
    unsafe_allow_html=True
)

# ---------------- Dashboard navigation button (outside form) ----------------
dcol1, dcol2 = st.columns([1, 5])
with dcol1:
    st.markdown('<div class="dash-btn">', unsafe_allow_html=True)
    if st.button("📊 View dashboard"):
        st.switch_page("pages/1_📊_Dashboard.py")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Input form ----------------
with st.form("patient_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**01 · Demographics & vitals**")
        age = st.slider("Age (years)", 18, 90, 45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
        chol = st.number_input("Serum cholesterol (mg/dl)", 100, 600, 220)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl?", ["No", "Yes"])

    with c2:
        st.markdown("**02 · Cardiac test results**")
        cp = st.selectbox("Chest pain type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
        restecg = st.selectbox("Resting ECG result", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])
        ca = st.selectbox("Major vessels colored (0–3)", [0, 1, 2, 3])

    with c3:
        st.markdown("**03 · Exercise stress test**")
        thalach = st.slider("Max heart rate achieved", 70, 210, 150)
        exang = st.selectbox("Exercise-induced angina?", ["No", "Yes"])
        oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox("Slope of peak ST segment", ["Upsloping", "Flat", "Downsloping"])

    submitted = st.form_submit_button("Generate report →")

# ---------------- Prediction + report ----------------
if submitted:
    row = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp),
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "restecg": ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg),
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": ["Upsloping", "Flat", "Downsloping"].index(slope),
        "ca": ca,
        "thal": ["Normal", "Fixed defect", "Reversible defect"].index(thal),
    }

    X = np.array([[row[f] for f in features]], dtype=float)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0][1]
    pct = round(proba * 100)

    if pct < 30:
        level, color = "Low risk", "#4ECDB6"
    elif pct < 60:
        level, color = "Moderate risk", "#F0A93F"
    else:
        level, color = "High risk", "#FF6A52"

    # ---- Threshold checks for every field (watch, flag) ----
    # watch = borderline / outside comfortable range
    # flag  = clearly outside healthy range
    factors = [
        ("Age", age >= 45, age >= 60, f"{age} yrs"),
        ("Resting blood pressure", trestbps >= 120, trestbps >= 140, f"{trestbps} mm Hg"),
        ("Cholesterol", chol >= 200, chol >= 240, f"{chol} mg/dl"),
        ("Fasting blood sugar", row["fbs"] == 1, False, fbs),
        ("Resting ECG", row["restecg"] >= 1, row["restecg"] == 2, restecg),
        ("Chest pain type", row["cp"] >= 1, row["cp"] == 3, cp),
        ("Max heart rate", thalach < 140, thalach < 110, f"{thalach} bpm"),
        ("Exercise-induced angina", False, row["exang"] == 1, exang),
        ("ST depression (oldpeak)", oldpeak >= 1, oldpeak >= 2, f"{oldpeak:.1f}"),
        ("Slope of peak ST segment", row["slope"] == 1, row["slope"] == 2, slope),
        ("Major vessels colored", ca >= 1, ca >= 2, f"{ca} of 3"),
        ("Thalassemia", row["thal"] == 1, row["thal"] == 2, thal),
    ]

    # ---- Advice text for each factor when Watch / Flagged ----
    suggestions = {
        "Age": "Age itself can't be changed, but yearly cardiac check-ups become more important after 45.",
        "Resting blood pressure": "Cut down salt intake, monitor BP weekly, and stay active — sustained readings above 130 mm Hg need a doctor's input.",
        "Cholesterol": "Reduce fried/oily food and red meat, add fibre (oats, fruits), and get a lipid profile test done.",
        "Fasting blood sugar": "Get an HbA1c / fasting glucose test done; watch sugar and refined-carb intake.",
        "Resting ECG": "An abnormal ECG finding should be reviewed by a cardiologist along with a follow-up ECG or echo.",
        "Chest pain type": "Any recurring chest discomfort, especially of the asymptomatic/atypical kind, should be evaluated promptly — it can be silent but serious.",
        "Max heart rate": "A lower-than-expected max heart rate on exertion can indicate reduced cardiac capacity; a stress test with a cardiologist is advisable.",
        "Exercise-induced angina": "Chest pain triggered by exertion is a red flag — avoid strenuous activity until evaluated and get a stress test done.",
        "ST depression (oldpeak)": "Higher ST depression during stress testing suggests reduced blood flow to the heart — discuss this reading with a cardiologist.",
        "Slope of peak ST segment": "A flat or downsloping ST segment during exercise can indicate ischemia — worth a cardiology review.",
        "Major vessels colored": "More vessels showing blockage on fluoroscopy is a significant finding — this needs direct cardiologist follow-up.",
        "Thalassemia": "An abnormal thalassemia (perfusion) result should be discussed with a cardiologist, especially if paired with other flagged factors.",
    }

    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    rc1, rc2 = st.columns([1, 2])

    with rc1:
        st.markdown('<div class="report-k">Heart risk screening — result</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="risk-pct" style="color:{color}">{pct}%</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:{color}; font-family:IBM Plex Mono; font-weight:600;">{level}</div>', unsafe_allow_html=True)
        st.caption(f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')}")

    with rc2:
        st.markdown('<div class="report-title">Key contributing factors</div>', unsafe_allow_html=True)
        for name, watch, flag, extra in factors:
            tag_class = "tag-flag" if flag else ("tag-watch" if watch else "tag-good")
            tag_label = "Flagged" if flag else ("Watch" if watch else "Normal")
            st.markdown(
                f'<div class="factor-row"><span>{name} — {extra}</span>'
                f'<span class="{tag_class}">{tag_label}</span></div>',
                unsafe_allow_html=True
            )

        flagged_names = [f[0] for f in factors if f[2]]
        watch_names = [f[0] for f in factors if f[1] and not f[2]]
        flagged, watch_c = len(flagged_names), len(watch_names)

        # ---- FIXED narrative: now matches the model's own risk band,
        # instead of contradicting it when flagged/watch counts are low. ----
        if level == "Low risk":
            if watch_c:
                story = f"The model score is low, though {watch_c} factor(s) are borderline and worth keeping an eye on."
            else:
                story = "Most indicators fall within expected ranges — nothing here points strongly toward cardiac risk."
        elif level == "Moderate risk":
            if flagged or watch_c:
                story = f"The model places this profile at moderate risk, with {flagged} flagged and {watch_c} borderline factor(s) nudging the score upward. Worth discussing with a doctor."
            else:
                story = "The model places this profile at moderate risk based on the overall combination of readings, even though no single factor is clearly out of range on its own. Worth discussing with a doctor."
        else:  # High risk
            if flagged:
                story = f"{flagged} factor(s) are clearly outside the healthy range, pushing this profile into the high-risk band. A clinical evaluation is recommended soon."
            elif watch_c:
                story = f"No single factor is clearly out of range, but the model is weighing {watch_c} borderline factor(s) together as a high-risk combination. A clinical evaluation is recommended soon."
            else:
                story = "The model has scored this profile as high-risk based on the overall pattern of readings, even though individual factors look close to normal. A clinical evaluation is recommended soon."
        st.markdown(f'<div style="margin-top:20px; line-height:1.7;">{story}</div>', unsafe_allow_html=True)

        # ---- Suggestions block ----
        concern_names = flagged_names + watch_names
        if concern_names:
            st.markdown('<div class="suggest-box"><h4>Suggestions</h4>', unsafe_allow_html=True)
            for name in flagged_names:
                st.markdown(f'<div class="suggest-item">🔴 <b>{name}:</b> {suggestions.get(name, "")}</div>', unsafe_allow_html=True)
            for name in watch_names:
                st.markdown(f'<div class="suggest-item">🟡 <b>{name}:</b> {suggestions.get(name, "")}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="disclaimer">This score is generated by a logistic regression model trained on '
        f'{n_patients} patient records ({cv_accuracy*100:.1f}% cross-validated accuracy) — it is <strong>not a diagnosis</strong>. '
        f'Please consult a cardiologist for clinical evaluation.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)