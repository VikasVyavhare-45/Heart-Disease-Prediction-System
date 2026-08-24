import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Heart Disease Dashboard", page_icon="❤️", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("heart1.csv")
    return df

df = load_data()

st.title("❤️ Heart Disease Analysis Dashboard")
st.caption("Interactive dashboard based on the Heart Disease dataset (1,025 records)")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔍 Filters")

age_range = st.sidebar.slider(
    "Age Range",
    int(df["age"].min()),
    int(df["age"].max()),
    (int(df["age"].min()), int(df["age"].max()))
)

sex_option = st.sidebar.multiselect(
    "Sex",
    options=["Male", "Female"],
    default=["Male", "Female"]
)

target_option = st.sidebar.multiselect(
    "Heart Disease Status",
    options=["Disease", "No Disease"],
    default=["Disease", "No Disease"]
)

sex_map = {"Male": 1, "Female": 0}
target_map = {"Disease": 1, "No Disease": 0}

filtered = df[
    (df["age"].between(age_range[0], age_range[1])) &
    (df["sex"].isin([sex_map[x] for x in sex_option])) &
    (df["target"].isin([target_map[x] for x in target_option]))
].copy()

filtered["Sex"] = filtered["sex"].map({1: "Male", 0: "Female"})
filtered["Status"] = filtered["target"].map({1: "Heart Disease", 0: "No Heart Disease"})

# ---------------- KPI ----------------
total = len(filtered)
disease = int((filtered["target"] == 1).sum())
no_disease = int((filtered["target"] == 0).sum())
disease_rate = (disease / total * 100) if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Patients", f"{total:,}")
c2.metric("Heart Disease Cases", f"{disease:,}")
c3.metric("No Disease Cases", f"{no_disease:,}")
c4.metric("Disease Rate", f"{disease_rate:.1f}%")

st.divider()

# ---------------- CHARTS ----------------
left, right = st.columns(2)

with left:
    fig = px.pie(
        filtered,
        names="Status",
        title="Heart Disease Distribution",
        hole=0.55
    )
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with right:
    age_bins = [20, 30, 40, 50, 60, 70, 80]
    age_labels = ["20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    chart_df = filtered.copy()
    chart_df["Age Group"] = pd.cut(
        chart_df["age"], bins=age_bins, labels=age_labels, right=False
    )
    age_chart = (
        chart_df.groupby(["Age Group", "Status"], observed=False)
        .size()
        .reset_index(name="Patients")
    )
    fig = px.bar(
        age_chart,
        x="Age Group",
        y="Patients",
        color="Status",
        barmode="group",
        title="Heart Disease by Age Group"
    )
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:
    sex_chart = (
        filtered.groupby(["Sex", "Status"])
        .size()
        .reset_index(name="Patients")
    )
    fig = px.bar(
        sex_chart,
        x="Sex",
        y="Patients",
        color="Status",
        barmode="group",
        title="Gender-wise Disease Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.scatter(
        filtered,
        x="age",
        y="thalach",
        color="Status",
        size="chol",
        hover_data=["trestbps", "oldpeak", "Sex"],
        title="Age vs Maximum Heart Rate"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------- MEDICAL METRICS ----------------
st.subheader("📊 Health Measurements")

m1, m2 = st.columns(2)

with m1:
    fig = px.histogram(
        filtered,
        x="chol",
        color="Status",
        nbins=30,
        barmode="overlay",
        title="Cholesterol Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

with m2:
    fig = px.box(
        filtered,
        x="Status",
        y="trestbps",
        color="Status",
        title="Resting Blood Pressure by Disease Status"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- FEATURE AVERAGES ----------------
st.subheader("🩺 Average Health Values")

metrics = ["age", "trestbps", "chol", "thalach", "oldpeak"]
avg_df = (
    filtered.groupby("Status")[metrics]
    .mean()
    .round(2)
    .T
    .reset_index()
    .rename(columns={"index": "Feature"})
)

fig = px.bar(
    avg_df.melt(id_vars="Feature", var_name="Status", value_name="Average"),
    x="Feature",
    y="Average",
    color="Status",
    barmode="group",
    title="Average Values by Heart Disease Status"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- DATA TABLE ----------------
st.subheader("📋 Filtered Dataset")
show_cols = [
    "age", "Sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "Status"
]
st.dataframe(filtered[show_cols], use_container_width=True, height=400)

csv = filtered.drop(columns=["Sex", "Status"]).to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Filtered Data",
    csv,
    "filtered_heart_data.csv",
    "text/csv"
)

st.caption("⚠️ Educational dashboard only. This analysis is not a medical diagnosis.")
