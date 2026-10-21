"""Smart Crop Recommendation System - Streamlit app.  Run: streamlit run app.py"""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from crop_info import lookup

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "Crop_recommendation.csv"
MODEL = ROOT / "artifacts" / "model.joblib"
META = ROOT / "artifacts" / "metadata.json"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
GREENS = ["#1b5e20", "#2e7d32", "#43a047", "#66bb6a", "#a5d6a7"]

st.set_page_config(page_title="Smart Crop Recommendation", page_icon="🌾", layout="wide")

st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#1b5e20,#43a047);color:#fff;padding:2.2rem 2rem;border-radius:18px;margin-bottom:1.2rem}
.hero h1{color:#fff;margin:0 0 .4rem 0;font-size:2.2rem}
.hero p{margin:0;font-size:1.05rem;opacity:.95}
.card{background:#f1f8e9;border:1px solid #c5e1a5;border-radius:14px;padding:1rem 1.2rem;height:100%}
.card h4{margin:0 0 .3rem 0;color:#1b5e20}
.metric{background:#fff;border:1px solid #c8e6c9;border-left:6px solid #2e7d32;border-radius:12px;padding:1rem 1.2rem;box-shadow:0 2px 6px rgba(0,0,0,.05)}
.metric .v{font-size:1.9rem;font-weight:700;color:#1b5e20}
.metric .l{color:#4a5a4a;font-size:.9rem}
.result{background:linear-gradient(120deg,#e8f5e9,#c8e6c9);border:2px solid #43a047;border-radius:18px;padding:1.5rem;text-align:center;margin:1rem 0}
.result .name{font-size:2.6rem;font-weight:800;color:#1b5e20}
.result .sub{color:#33691e}
div.stButton>button{background:#2e7d32;color:#fff;border:none;border-radius:10px;font-weight:700;padding:.6rem 1.2rem}
div.stButton>button:hover{background:#1b5e20;color:#fff}
</style>
""", unsafe_allow_html=True)


# ---------- loading ----------
if not DATA.exists():
    st.error("📂 **Dataset not found.** Expected file: `data/Crop_recommendation.csv`")
    st.info("Download the public *Crop Recommendation Dataset* from Kaggle (free account), put the CSV in the "
            "`data` folder, then run `python train_model.py`. See `data/README_DATASET.txt`.")
    st.stop()
if not MODEL.exists() or not META.exists():
    st.error("🧠 **Trained model not found.** Run `python train_model.py` in the project folder first, then refresh.")
    st.stop()


@st.cache_data
def load_data():
    d = pd.read_csv(DATA)
    d["label"] = d["label"].astype(str).str.strip().str.lower()
    return d


@st.cache_resource
def load_model():
    return joblib.load(MODEL)


@st.cache_data
def load_meta():
    return json.loads(META.read_text())


df, model, meta = load_data(), load_model(), load_meta()
ranges = meta["feature_ranges"]


def metric(col, value, label, icon):
    col.markdown(f'<div class="metric"><div class="v">{icon} {value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


def card(col, title, text):
    col.markdown(f'<div class="card"><h4>{title}</h4>{text}</div>', unsafe_allow_html=True)


# ---------- navigation ----------
PAGES = ["🏠 Home", "🌱 Crop Recommendation", "📊 Data Analytics", "🏆 Model Performance", "ℹ️ About Project"]
st.sidebar.title("🌾 Smart Crop System")
page = st.sidebar.radio("Navigate", PAGES, key="page")
st.sidebar.caption("Runs fully offline once installed.")


def go_recommend():
    st.session_state["page"] = PAGES[1]


# ---------- HOME ----------
if page == PAGES[0]:
    st.markdown('<div class="hero"><h1>🌾 Smart Crop Recommendation System</h1>'
                '<p>Enter soil nutrients and weather conditions — a machine learning model suggests the crop '
                'that best matches patterns in a real agricultural dataset.</p></div>', unsafe_allow_html=True)
    c = st.columns(4)
    metric(c[0], meta["n_classes"], "Crop categories", "🌿")
    metric(c[1], len(FEATURES), "Input parameters", "🧪")
    metric(c[2], f'{meta["rows_clean"]:,}', "Dataset records (cleaned)", "🗂️")
    metric(c[3], f'{meta["test_accuracy"]*100:.2f}%', f'Test accuracy ({meta["selected_model"]})', "🎯")
    st.write("")
    st.button("🌱 Open Crop Recommendation", on_click=go_recommend)

    st.subheader("✨ Features")
    c = st.columns(4)
    card(c[0], "🤖 ML Prediction", "Model chosen by comparing three algorithms.")
    card(c[1], "⚠️ Input Checks", "Validation and out-of-range warnings.")
    card(c[2], "📈 Interactive Charts", "Explore the data with Plotly.")
    card(c[3], "📚 Crop Info", "Short notes on every supported crop.")

    st.subheader("⚙️ How it works")
    c = st.columns(4)
    card(c[0], "1️⃣ Enter", "N, P, K, temperature, humidity, pH, rainfall.")
    card(c[1], "2️⃣ Validate", "Values are checked against valid and observed ranges.")
    card(c[2], "3️⃣ Predict", "The saved model classifies the conditions.")
    card(c[3], "4️⃣ Review", "See the crop, inputs and model-estimated probabilities.")

    st.subheader("🧰 Technology stack")
    st.write("Python • Pandas • NumPy • Scikit-learn • Plotly • Streamlit • Joblib")
    st.subheader("🚧 Limitations")
    st.warning("Predictions reflect patterns in one public dataset. They do not consider soil type, market prices, "
               "pests, irrigation, season or local regulations, and are **not** a guarantee of real-world "
               "crop suitability. Consult local agricultural experts before making farming decisions.")

# ---------- RECOMMENDATION ----------
elif page == PAGES[1]:
    st.title("🌱 Crop Recommendation")
    st.caption("Enter your field conditions. Defaults are dataset medians.")

    # (key, label, unit, hard_min, hard_max, step, help)
    specs = [
        ("N", "Nitrogen (N)", "dataset scale", 0.0, 500.0, 1.0, "Nitrogen content in soil (same scale as the dataset)."),
        ("P", "Phosphorus (P)", "dataset scale", 0.0, 500.0, 1.0, "Phosphorus content in soil."),
        ("K", "Potassium (K)", "dataset scale", 0.0, 500.0, 1.0, "Potassium content in soil."),
        ("temperature", "Temperature", "°C", -10.0, 60.0, 0.1, "Average temperature in °C."),
        ("humidity", "Relative humidity", "%", 0.0, 100.0, 0.1, "Relative humidity in percent (0–100)."),
        ("ph", "Soil pH", "pH", 0.0, 14.0, 0.1, "Soil pH (0–14)."),
        ("rainfall", "Rainfall", "mm", 0.0, 1000.0, 0.1, "Rainfall in millimetres."),
    ]
    vals = {}
    left, right = st.columns(2)
    for i, (k, lab, unit, lo, hi, step, hlp) in enumerate(specs):
        with (left if i % 2 == 0 else right):
            vals[k] = st.number_input(f"{lab} ({unit})", min_value=lo, max_value=hi,
                                      value=float(round(ranges[k]["median"], 2)), step=step, help=hlp, key=f"in_{k}")

    if st.button("RECOMMEND SUITABLE CROP", use_container_width=True):
        errors = []
        for k, lab, unit, lo, hi, *_ in specs:
            v = vals[k]
            if v is None or pd.isna(v):
                errors.append(f"{lab} is missing.")
            elif not (lo <= v <= hi):
                errors.append(f"{lab} must be between {lo:g} and {hi:g} {unit}.")
        if errors:
            st.session_state.pop("result", None)
            for e in errors:
                st.error(e)
        else:
            X = pd.DataFrame([[vals[k] for k in FEATURES]], columns=FEATURES)
            pred = str(model.predict(X)[0])
            probs = None
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)[0]
                probs = (pd.DataFrame({"Crop": [c.title() for c in model.classes_], "Probability": p})
                         .sort_values("Probability", ascending=False).head(5))
            st.session_state["result"] = {"pred": pred, "inputs": dict(vals), "probs": probs}

    res = st.session_state.get("result")
    if res:
        warns = [f"**{k}** = {v:g} is outside the training range ({ranges[k]['min']:g} – {ranges[k]['max']:g}); "
                 "the prediction may be less reliable." for k, v in res["inputs"].items()
                 if v < ranges[k]["min"] or v > ranges[k]["max"]]
        for w in warns:
            st.warning(w)
        st.markdown(f'<div class="result"><div class="sub">Recommended crop</div>'
                    f'<div class="name">🌾 {res["pred"].title()}</div></div>', unsafe_allow_html=True)

        a, b = st.columns(2)
        with a:
            st.subheader("Your inputs")
            show = pd.DataFrame({"Parameter": [s[1] for s in specs], "Value": [res["inputs"][s[0]] for s in specs],
                                 "Unit": [s[2] for s in specs]})
            st.dataframe(show, hide_index=True, use_container_width=True)
        with b:
            st.subheader("Crop information")
            info = lookup(res["pred"])
            if info:
                st.markdown(f"**About:** {info[0]}  \n**General conditions:** {info[1]}  \n**Note:** {info[2]}")
            else:
                st.info("No curated information is available for this crop label.")
            sub = df[df["label"] == res["pred"]]
            if len(sub):
                st.caption(f"Average values for **{res['pred']}** in the dataset ({len(sub)} records):")
                st.dataframe(sub[FEATURES].mean().round(1).to_frame("Mean").T, hide_index=True,
                             use_container_width=True)

        if res["probs"] is not None:
            st.subheader("Model-estimated probabilities (top 5)")
            fig = px.bar(res["probs"], x="Probability", y="Crop", orientation="h", color_discrete_sequence=[GREENS[1]])
            fig.update_layout(yaxis=dict(autorange="reversed"), height=280, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("These are the model's own estimates, not guarantees of real-world crop suitability.")

        if st.button("🔄 Make another prediction"):
            st.session_state.pop("result", None)
            st.rerun()

# ---------- ANALYTICS ----------
elif page == PAGES[2]:
    st.title("📊 Data Analytics")
    t1, t2, t3 = st.tabs(["Overview", "Distributions", "Relationships"])
    with t1:
        c = st.columns(3)
        metric(c[0], f"{df.shape[0]:,} × {df.shape[1]}", "Rows × columns (raw file)", "📐")
        metric(c[1], int(df.isna().sum().sum()), "Missing values", "❓")
        metric(c[2], int(df.duplicated().sum()), "Duplicate rows", "♊")
        st.subheader("First five records")
        st.dataframe(df.head(), use_container_width=True, hide_index=True)
        a, b = st.columns(2)
        a.subheader("Column data types")
        a.dataframe(df.dtypes.astype(str).rename("dtype").to_frame(), use_container_width=True)
        b.subheader("Summary statistics")
        b.dataframe(df[FEATURES].describe().round(2).T, use_container_width=True)
        counts = df["label"].value_counts().rename_axis("Crop").reset_index(name="Records")
        st.subheader("Crop label distribution")
        fig = px.bar(counts, x="Crop", y="Records", color_discrete_sequence=[GREENS[1]])
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        f = st.selectbox("Feature", FEATURES)
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(df, x=f, nbins=30, color_discrete_sequence=[GREENS[1]],
                                    title=f"Distribution of {f}"), use_container_width=True)
        b.plotly_chart(px.box(df, x="label", y=f, color_discrete_sequence=[GREENS[2]],
                              title=f"{f} by crop"), use_container_width=True)
    with t3:
        st.plotly_chart(px.imshow(df[FEATURES].corr(), text_auto=".2f", color_continuous_scale="Greens",
                                  title="Feature correlation"), use_container_width=True)
        a, b = st.columns(2)
        x = a.selectbox("X axis", FEATURES, index=3)
        y = b.selectbox("Y axis", FEATURES, index=6)
        st.plotly_chart(px.scatter(df, x=x, y=y, color="label", title=f"{x} vs {y}"), use_container_width=True)

# ---------- PERFORMANCE ----------
elif page == PAGES[3]:
    st.title("🏆 Model Performance")
    comp = pd.DataFrame(meta["comparison"])
    st.success(f"Selected model: **{meta['selected_model']}** — chosen by highest 5-fold cross-validated macro F1 "
               f"on the training data. Final metrics below come from the untouched test set "
               f"({meta['test_size']} records; train {meta['train_size']}; seed {meta['seed']}).")
    st.subheader("Model comparison (same train/test split)")
    st.dataframe(comp.style.format({c: "{:.4f}" for c in comp.columns if c != "Model"}),
                 hide_index=True, use_container_width=True)
    long = comp.melt(id_vars="Model", value_vars=["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"])
    st.plotly_chart(px.bar(long, x="Model", y="value", color="variable", barmode="group",
                           color_discrete_sequence=GREENS, labels={"value": "Score", "variable": "Metric"}),
                    use_container_width=True)
    st.subheader(f"Confusion matrix — {meta['selected_model']}")
    labs = meta["labels"]
    st.plotly_chart(px.imshow(meta["confusion_matrix"], x=labs, y=labs, text_auto=True,
                              color_continuous_scale="Greens",
                              labels=dict(x="Predicted", y="Actual", color="Count"), height=700),
                    use_container_width=True)
    st.subheader("Per-crop report (test set)")
    rep = pd.DataFrame(meta["classification_report"]).T
    st.dataframe(rep.round(3), use_container_width=True)
    st.caption("Very high scores on this dataset are common because crop classes are well separated; "
               "real-world performance on new regions may be lower.")

# ---------- ABOUT ----------
else:
    st.title("ℹ️ About Project")
    st.markdown(f"""
**Smart Crop Recommendation System Using Machine Learning** — a BTech CSE minor project.

**Dataset:** public *Crop Recommendation* CSV with {meta['rows_clean']:,} cleaned records, {meta['n_classes']} crops and
7 features (N, P, K, temperature, humidity, pH, rainfall).

**Pipeline:** load → inspect (shape, types, missing, duplicates, labels, invalid values) → clean → stratified 80/20 split
(seed {meta['seed']}) → train Decision Tree, Random Forest and KNN (scaling inside a Scikit-learn Pipeline to avoid
leakage) → compare → save the best with Joblib → load in this app.

**Tech:** Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit, Joblib.

**Limitations:**
- Trained on a single public dataset; may not generalise to every region or season.
- Ignores soil type, irrigation, pests, economics and local practice.
- Probabilities are model estimates, not guarantees.
- Crop notes are general and intentionally not precise cultivation guides.
""")
