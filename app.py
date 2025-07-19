import streamlit as st
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
import math

# --- Page Config & Custom CSS ---
st.set_page_config(page_title="Refinery Blend AI", layout="centered")
st.markdown("""
    <style>
    body { background-color: #1a1a1a; }
    .main { background-color: #1e1e1e; color: #f0f0f0; }
    h1, h2, h3, h4 { color: #e3e3e3; }
    .block-container { padding-top: 2rem; }
    .stTextInput > div > input, .stNumberInput input {
        background-color: #2c2c2c; color: white; border-radius: 6px;
    }
    .stButton button {
        background-color: #004080; color: white; border-radius: 10px; padding: 10px 16px;
    }
    .stButton button:hover { background-color: #0059b3; }
    .stRadio > div {
        background-color: #cccccc !important; color: #111 !important;
        padding: 12px; border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("🛠️ Control Panel")
menu = st.sidebar.radio("Choose Module:", ["📈 Prediction", "📤 Update Model", "🧪 Blending Calculator"])

# --- Load Models ---
@st.cache_resource
def load_models():
    model_pp = joblib.load("model_pour_point.pkl")
    model_visco = joblib.load("model_visco50.pkl")
    return model_pp, model_visco

model_pp, model_visco = load_models()

feature_names = ["%VB", "Density Blend", "Total Sulphur", "Linear Visco", "Core Visco", "Linear Pp", "Corelation Pp"]

# --- Blending Calculations ---
def calculate_blending_features(num_parts, blending_data):
    try:
        total_sulphur = sum([p['Sulphur'] * p['MassFraction'] for p in blending_data])
        linear_pp = sum([p['Pour Point'] * p['MassFraction'] for p in blending_data])
        linear_visc = sum([p['Viscosity'] * p['MassFraction'] for p in blending_data])

        bi_pp_blend = 0
        for p in blending_data:
            pp_r = (p['Pour Point'] + 273.15) * 1.8
            bi_pp_i = 3262000 * ((pp_r / 1000) ** 12.5)
            bi_pp_blend += bi_pp_i * p['MassFraction']
        pp_blend_r = ((bi_pp_blend / 3262000) ** (1 / 12.5)) * 1000
        corr_pp = (pp_blend_r / 1.8) - 273.15

        corr_visc = math.exp(sum([p['MassFraction'] * math.log(p['Viscosity']) for p in blending_data]))

        return total_sulphur, linear_pp, linear_visc, corr_pp, corr_visc
    except Exception as e:
        st.error(f"Calculation Error: {e}")
        return 0, 0, 0, 0, 0

# --- Session State Init ---
if "manual_data" not in st.session_state:
    st.session_state["manual_data"] = pd.DataFrame(columns=[
        "%VB", "Density Blend", "Total Sulphur", "Linear Visco", "Core Visco",
        "Visco 50", "Linear Pp", "Corelation Pp", "Pour Point"])

# --- Prediction ---
if menu == "📈 Prediction":
    st.header("📝 Manual Prediction Input")
    features = [st.number_input(label, value=0.0) for label in feature_names]
    if st.button("🔮 Predict"):
        input_array = np.array([features])
        st.success(f"🟦 Pour Point: `{model_pp.predict(input_array)[0]:.2f} °C`")
        st.success(f"🟨 Visco 50: `{model_visco.predict(input_array)[0]:.2f}`")

# --- Update Model ---
elif menu == "📤 Update Model":
    st.header("🧠 Model Update & Data Entry")
    mode = st.radio("Choose update method:", ["📄 Upload Excel", "✍️ Manual Add"], horizontal=True)

    if mode == "📄 Upload Excel":
        uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file).iloc[:, 1:]  # Drop first col if needed
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                st.session_state["manual_data"] = pd.concat([st.session_state["manual_data"], df], ignore_index=True)
                st.success("✅ Data added to training set.")
            except Exception as e:
                st.error(f"❌ File error: {e}")

    elif mode == "✍️ Manual Add":
        with st.form("manual_entry"):
            cols = st.columns(3)
            vb = cols[0].number_input("%VB", value=0.0)
            density = cols[1].number_input("Density Blend", value=0.0)
            sulphur = cols[2].number_input("Total Sulphur", value=0.0)
            lin_visco = cols[0].number_input("Linear Visco", value=0.0)
            cor_visco = cols[1].number_input("Core Visco", value=0.0)
            visco_50 = cols[2].number_input("Visco 50", value=0.0)
            lin_pp = cols[0].number_input("Linear Pp", value=0.0)
            cor_pp = cols[1].number_input("Corelation Pp", value=0.0)
            pp = cols[2].number_input("Pour Point", value=0.0)
            if st.form_submit_button("Add Row"):
                new_row = pd.DataFrame([{ "%VB": vb, "Density Blend": density, "Total Sulphur": sulphur,
                    "Linear Visco": lin_visco, "Core Visco": cor_visco, "Visco 50": visco_50,
                    "Linear Pp": lin_pp, "Corelation Pp": cor_pp, "Pour Point": pp }])
                st.session_state["manual_data"] = pd.concat([st.session_state["manual_data"], new_row], ignore_index=True)
                st.success("✅ Row added.")

    st.subheader("📊 Current Dataset")
    st.dataframe(st.session_state["manual_data"], use_container_width=True)

    if st.button("🚀 Retrain Models"):
        try:
            df = st.session_state["manual_data"].copy()
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            X = df.drop(columns=["Pour Point", "Visco 50"])
            y_pp = df["Pour Point"]
            y_visco = df["Visco 50"]

            model_pp_new = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4)
            model_pp_new.fit(X, y_pp)
            model_visco_new = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4)
            model_visco_new.fit(X, y_visco)

            joblib.dump(model_pp_new, "model_pour_point.pkl")
            joblib.dump(model_visco_new, "model_visco50.pkl")
            st.success("✅ Models retrained & saved.")
        except Exception as e:
            st.error(f"Training error: {e}")

# --- Blending Calculator ---
elif menu == "🧪 Blending Calculator":
    st.header("🔬 Calculate Blend Features")
    num_parts = st.number_input("Number of blend components", min_value=1, step=1)
    vb = st.number_input("Enter %VB for blend:", min_value=0.0, value=0.0)

    blending_data, total_mass = [], 0
    for i in range(num_parts):
        st.markdown(f"##### Component {i + 1}")
        cols = st.columns(5)
        sulphur = cols[0].number_input(f"Sulphur [{i+1}]", value=0.0)
        viscosity = cols[1].number_input(f"Viscosity [{i+1}]", min_value=0.001, value=1.0)
        density = cols[2].number_input(f"Density [{i+1}]", value=0.0)
        pour_point = cols[3].number_input(f"Pour Point (°C) [{i+1}]", value=0.0)
        mass = cols[4].number_input(f"Mass (kg) [{i+1}]", min_value=0.001, value=1.0)
        blending_data.append({
            'Sulphur': sulphur, 'Viscosity': viscosity, 'Density': density,
            'Pour Point': pour_point, 'Mass': mass })
        total_mass += mass

    for part in blending_data:
        part['MassFraction'] = part['Mass'] / total_mass if total_mass > 0 else 0

    if st.button("🧮 Calculate & Predict"):
        total_sulphur, linear_pp, linear_visc, corr_pp, corr_visc = calculate_blending_features(num_parts, blending_data)
        st.markdown("### 📑 Calculated Features")
        st.code(f"%VB: {vb:.2f}\nTotal Sulphur: {total_sulphur:.2f}\nLinear PP: {linear_pp:.2f} °C\n"
                f"Linear Viscosity: {linear_visc:.2f}\nCorrelation PP: {corr_pp:.2f} °C\n"
                f"Correlation Viscosity: {corr_visc:.2f}")

        input_features = np.array([[vb, 0, total_sulphur, linear_visc, corr_visc, linear_pp, corr_pp]])
        pred_pp = model_pp.predict(input_features)[0]
        pred_visco = model_visco.predict(input_features)[0]

        st.markdown("### 🧠 ML Predictions")
        st.success(f"🔥 Pour Point: `{pred_pp:.2f} °C`")
        st.success(f"🧊 Visco 50: `{pred_visco:.2f}`")
