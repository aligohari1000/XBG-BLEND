import streamlit as st
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
import math

# --- App Header ---
st.set_page_config(page_title="Blend Predictor", layout="centered")
st.title("🧪 Pour Point & Visco 50 Prediction + Model Update System")

# --- Sidebar Navigation ---
st.sidebar.title("📂 Navigation")
menu = st.sidebar.radio("Select a section:", ["Prediction", "Update Model", "Blending Calculator"])

# --- Load Models ---
@st.cache_resource
def load_models():
    model_pp = joblib.load("model_pour_point.pkl")
    model_visco = joblib.load("model_visco50.pkl")
    return model_pp, model_visco

model_pp, model_visco = load_models()

# --- Feature List ---
feature_names = [
    "%VB", "Density Blend",
    "Total Sulphur", "Linear Visco", "Core Visco", "Linear Pp", "Corelation Pp"
]

# --- Blending Calculations ---
def calculate_blending_features(num_parts, blending_data):
    try:
        total_sulphur = sum([part['Sulphur'] * part['MassFraction'] for part in blending_data])
        linear_pour_point = sum([part['Pour Point'] * part['MassFraction'] for part in blending_data])
        linear_viscosity = sum([part['Viscosity'] * part['MassFraction'] for part in blending_data])

        # Correlation Pour Point
        bi_pp_blend = 0
        for part in blending_data:
            pp_c = part['Pour Point']
            pp_rankine = (pp_c + 273.15) * 1.8
            bi_pp_i = 3262000 * ((pp_rankine / 1000) ** 12.5)
            bi_pp_blend += bi_pp_i * part['MassFraction']

        pp_blend_rankine = ((bi_pp_blend / 3262000) ** (1 / 12.5)) * 1000
        correlation_pour_point = (pp_blend_rankine / 1.8) - 273.15

        # Correlation Viscosity
        correlation_viscosity = 0
        for part in blending_data:
            ln_visc = math.log(part['Viscosity'])
            correlation_viscosity += part['MassFraction'] * ln_visc
        correlation_viscosity = math.exp(correlation_viscosity)

        return total_sulphur, linear_pour_point, linear_viscosity, correlation_pour_point, correlation_viscosity
    except Exception as e:
        st.error(f"Calculation Error: {e}")
        return 0, 0, 0, 0, 0

# --- Prediction Section ---
if menu == "Prediction":
    st.subheader("📝 Enter Feature Values")

    features = [st.number_input(label, value=0.0) for label in feature_names]

    if st.button("🔮 Predict"):
        input_array = np.array([features])
        pred_pp = model_pp.predict(input_array)[0]
        pred_visco = model_visco.predict(input_array)[0]

        st.success(f"✅ Predicted Pour Point: {pred_pp:.2f} °C")
        st.success(f"✅ Predicted Visco 50: {pred_visco:.2f}")

# --- Update Model Section ---
elif menu == "Update Model":
    st.subheader("🔁 Update Models")

    # Initialize session state for manual data
    if "manual_data" not in st.session_state:
        st.session_state["manual_data"] = pd.DataFrame(columns=[
            "%VB", "Density Blend", "Total Sulphur", "Linear Visco", "Core Visco",
            "Visco 50", "Linear Pp", "Corelation Pp", "Pour Point"
        ])

    mode = st.radio("Choose update method:", ["📄 Upload Excel", "✍️ Manual Add"])

    if mode == "📄 Upload Excel":
        uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                df = df.iloc[:, 1:]

                expected_columns = st.session_state["manual_data"].columns.tolist()
                if all(col in df.columns for col in expected_columns):
                    st.session_state["manual_data"] = pd.concat([st.session_state["manual_data"], df], ignore_index=True)
                    st.success("✅ Data added to training set.")
                else:
                    st.error(f"❌ Missing required columns. Expected: {expected_columns}")
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

    elif mode == "✍️ Manual Add":
        with st.form("manual_add_form"):
            st.markdown("### ➕ Add a New Row")

            vb = st.number_input("%VB", value=0.0)
            density = st.number_input("Density Blend", value=0.0)
            sulphur = st.number_input("Total Sulphur", value=0.0)
            lin_visco = st.number_input("Linear Visco", value=0.0)
            cor_visco = st.number_input("Core Visco", value=0.0)
            visco_50 = st.number_input("Visco 50", value=0.0)
            lin_pp = st.number_input("Linear Pp", value=0.0)
            cor_pp = st.number_input("Corelation Pp", value=0.0)
            pp = st.number_input("Pour Point", value=0.0)

            submitted = st.form_submit_button("Add Row")
            if submitted:
                new_row = pd.DataFrame([{
                    "%VB": vb,
                    "Density Blend": density,
                    "Total Sulphur": sulphur,
                    "Linear Visco": lin_visco,
                    "Core Visco": cor_visco,
                    "Visco 50": visco_50,
                    "Linear Pp": lin_pp,
                    "Corelation Pp": cor_pp,
                    "Pour Point": pp
                }])
                st.session_state["manual_data"] = pd.concat([st.session_state["manual_data"], new_row], ignore_index=True)
                st.success("✅ Row added successfully!")

    st.markdown("### 📊 Current Training Dataset")
    st.write(f"Total rows: {len(st.session_state['manual_data'])}")
    st.dataframe(st.session_state["manual_data"], use_container_width=True)

    if st.button("🔄 Retrain Models"):
        try:
            df = st.session_state["manual_data"]
            X = df.drop(columns=["Pour Point", "Visco 50"])
            y_pp = df["Pour Point"]
            y_visco = df["Visco 50"]

            model_pp_new = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, objective='reg:squarederror')
            model_pp_new.fit(X, y_pp)

            model_visco_new = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, objective='reg:squarederror')
            model_visco_new.fit(X, y_visco)

            joblib.dump(model_pp_new, "model_pour_point.pkl")
            joblib.dump(model_visco_new, "model_visco50.pkl")

            st.success("✅ Models retrained and saved successfully!")
        except Exception as e:
            st.error(f"❌ Error during training: {e}")

# --- Blending Section ---
elif menu == "Blending Calculator":
    st.subheader("🧪 Blending Feature Calculator")

    num_parts = st.number_input("Number of blend components:", min_value=1, step=1)
    vb = st.number_input("Enter %VB for the blend:", min_value=0.0, value=0.0)

    blending_data = []
    total_mass = 0

    for i in range(num_parts):
        st.markdown(f"### Component {i + 1}")
        sulphur = st.number_input(f"Sulphur [{i+1}]:", value=0.0)
        viscosity = st.number_input(f"Viscosity [{i+1}]:", min_value=0.001, value=1.0)
        density = st.number_input(f"Density [{i+1}]:", value=0.0)
        pour_point = st.number_input(f"Pour Point (°C) [{i+1}]:", value=0.0)
        mass = st.number_input(f"Mass (kg) [{i+1}]:", min_value=0.001, value=1.0)

        blending_data.append({
            'Sulphur': sulphur,
            'Viscosity': viscosity,
            'Density': density,
            'Pour Point': pour_point,
            'Mass': mass
        })
        total_mass += mass

    for part in blending_data:
        part['MassFraction'] = part['Mass'] / total_mass if total_mass > 0 else 0

    if st.button("🔍 Calculate & Predict"):
        total_sulphur, linear_pp, linear_visc, corr_pp, corr_visc = calculate_blending_features(num_parts, blending_data)

        st.markdown("### 🧾 Calculated Features")
        st.write(f"**%VB (user input):** {vb:.2f}")
        st.write(f"**Total Sulphur:** {total_sulphur:.2f}")
        st.write(f"**Linear Pour Point:** {linear_pp:.2f} °C")
        st.write(f"**Linear Viscosity:** {linear_visc:.2f}")
        st.write(f"**Correlation Pour Point:** {corr_pp:.2f} °C")
        st.write(f"**Correlation Viscosity:** {corr_visc:.2f}")

        input_features = np.array([[vb, 0, total_sulphur, linear_visc, corr_visc, linear_pp, corr_pp]])
        pred_pp = model_pp.predict(input_features)[0]
        pred_visco = model_visco.predict(input_features)[0]

        st.markdown("### 🔮 ML Predictions")
        st.success(f"**Predicted Pour Point:** {pred_pp:.2f} °C")
        st.success(f"**Predicted Visco 50:** {pred_visco:.2f}")
