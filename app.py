import streamlit as st
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
import math

# --- UI HEADER ---
st.title("🧪 پیش‌بینی Pour Point و Visco 50 + سیستم آپدیت مدل")

st.sidebar.title("📂 عملیات")
menu = st.sidebar.radio("بخش مورد نظر:", ["پیش‌بینی", "آپدیت مدل", "محاسبه بلندینگ"])

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    model_pp = joblib.load("model_pour_point.pkl")
    model_visco = joblib.load("model_visco50.pkl")
    return model_pp, model_visco

model_pp, model_visco = load_models()

# --- FEATURE LIST ---
feature_names = [
    "%VB", "%CSO", "%MC", "%ISO FEED", "%ISO REC", "%LUBCUT", "%CFO",
    "%CO", "%OIL", "%BN", "CUT 3", "CUT 2", "%RPO", "Density Blend",
    "Total Sulphur", "Linear Visco", "Core Visco", "Linear Pp", "Corelation Pp"
]

# --- محاسبات برای بلندینگ ---
def calculate_blending_features(num_parts, blending_data):
    # محاسبه %VB
    vb_sum = sum([part['Sulphur'] * part['Viscosity'] * part['Density'] for part in blending_data])
    vb = vb_sum / num_parts

    # محاسبه Total Sulphur
    total_sulphur = sum([part['Sulphur'] for part in blending_data])

    # محاسبه Linear Pour Point (محاسبه‌ شده به روش خطی)
    linear_pour_point = sum([part['Sulphur'] * part['Pour Point'] for part in blending_data])

    # محاسبه Correlation Pour Point
    correlation_pour_point = 0
    for part in blending_data:
        temp_rankine = (part['Pour Point'] + 273.15) * 1.8
        index = 3262000 * ((temp_rankine / 1000) ** 12.5)
        correlation_pour_point += (index * part['Sulphur'])

    correlation_pour_point = (((correlation_pour_point / 3262000) ** (1 / 12.5)) * 1000) / 1.8 - 273.15

    # محاسبه Correlation Viscosity
    correlation_viscosity = 0
    for part in blending_data:
        ln_visc = math.log(part['Viscosity'])  # گرفتن log ویسکوزیته
        correlation_viscosity += ln_visc * part['Sulphur']  # ضرب در درصد جرمی

    correlation_viscosity = math.exp(correlation_viscosity)  # گرفتن exp از مجموع

    return vb, total_sulphur, linear_pour_point, correlation_pour_point, correlation_viscosity


# --- بخش پیش‌بینی ---
if menu == "پیش‌بینی":
    st.subheader("📝 وارد کردن مقادیر ویژگی‌ها")

    features = [st.number_input(label, value=0.0) for label in feature_names]

    if st.button("🔮 پیش‌بینی کن"):
        input_array = np.array([features])
        pred_pp = model_pp.predict(input_array)[0]
        pred_visco = model_visco.predict(input_array)[0]

        st.success(f"✅ Pour Point پیش‌بینی شده: {pred_pp:.2f}")
        st.success(f"✅ Visco 50 پیش‌بینی شده: {pred_visco:.2f}")

# --- بخش آپدیت مدل ---
if menu == "آپدیت مدل":
    st.subheader("🔁 آپدیت مدل با فایل Excel جدید")

    uploaded_file = st.file_uploader("📄 فایل اکسل جدید را بارگذاری کنید", type=["xlsx"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            df = df.iloc[:, 1:]  # حذف ستون اول (مثل MT)

            if "Pour Point" in df.columns and "Visco 50" in df.columns:
                X = df.drop(columns=["Pour Point", "Visco 50"])
                y_pp = df["Pour Point"]
                y_visco = df["Visco 50"]

                # آموزش مدل Pour Point
                model_pp_new = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, objective='reg:squarederror')
                model_pp_new.fit(X, y_pp)

                # آموزش مدل Visco 50
                model_visco_new = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, objective='reg:squarederror')
                model_visco_new.fit(X, y_visco)

                # ذخیره مدل‌ها
                joblib.dump(model_pp_new, "model_pour_point.pkl")
                joblib.dump(model_visco_new, "model_visco50.pkl")

                st.success("✅ مدل‌ها با موفقیت آپدیت شدند! برای استفاده، اپ را یک‌بار Refresh کنید.")

            else:
                st.error("❌ فایل اکسل باید شامل ستون‌های 'Pour Point' و 'Visco 50' باشد.")

        except Exception as e:
            st.error(f"❌ خطا در پردازش فایل: {e}")

# --- بخش محاسبه بلندینگ ---
if menu == "محاسبه بلندینگ":
    st.subheader("🔄 محاسبه ویژگی‌های بلندینگ")

    num_parts = st.number_input("تعداد اجزای بلندینگ:", min_value=1, step=1)

    blending_data = []

    for i in range(num_parts):
        st.subheader(f"جزء {i + 1}")

        Sulphur = st.number_input(f"Sulphur برای جزء {i + 1}:", value=0.0)
        Viscosity = st.number_input(f"Viscosity برای جزء {i + 1}:", value=0.0)
        Density = st.number_input(f"Density برای جزء {i + 1}:", value=0.0)
        Pour_Point = st.number_input(f"Pour Point برای جزء {i + 1}:", value=0.0)

        blending_data.append({
            'Sulphur': Sulphur,
            'Viscosity': Viscosity,
            'Density': Density,
            'Pour Point': Pour_Point
        })

    if st.button("محاسبه ویژگی‌ها"):
        vb, total_sulphur, linear_pour_point, correlation_pour_point, correlation_viscosity = calculate_blending_features(num_parts, blending_data)
        st.write(f"ویژگی‌های محاسبه شده:")
        st.write(f"1. %VB: {vb:.2f}")
        st.write(f"2. Total Sulphur: {total_sulphur:.2f}")
        st.write(f"3. Linear Pour Point: {linear_pour_point:.2f}")
        st.write(f"4. Correlation Pour Point: {correlation_pour_point:.2f}")
        st.write(f"5. Correlation Viscosity: {correlation_viscosity:.2f}")

