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
    "%VB", "Density Blend",
    "Total Sulphur", "Linear Visco", "Core Visco", "Linear Pp", "Corelation Pp"
]

# --- محاسبات برای بلندینگ ---
def calculate_blending_features(num_parts, blending_data):
    # محاسبه Total Sulphur
    total_sulphur = sum([part['Sulphur'] * part['MassFraction'] for part in blending_data])

    # محاسبه Linear Pour Point
    linear_pour_point = sum([
        part['Pour Point'] * part['MassFraction'] for part in blending_data
    ])

    # محاسبه BI_PP (براساس فرمول اصلاح‌شده)
    bi_pp_blend = 0
    for part in blending_data:
        pp_c = part['Pour Point']  # مقدار ورودی به سلسیوس است
        pp_rankine = (pp_c + 273.15) * 1.8  # تبدیل به Rankine
        bi_pp_i = 3262000 * ((pp_rankine / 1000) ** 12.5)
        bi_pp_blend += bi_pp_i * part['MassFraction']

    # محاسبه Pour Point بلندی (برگرداندن از BI به PP در Rankine و بعد به سلسیوس)
    pp_blend_rankine = ((bi_pp_blend / 3262000) ** (1 / 12.5)) * 1000
    correlation_pour_point = (pp_blend_rankine / 1.8) - 273.15  # بازگشت به سلسیوس

    # محاسبه Visco Blend
    correlation_viscosity = 0
    for part in blending_data:
        ln_visc = math.log(part['Viscosity'])
        correlation_viscosity += part['MassFraction'] * ln_visc

    correlation_viscosity = math.exp(correlation_viscosity)

    return total_sulphur, linear_pour_point, correlation_pour_point, correlation_viscosity

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
# --- بخش محاسبه بلندینگ ---
if menu == "محاسبه بلندینگ":
    st.subheader("🔄 محاسبه ویژگی‌های بلندینگ")

    num_parts = st.number_input("تعداد اجزای بلندینگ:", min_value=1, step=1)

    # گرفتن %VB مستقیم از کاربر
    vb = st.number_input("مقدار %VB برای کل بلند:", min_value=0.0, value=0.0)

    blending_data = []
    total_mass = 0

    for i in range(num_parts):
        st.subheader(f"جزء {i + 1}")

        Sulphur = st.number_input(f"Sulphur برای جزء {i + 1}:", value=0.0)
        Viscosity = st.number_input(f"Viscosity برای جزء {i + 1}:", min_value=0.001, value=1.0)
        Density = st.number_input(f"Density برای جزء {i + 1}:", value=0.0)
        Pour_Point = st.number_input(f"Pour Point برای جزء {i + 1} (°C):", value=0.0)
        Mass = st.number_input(f"جرم (kg) برای جزء {i + 1}:", min_value=0.001, value=1.0)

        blending_data.append({
            'Sulphur': Sulphur,
            'Viscosity': Viscosity,
            'Density': Density,
            'Pour Point': Pour_Point,
            'Mass': Mass
        })
        total_mass += Mass

    for part in blending_data:
        part['MassFraction'] = part['Mass'] / total_mass if total_mass > 0 else 0

    def calculate_blending_features(num_parts, blending_data):
        try:
            # Total Sulphur
            total_sulphur = sum([part['Sulphur'] * part['MassFraction'] for part in blending_data])

            # Linear Pour Point
            linear_pour_point = sum([
                part['Pour Point'] * part['MassFraction'] for part in blending_data
            ])

            # Linear Viscosity
            linear_viscosity = sum([
                part['Viscosity'] * part['MassFraction'] for part in blending_data
            ])

            # Correlation Pour Point
            bi_pp_blend = 0
            for part in blending_data:
                pp_c = part['Pour Point']
                pp_rankine = (pp_c + 273.15) * 1.8
                if pp_rankine <= 0:
                    raise ValueError(f"Pour Point Rankine نامعتبر برای جزء با مقدار {pp_c}")
                bi_pp_i = 3262000 * ((pp_rankine / 1000) ** 12.5)
                bi_pp_blend += bi_pp_i * part['MassFraction']

            pp_blend_rankine = ((bi_pp_blend / 3262000) ** (1 / 12.5)) * 1000
            correlation_pour_point = (pp_blend_rankine / 1.8) - 273.15

            # Correlation Viscosity
            correlation_viscosity = 0
            for part in blending_data:
                if part['Viscosity'] <= 0:
                    raise ValueError(f"ویسکوزیته باید > 0 باشد. مقدار نامعتبر: {part['Viscosity']}")
                ln_visc = math.log(part['Viscosity'])
                correlation_viscosity += part['MassFraction'] * ln_visc

            correlation_viscosity = math.exp(correlation_viscosity)

            return total_sulphur, linear_pour_point, linear_viscosity, correlation_pour_point, correlation_viscosity

        except Exception as e:
            st.error(f"❌ خطا در محاسبات: {e}")
            return 0, 0, 0, 0, 0

    if st.button("محاسبه ویژگی‌ها"):
        total_sulphur, linear_pour_point, linear_viscosity, correlation_pour_point, correlation_viscosity = calculate_blending_features(num_parts, blending_data)

        st.write(f"ویژگی‌های محاسبه شده:")
        st.write(f"1. %VB (ورودی کاربر): {vb:.2f}")
        st.write(f"2. Total Sulphur: {total_sulphur:.2f}")
        st.write(f"3. Linear Pour Point: {linear_pour_point:.2f} °C")
        st.write(f"4. Linear Viscosity: {linear_viscosity:.2f}")
        st.write(f"5. Correlation Pour Point: {correlation_pour_point:.2f} °C")
        st.write(f"6. Correlation Viscosity: {correlation_viscosity:.2f}")

