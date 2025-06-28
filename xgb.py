import streamlit as st
import numpy as np
import joblib

# لود مدل‌ها
model_pp = joblib.load("model_pour_point.pkl")
model_visco = joblib.load("model_visco50.pkl")

# عنوان اپ
st.title("🧪 پیش‌بینی Pour Point و Visco 50 با مدل XGBoost")

st.markdown("لطفاً مقادیر ویژگی‌ها را وارد کنید 👇")

# تعریف فیلدهای ورودی بر اساس 19 فیچر
features = {
    "%VB": st.number_input("%VB", value=0.0),
    "Density Blend": st.number_input("Density Blend", value=0.0),
    "Total Sulphur": st.number_input("Total Sulphur", value=0.0),
    "Linear Visco": st.number_input("Linear Visco", value=0.0),
    "Core Visco": st.number_input("Core Visco", value=0.0),
    "Linear Pp": st.number_input("Linear Pp", value=0.0),
    "Corelation Pp": st.number_input("Corelation Pp", value=0.0),
}

# دکمه پیش‌بینی
if st.button("🔮 پیش‌بینی کن"):
    input_values = np.array([list(features.values())])
    
    pp_pred = model_pp.predict(input_values)[0]
    visco_pred = model_visco.predict(input_values)[0]
    
    st.success(f"✅ Pour Point پیش‌بینی شده: {pp_pred:.2f}")
    st.success(f"✅ Visco 50 پیش‌بینی شده: {visco_pred:.2f}")
