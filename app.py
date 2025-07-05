import streamlit as st
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
import os

# --- UI HEADER ---
st.title("🧪 پیش‌بینی Pour Point و Visco 50 + سیستم آپدیت مدل")

st.sidebar.title("📂 عملیات")
menu = st.sidebar.radio("بخش مورد نظر:", ["پیش‌بینی", "آپدیت مدل"])

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    model_pp = joblib.load("model_pour_point.pkl")
    model_visco = joblib.load("model_visco50.pkl")
    return model_pp, model_visco

model_pp, model_visco = load_models()

# --- FEATURE LIST ---
feature_names = [
    "%VB",  "Density Blend",
    "Total Sulphur", "Linear Visco", "Core Visco", "Linear Pp", "Corelation Pp"
]

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
                model_pp_new = XGBRegressor(alpha=1e-05, base_score=0.5, booster='gbtree', callbacks=None, colsample_bylevel=0.9, colsample_bynode=1, colsample_bytree=0.88, early_stopping_rounds=None, enable_categorical=False, eta=0.01, eval_metric=None, feature_types=None, gamma=0, gpu_id=-1, grow_policy='lossguide', importance_type=None, interaction_constraints='', learning_rate=0.1, max_bin=20, max_cat_threshold=64, max_cat_to_onehot=4, max_delta_step=10, max_depth=10, max_leaves=0, min_child_weight=1, monotone_constraints='()', n_estimators=360, n_jobs=10, num_parallel_tree=10, subsample=0.99, objective='reg:squarederror')
                model_pp_new.fit(X, y_pp)

                # آموزش مدل Visco 50
                model_visco_new = XGBRegressor(alpha=1e-05, base_score=0.5, booster='gbtree', callbacks=None, colsample_bylevel=0.9, colsample_bynode=1, colsample_bytree=0.88, early_stopping_rounds=None, enable_categorical=False, eta=0.01, eval_metric=None, feature_types=None, gamma=0, gpu_id=-1, grow_policy='lossguide', importance_type=None, interaction_constraints='', learning_rate=0.1, max_bin=20, max_cat_threshold=64, max_cat_to_onehot=4, max_delta_step=10, max_depth=10, max_leaves=0, min_child_weight=1, monotone_constraints='()', n_estimators=360, n_jobs=10, num_parallel_tree=10, subsample=0.99, objective='reg:squarederror')
                model_visco_new.fit(X, y_visco)

                # ذخیره مدل‌ها
                joblib.dump(model_pp_new, "model_pour_point.pkl")
                joblib.dump(model_visco_new, "model_visco50.pkl")

                st.success("✅ مدل‌ها با موفقیت آپدیت شدند! برای استفاده، اپ را یک‌بار Refresh کنید.")

            else:
                st.error("❌ فایل اکسل باید شامل ستون‌های 'Pour Point' و 'Visco 50' باشد.")

        except Exception as e:
            st.error(f"❌ خطا در پردازش فایل: {e}")
