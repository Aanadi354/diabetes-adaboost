import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================
# LOAD MODEL & ENCODER
# =========================
try:
    model = joblib.load('model2.pkl')
    gender_encoder = joblib.load('label_encoder_gender.pkl')
except FileNotFoundError:
    st.error("File model tidak ditemukan! Pastikan 'model2.pkl' dan 'label_encoder_gender.pkl' ada di folder yang sama.")
    st.stop()

# =========================
# SIDEBAR MENU
# =========================
st.sidebar.title("📌 Menu Aplikasi")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["ℹ️ Informasi Umum", "🩺 Prediksi Diabetes"]
)

# =========================
# HALAMAN INFORMASI UMUM
# =========================
if menu == "ℹ️ Informasi Umum":
    st.title("📊 Aplikasi Prediksi Diabetes")
    
    st.markdown("""
    ### Tentang Aplikasi
    Aplikasi ini digunakan untuk **memprediksi kemungkinan seseorang mengalami diabetes**  
    berdasarkan beberapa parameter kesehatan, yaitu:
    
    - **Jenis Kelamin**
    - **Usia**
    - **Kadar HbA1c**
    - **Kadar Gula Darah**
    
    Prediksi dilakukan menggunakan **model Machine Learning** yang telah dilatih sebelumnya.
    
    ---
    ### Tujuan Aplikasi
    ✅ Membantu deteksi dini diabetes  
    ✅ Sebagai media pembelajaran Machine Learning  
    ✅ Memberikan gambaran risiko kesehatan
    
    ---
    ⚠️ **Disclaimer:**  
    Aplikasi ini **hanya untuk tujuan edukasi** dan **bukan pengganti diagnosis dokter**.
    """)

# =========================
# HALAMAN PREDIKSI
# =========================
elif menu == "🩺 Prediksi Diabetes":
    st.title("🩺 Prediksi Diabetes")
    st.write("Silakan masukkan data pasien di bawah ini:")

    col1, col2 = st.columns(2)

    with col1:
        gender_options = ['Female', 'Male', 'Other']
        gender_input = st.selectbox("Jenis Kelamin", gender_options)

        age = st.slider("Usia (Tahun)", 0, 100, 30)

    with col2:
        hba1c_level = st.number_input("Kadar HbA1c", min_value=0.0, max_value=20.0, value=5.7, step=0.1)
        blood_glucose_level = st.number_input("Kadar Gula Darah", min_value=0, max_value=600, value=120)

    # =========================
    # ENCODE GENDER
    # =========================
    try:
        gender_encoded = gender_encoder.transform([gender_input])[0]
    except ValueError:
        st.error("Jenis kelamin tidak dikenali oleh sistem!")
        gender_encoded = None

    # =========================
    # BUTTON PREDIKSI
    # =========================
    if st.button("🔍 Prediksi Sekarang"):
        if gender_encoded is not None:
            input_data = pd.DataFrame([[gender_encoded, age, hba1c_level, blood_glucose_level]],
                                      columns=['gender', 'age', 'HbA1c_level', 'blood_glucose_level'])

            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

            st.markdown("---")
            st.subheader("📌 Hasil Prediksi:")

            if prediction[0] == 1:
                st.error(f"⚠️ **Terprediksi Diabetes**\n\nProbabilitas: **{prediction_proba[0][1]:.2f}**")
                st.write("💡 Disarankan untuk segera melakukan pemeriksaan ke tenaga medis.")
            else:
                st.success(f"✅ **Tidak Terprediksi Diabetes**\n\nProbabilitas: **{prediction_proba[0][0]:.2f}**")
                st.write("💡 Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin.")

        else:
            st.warning("Data tidak valid, silakan periksa kembali inputan Anda.")

    st.markdown("---")
    st.caption("⚠️ Aplikasi ini hanya untuk keperluan edukasi, bukan sebagai alat diagnosis medis.")
