import streamlit as st
import pickle
import re

# Load model dan tfidf
model = pickle.load(open('model_sentimen.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

st.title("🌴 Aplikasi Analisis Sentimen Ulasan")
ulasan = st.text_area("Masukkan ulasan wisata di sini:")

if st.button("Analisis"):
    # Pembersihan teks
    teks_bersih = re.sub(r'[^a-z\s]', '', str(ulasan).lower())
    # Prediksi
    hasil = model.predict(tfidf.transform([teks_bersih]))[0]
    st.write(f"Hasil: **{hasil}**")