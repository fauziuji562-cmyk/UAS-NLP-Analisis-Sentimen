import streamlit as st
import pickle
import re
import time

# 1. Konfigurasi Halaman (Lebar, ada icon, judul tab)
st.set_page_config(page_title="Analisis Sentimen Pariwisata", page_icon="🏖️", layout="wide")

# 2. Menyembunyikan menu bawaan Streamlit agar terlihat seperti web profesional
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 3. SIDEBAR (Navigasi & Info Kelompok)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3069/3069172.png", width=100) # Ikon AI
    st.title("🧭 Panel Navigasi")
    st.markdown("---")
    st.subheader("👨‍💻 Tim Pengembang")
    st.write("1. Nama Anggota 1 (NIM)")
    st.write("2. Nama Anggota 2 (NIM)")
    st.write("3. Nama Anggota 3 (NIM)")
    st.markdown("---")
    st.info("📚 **Mata Kuliah:** Natural Language Processing (NLP)\n\n👨‍🏫 **Dosen:** Bapak Waziruddin")

# 4. HEADER UTAMA (Gambar Banner Pariwisata/Pantai)
st.image("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1200&q=80", use_column_width=True)
st.title("🌊 Sistem Analisis Sentimen Ulasan Wisata")
st.markdown("**Artificial Intelligence (AI)** untuk mendeteksi opini publik (Positif/Negatif) terhadap destinasi wisata berdasarkan ulasan pengunjung di Google Maps.")
st.markdown("---")

# 5. Fungsi Load Model (Di-cache agar web sangat cepat)
@st.cache_resource
def load_model():
    model = pickle.load(open('model_sentimen.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    return model, tfidf

model, tfidf = load_model()

# 6. LAYOUT BAGI DUA KOLOM (Kiri Input, Kanan Info)
col1, col2 = st.columns([2, 1])

# KOLOM KIRI (Area Input AI)
with col1:
    st.subheader("📝 Masukkan Ulasan Wisatawan")
    # Menggunakan form agar tombol enter lebih rapi
    with st.form("form_analisis"):
        ulasan = st.text_area("Ketik atau paste ulasan dari Google Maps di sini:", height=150, 
                              placeholder="Contoh: Pantai ini pemandangannya sangat luar biasa indah, tapi sayang sekali tempat parkirnya sempit dan banyak sampah berserakan...")
        submit_btn = st.form_submit_button("🔍 Analisis Sentimen AI")

    # Logika jika tombol ditekan
    if submit_btn and ulasan:
        with st.spinner('🤖 AI sedang membaca dan menganalisis teks...'):
            time.sleep(1) # Animasi loading buatan biar keren
            # Pembersihan teks
            teks_bersih = re.sub(r'[^a-z\s]', '', str(ulasan).lower())
            # Prediksi menggunakan model AI
            hasil = model.predict(tfidf.transform([teks_bersih]))[0]
            
            st.markdown("### 📊 Hasil Analisis AI:")
            if hasil == 'Positif':
                st.success("✨ **KATEGORI: POSITIF**")
                st.write("Wisatawan ini memiliki pengalaman yang menyenangkan dan memberikan opini baik.")
                st.balloons() # Muncul animasi balon
            else:
                st.error("🚨 **KATEGORI: NEGATIF**")
                st.write("Wisatawan ini menyampaikan keluhan atau memiliki pengalaman kurang menyenangkan.")

    elif submit_btn and not ulasan:
        st.warning("⚠️ Ulasan tidak boleh kosong! Silakan ketik sesuatu terlebih dahulu.")

# KOLOM KANAN (Info Tambahan & Ilustrasi Maps)
with col2:
    st.subheader("🗺️ Tentang Sistem")
    st.image("https://images.unsplash.com/photo-1524661135-423995f22d0b?auto=format&fit=crop&w=500&q=60", caption="Analisis Sentimen Berbasis Lokasi")
    st.markdown("""
    **Bagaimana AI ini bekerja?**
    1. **Input Data:** Sistem menerima teks ulasan mentah.
    2. **Preprocessing:** Teks dibersihkan dari simbol dan angka.
    3. **Ekstraksi Fitur:** Kata-kata diubah menjadi bobot matriks menggunakan algoritma **TF-IDF**.
    4. **Klasifikasi:** Model **Logistic Regression** memprediksi hasil akhir secara akurat.
    """)
