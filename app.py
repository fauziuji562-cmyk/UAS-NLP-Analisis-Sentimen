import streamlit as st
import pickle
import re
import time

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Analisis Sentimen Pariwisata", page_icon="🗺️", layout="wide")

# 2. CSS Custom untuk Background Full Layar dan Sembunyikan Menu
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Mengatur Background Keseluruhan Halaman (Tema Pantai/Maps) */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Efek kotak gelap transparan agar teks tetap jelas terbaca di atas gambar background */
    .main .block-container {
        background-color: rgba(20, 25, 30, 0.85);
        padding: 3rem;
        border-radius: 15px;
        color: #ffffff;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# 3. HEADER UTAMA & SUBJUDUL
st.title("📍 Google Maps Review Analyzer")
st.subheader("Sistem Analisis Sentimen Ulasan Wisata")
st.markdown("---")

# 4. Fungsi Load Model
@st.cache_resource
def load_model():
    model = pickle.load(open('model_sentimen.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    return model, tfidf

model, tfidf = load_model()

# 5. LAYOUT BAGI DUA KOLOM
col1, col2 = st.columns([1.5, 1])

# KOLOM KIRI (Area Input AI)
with col1:
    st.write("📝 **Masukkan Ulasan Wisatawan**")
    with st.form("form_analisis"):
        ulasan = st.text_area("Ketik atau paste ulasan dari Google Maps di sini:", height=150, 
                              placeholder="Contoh: Tempatnya bagus banget, pantainya bersih tapi akses jalan ke sana masih berlubang...")
        submit_btn = st.form_submit_button("🔍 Analisis Sentimen AI")

    # Logika jika tombol ditekan
    if submit_btn and ulasan:
        with st.spinner('🤖 AI sedang menganalisis teks...'):
            time.sleep(1) # Animasi loading
            # Pembersihan teks
            teks_bersih = re.sub(r'[^a-z\s]', '', str(ulasan).lower())
            # Prediksi menggunakan model AI
            hasil = model.predict(tfidf.transform([teks_bersih]))[0]
            
            st.markdown("### 📊 Hasil Prediksi:")
            if hasil == 'Positif':
                st.success("✨ **KATEGORI: POSITIF**\n\nBerdasarkan gaya bahasanya, ulasan ini memuat pengalaman pengunjung yang menyenangkan atau memuaskan.")
                st.balloons()
            else:
                st.error("🚨 **KATEGORI: NEGATIF**\n\nBerdasarkan gaya bahasanya, ulasan ini memuat keluhan, kritik, atau pengalaman buruk dari pengunjung.")

    elif submit_btn and not ulasan:
        st.warning("⚠️ Ulasan tidak boleh kosong! Silakan ketik sesuatu terlebih dahulu.")

# KOLOM KANAN (Informasi Lengkap Sistem & Ikon Maps)
with col2:
    # Menggunakan ikon aplikasi map sesuai permintaan
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Google_Maps_icon_%282020%29.svg/200px-Google_Maps_icon_%282020%29.svg.png", width=80)
    
    st.markdown("### 📌 Tentang Sistem")
    
    with st.expander("🎯 Fungsi & Tujuan", expanded=True):
        st.write("**Fungsi:** Mendeteksi dan mengklasifikasikan teks ulasan secara otomatis ke dalam sentimen Positif atau Negatif menggunakan *Machine Learning*.")
        st.write("**Tujuan:** Membantu pengelola destinasi wisata untuk memantau reputasi tempat mereka dengan cepat, serta membantu calon wisatawan mengambil keputusan tanpa harus membaca ribuan ulasan manual.")
        
    with st.expander("⚖️ Kelebihan & Kekurangan"):
        st.success("**Kelebihan:**\n- Analisis berjalan instan (Real-time).\n- Mampu memproses bahasa Indonesia karena telah dibekali teknik *Stemming* teks.\n- Antarmuka web ringan dan responsif.")
        st.error("**Kekurangan:**\n- AI masih kesulitan memahami bahasa gaul (*slang*), sarkasme tingkat tinggi, atau singkatan daerah yang sangat spesifik.\n- Akurasi model masih sangat bergantung pada variasi data latih (*dataset*) awal.")
