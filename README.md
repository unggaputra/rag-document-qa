# ⚖️ AI Asisten Putusan Mahkamah Agung (RAG System)

Pengembangan lanjutan dari sistem Temu Kembali Informasi (*Information Retrieval*) yang sebelumnya berbasis Google Colab. Proyek ini telah direstrukturisasi menjadi skrip modular berbasis Python di lingkungan lokal, dengan rencana integrasi antarmuka menggunakan **Streamlit** untuk pengalaman pengguna yang lebih interaktif. Sistem ini dirancang untuk membaca, mengekstrak, dan menjawab pertanyaan spesifik berdasarkan dokumen putusan pengadilan secara akurat.

## 🚀 Fitur Utama
* **Mass PDF Ingestion Pipeline:** Pemrosesan *batch* puluhan dokumen PDF secara otomatis (`buat_database_massal.py`) yang dilengkapi dengan manajemen jeda waktu (*rate-limit handling*) untuk membangun *Vector Database* lokal secara aman.
* **Retrieval-Augmented Generation (RAG):** Menggunakan LangChain dan ChromaDB untuk mencari konteks hukum yang paling relevan dari putusan pengadilan sebelum merumuskan jawaban.
* **Interactive Web Interface:** Antarmuka *chat* interaktif menggunakan Streamlit (`app.py`) yang mempertahankan *session state* untuk mempermudah tanya jawab pengguna secara langsung.

## 🛠️ Teknologi yang Digunakan
* **Bahasa Pemrograman:** Python
* **Framework:** LangChain, Streamlit
* **AI & LLM:** Google Gemini API (`gemini-1.5-flash` untuk *generation*, `gemini-embedding-2` untuk *embedding*)
* **Vector Database:** ChromaDB (Penyimpanan lokal)
* **Sumber Data:** Dokumen Putusan Mahkamah Agung (PDF)

## ⚙️ Cara Menjalankan Proyek Secara Lokal

### 1. Persiapan Lingkungan (Setup)
Pastikan Python sudah terinstal, lalu instal seluruh *library* yang dibutuhkan:\
python -m pip install -r requirements.txt

### 2. Konfigurasi API Key
Buat file bernama .env di folder utama proyek, lalu masukkan Google Gemini API Key Anda:
GEMINI_API_KEY="masukkan_api_key_anda_di_sini"

### 3. Membangun Database (Data Ingestion)
Masukkan dokumen PDF putusan ke dalam folder data_pdf/. Jalankan skrip pemrosesan massal untuk membangun ChromaDB lokal:
"python buat_database_massal.py"

### 4. Menjalankan Antarmuka Web (Streamlit)
Setelah database db_hukum/ terbentuk, jalankan aplikasi web dengan perintah:
"python -m streamlit run app.py"

Aplikasi akan otomatis terbuka di browser pada alamat http://localhost:8501.
