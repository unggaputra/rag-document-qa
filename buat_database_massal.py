import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 1. Muat API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2. Siapkan Model Embedding
print("Menyiapkan penghubung ke database...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2", 
    google_api_key=api_key
)
vektor_db = Chroma(persist_directory="./db_hukum", embedding_function=embeddings)

# 3. Siapkan Alat Pemotong Teks
pemotong_teks = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)

# 4. Baca Folder PDF
folder_pdf = "./data_pdf"
semua_file = [f for f in os.listdir(folder_pdf) if f.endswith('.pdf')]

print(f"\n🚀 Ditemukan {len(semua_file)} file PDF. Mulai memproses...")
print("-" * 50)

# 5. Mulai Looping
for i, file_name in enumerate(semua_file, 1):
    file_path = os.path.join(folder_pdf, file_name)
    print(f"\n[{i}/{len(semua_file)}] Membaca: {file_name}...")
    
    try:
        # A. Ekstrak teks dari PDF
        loader = PyPDFLoader(file_path)
        dokumen = loader.load()
        
        # B. Potong teks
        potongan = pemotong_teks.split_documents(dokumen)
        print(f"   -> Ditemukan {len(potongan)} potongan teks. Memulai penyimpanan bertahap...")
        
        # C. BATCHING (Teknik Penyuapan Bertahap Anti-Error)
        ukuran_batch = 15 # Hanya kirim 15 potongan sekali jalan
        
        for j in range(0, len(potongan), ukuran_batch):
            # Ambil hanya 15 potongan
            batch_potongan = potongan[j : j + ukuran_batch]
            
            # Simpan ke database
            vektor_db.add_documents(batch_potongan)
            
            akhir_batch = min(j + ukuran_batch, len(potongan))
            print(f"      ✅ Potongan {j+1} s/d {akhir_batch} sukses disimpan.")
            
            # Jeda antar batch kecil (15 detik)
            if akhir_batch < len(potongan):
                time.sleep(15) 
        
        print(f"   🎉 File {file_name} SELESAI 100% masuk database!")
        
        # Jeda kecil antar file PDF yang berbeda
        if i < len(semua_file):
            print("   ⏳ Pindah ke file berikutnya... (Jeda 10 detik)")
            time.sleep(10)
            
    except Exception as e:
        print(f"   ❌ Gagal memproses {file_name}. Error: {e}")

print("\n🎉 SEMUA DOKUMEN TELAH SELESAI DIPROSES DAN MASUK KE DATABASE!")