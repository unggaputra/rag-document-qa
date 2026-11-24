import os, time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
emb = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)
db = Chroma(persist_directory="./db_hukum", embedding_function=emb)

# JANGAN diubah: pakai nilai chunk_size dan chunk_overlap yang sudah ada di file Anda sekarang
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)

BATCH = 10
TOKEN_PER_MENIT = 30000

def hapus_sebagian(path):
    try:
        db._collection.delete(where={"source": path})
    except Exception as e:
        print("  (gagal membersihkan sisa:", e, ")")

def tambah_batch(batch):
    for percobaan in range(5):
        try:
            db.add_documents(batch)
            return
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                tunggu = 60 * (percobaan + 1)
                print(f"  kena batas kuota, tunggu {tunggu} detik...")
                time.sleep(tunggu)
            else:
                raise
    raise RuntimeError("kuota tetap habis setelah 5 percobaan")

# Bersihkan sisa dari percobaan yang gagal tadi (HAPUS blok ini setelah proses selesai)
for f in ["zaed1d034a759ece9a56303833323238.pdf"]:
    hapus_sebagian(os.path.join("data_pdf", f))

sudah = {os.path.basename(m.get("source", "").replace("\\", "/"))
         for m in db.get(include=["metadatas"])["metadatas"]}
belum = [f for f in sorted(os.listdir("data_pdf"))
         if f.lower().endswith(".pdf") and f not in sudah]
print(f"{len(belum)} file akan diproses\n")

for f in belum:
    path = os.path.join("data_pdf", f)
    try:
        halaman = PyPDFLoader(path).load()
        potongan = [p for p in splitter.split_documents(halaman) if p.page_content.strip()]
        if not potongan:
            print(f"Lewati {f}: tidak ada teks")
            continue
        for i in range(0, len(potongan), BATCH):
            batch = potongan[i:i + BATCH]
            tambah_batch(batch)
            perkiraan_token = sum(len(p.page_content) for p in batch) / 3
            time.sleep(perkiraan_token / TOKEN_PER_MENIT * 60 * 1.3)
        print(f"OK: {f} ({len(potongan)} potongan)")
    except Exception as e:
        print(f"GAGAL {f}: {e}")
        hapus_sebagian(path)  # buang potongan setengah jadi supaya bisa diulang bersih