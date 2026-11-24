import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
emb = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)
db = Chroma(persist_directory="./db_hukum", embedding_function=emb)

sudah = {os.path.basename(m.get("source", "")) for m in db.get(include=["metadatas"])["metadatas"]}
semua = sorted(f for f in os.listdir("data_pdf") if f.lower().endswith(".pdf"))
belum = [f for f in semua if f not in sudah]

print(f"Sudah masuk: {len(sudah)} | Total PDF: {len(semua)} | Belum: {len(belum)}\n")
for f in belum:
    try:
        halaman = PyPDFLoader(os.path.join("data_pdf", f)).load()
        teks = sum(len(h.page_content.strip()) for h in halaman)
        status = "TEKS KOSONG (kemungkinan hasil scan)" if teks == 0 else f"ada teks ({teks} karakter, {len(halaman)} hlm)"
    except Exception as e:
        status = f"GAGAL DIBACA: {e}"
    print(f"- {f}: {status}")