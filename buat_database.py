import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

nama_file = "zaed49f6a20bbfd69578313332353134.pdf"

print("1. Sedang membaca dan memotong dokumen...")
loader = PyPDFLoader(nama_file)
dokumen_mentah = loader.load()
pemotong_teks = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=400)
potongan_dokumen = pemotong_teks.split_documents(dokumen_mentah)
print(f"Dokumen berhasil dipecah menjadi {len(potongan_dokumen)} potongan (chunks).\n")

print("2. Sedang mengubah teks menjadi vektor dan menyimpan ke ChromaDB... ")
print("   (Proses ini membutuhkan internet dan mungkin memakan waktu 1-2 menit...))")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2", 
    google_api_key=api_key
)

vektor_db = Chroma.from_documents(
    documents=potongan_dokumen,
    embedding=embeddings,
    persist_directory="./db_hukum"
)

print("Sukses! Vektor telah disimpan ke ChromaDB di folder 'db_hukum'.")