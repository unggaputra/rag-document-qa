import os
from google import genai
from dotenv import load_dotenv

# 1. Memuat API Key dari file .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# 2. Inisialisasi client
client = genai.Client(api_key=API_KEY)

print("Mencari daftar model yang tersedia untuk API Key ini...\n")

# 3. Menarik daftar model langsung dari server Google
try:
    models = client.models.list()
    for m in models:
        print(f"- {m.name}")
except Exception as e:
    print(f"Gagal mengambil daftar model: {e}")