import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

print("Mencari daftar model EMBEDDING yang tersedia untukmu...\n")

try:
    models = client.models.list()
    ditemukan = False
    for m in models:
        # Kita hanya menyaring model yang memiliki kata "embed" di namanya
        if "embed" in m.name.lower():
            print(f"👉 NAMA MODEL VALID: {m.name}")
            ditemukan = True
            
    if not ditemukan:
        print("Tidak ada model embedding yang ditemukan.")
except Exception as e:
    print(f"Error: {e}")