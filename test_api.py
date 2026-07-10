import os
from google import genai
from dotenv import load_dotenv

# 1. Memuat API Key dari file .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# 2. Inisialisasi client Gemini versi TERBARU
client = genai.Client(api_key=API_KEY)

# 3. Memberikan perintah (Prompt)
prompt = "Jelaskan apa itu Retrieval-Augmented Generation (RAG) dalam satu kalimat sederhana."
print(f"Pertanyaan: {prompt}\n")
print("Sedang berpikir...\n")

# 4. Mendapatkan jawaban dari model
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt
)
print(f"Jawaban AI: {response.text}")