import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 1. Load Database
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=api_key)
vektor_db = Chroma(persist_directory="./db_hukum", embedding_function=embeddings)

# 2. Cari dokumen relevan (Retrieval)
pertanyaan = "Siapa nama terdakwa dan apa hukumannya?"
docs = vektor_db.similarity_search(pertanyaan, k=5)
context_text = "\n\n".join([d.page_content for d in docs])

# 3. Kirim ke Gemini (Generation - Tanpa Chain)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
jawaban = llm.invoke(f"Gunakan dokumen ini: {context_text}. Jawab pertanyaan: {pertanyaan}")

print("\n--- JAWABAN AI ---\n", jawaban.content)