import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 1. Konfigurasi Halaman
st.set_page_config(page_title="AI Hukum MA", page_icon="⚖️")
st.title("⚖️ AI Asisten Putusan Mahkamah Agung")
load_dotenv()

# 2. Inisialisasi Database (Caching agar cepat)
@st.cache_resource
def get_db():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2", 
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    return Chroma(persist_directory="./db_hukum", embedding_function=embeddings)

db = get_db()
# Ubah dari gemini-1.5-pro ke gemini-1.5-flash
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

# 3. Antarmuka Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanya tentang putusan pengadilan..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Retrieval
        docs = db.similarity_search(prompt, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        
        # Generation
        response = llm.invoke(f"Gunakan dokumen ini untuk menjawab: {context}\n\nPertanyaan: {prompt}")
        st.markdown(response.content)
        
    st.session_state.messages.append({"role": "assistant", "content": response.content})