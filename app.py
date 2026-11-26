import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 1. Konfigurasi Halaman
st.set_page_config(page_title="RAG Document QA", page_icon="⚖️")
st.title("⚖️ Asisten Tanya Jawab Dokumen Putusan")
load_dotenv()

def nama_file(path):
    return path.replace("\\", "/").split("/")[-1]

def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")

API_KEY = get_api_key()
if not API_KEY:
    st.error("GEMINI_API_KEY belum diatur.")
    st.stop()

# 2. Inisialisasi Database & LLM (di-cache)
@st.cache_resource
def get_db():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=API_KEY,
    )
    return Chroma(persist_directory="./db_hukum", embedding_function=embeddings)

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)

@st.cache_data
def get_stats():
    data = get_db().get(include=["metadatas"])
    files = {os.path.basename(m.get("source", "tidak diketahui")) for m in data["metadatas"]}
    return len(data["metadatas"]), sorted(files)

db = get_db()
llm = get_llm()

# Sidebar: info database (menjawab pertanyaan "ada berapa dokumen")
n_chunk, daftar_file = get_stats()
with st.sidebar:
    st.header("📚 Basis Data")
    st.write(f"**{len(daftar_file)}** putusan, **{n_chunk}** potongan teks")
    with st.expander("Lihat daftar dokumen"):
        for f in daftar_file:
            st.write("•", f)

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
        with st.spinner("Mencari di putusan..."):
            # MMR: hasil lebih beragam, tidak semua dari satu dokumen
            docs = db.max_marginal_relevance_search(prompt, k=6, fetch_k=20)

            context = "\n\n".join(
                f"[Sumber: {os.path.basename(d.metadata.get('source', '?'))}, "
                f"hlm. {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
                for d in docs
            )

            # Ikutkan beberapa percakapan terakhir agar pertanyaan lanjutan nyambung
            riwayat = "\n".join(
                f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:-1]
            )

            instruksi = f"""Anda adalah asisten yang menjawab pertanyaan tentang putusan Mahkamah Agung.
Aturan:
- Jawab HANYA berdasarkan konteks di bawah. Jangan menambah informasi dari luar.
- Jika konteks tidak cukup, katakan terus terang bahwa informasinya tidak ditemukan.
- Anda hanya melihat sebagian potongan dokumen, bukan seluruh database. Jangan menyimpulkan jumlah atau daftar dokumen dari konteks.
- Sebutkan nama dokumen dan halaman yang menjadi dasar jawaban.

Informasi basis data: berisi {len(daftar_file)} putusan ({n_chunk} potongan teks).
Daftar dokumen: {", ".join(daftar_file)}

Riwayat percakapan:
{riwayat}

Konteks:
{context}

Pertanyaan: {prompt}"""

            try:
                answer = llm.invoke(instruksi).content
            except Exception as e:
                answer = f"Terjadi kesalahan saat menghubungi model: {e}"

        st.markdown(answer)

        with st.expander("Sumber yang digunakan"):
            for d in docs:
                st.markdown(
                    f"**{os.path.basename(d.metadata.get('source', '?'))}**, "
                    f"hlm. {d.metadata.get('page', 0) + 1}"
                )
                st.caption(d.page_content[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
