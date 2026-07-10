from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

nama_file = "zaed49f6a20bbfd69578313332353134.pdf"
print(f"Sedang membaca {nama_file}...")

try:
    loader = PyPDFLoader(nama_file)
    dokumen_mentah = loader.load()
    print(f"Berhasil! Dokumen ini memiliki {len(dokumen_mentah)} halaman.\n")

    print("Sedang memecah dokumen menjadi potongan-potongan teks...")
    print("Sedang memotong dokumen menjadi bagian-bagian kecil...")

    pemotong_teks = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Ukuran setiap potongan teks
        chunk_overlap=200,  # Tumpang tindih antara potongan teks
        length_function=len  # Fungsi untuk menghitung panjang teks
    )

    potongan_dokumen = pemotong_teks.split_documents(dokumen_mentah)
    print(f"Selesai! Dokumen berhasil dipecah menjadi {len(potongan_dokumen)} potongan (chunks).\n")

    print("- Contoh Potongan Pertama")
    print(potongan_dokumen[0].page_content)
    print("-----------------------------------------------\n")

except FileNotFoundError:
    print(f"Error: File {nama_file} tidak ditemukan. Pastikan namanya benar dan ada di folder yang sama.")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")