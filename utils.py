import os
import base64
from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

class PDFReport(FPDF):
    def header(self):
        # Set a safe font for headers
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'AKILLI ECZANE TIBBI ANALIZ RAPORU', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def setup_rag():
    """PDF'leri okur ve YEREL (Local) HuggingFace modelleri ile vektör veritabanını hazırlar."""
    try:
        corpus_path = "data/corpus"
        vector_path = "data/vector_db"
        
        if not os.path.exists(corpus_path): os.makedirs(corpus_path)
        if not os.path.exists(vector_path): os.makedirs(vector_path)
            
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Eğer vektör veritabanı zaten varsa doğrudan yükle
        if os.path.exists(os.path.join(vector_path, "index.faiss")):
            print("RAG: Mevcut FAISS veritabanı diskten yükleniyor...")
            vector_db = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
            return vector_db

        loader = DirectoryLoader(corpus_path, glob="./*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        if not documents:
            print("RAG: Veri klasörü bos, indeksleme atlanıyor.")
            return None
            
        print(f"RAG: {len(documents)} sayfa doküman yüklendi. Indeksleme yapılıyor...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        vector_db = FAISS.from_documents(
            documents=texts,
            embedding=embeddings
        )
        vector_db.save_local(vector_path)
        return vector_db
    except Exception as e:
        print(f"RAG Hatası: {e}")
        return None

def generate_pdf_report(analysis_data):
    """Analiz verilerini PDF formatına dönüştürür. Türkçe karakter uyumluluğu için temizlik yapar."""
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    
    # helper to clean turkish chars for helvetica
    def clean_txt(t):
        t = t.replace('ü', 'u').replace('ı', 'i').replace('ö', 'o').replace('ş', 's').replace('ç', 'c').replace('ğ', 'g').replace('Ü', 'U').replace('İ', 'I').replace('Ö', 'O').replace('Ş', 'S').replace('Ç', 'C').replace('Ğ', 'G')
        # Filtreleme: Latin-1 tablosunda olmayan karakterleri temizle (örn. Çince/Arapça/Emoji)
        t = t.encode('latin-1', 'ignore').decode('latin-1')
        return t

    lines = analysis_data.split('\n')
    for line in lines:
        cleaned_line = clean_txt(line).strip()
        if not cleaned_line:
            pdf.ln(5)
            continue
            
        # 30 karakterden uzun aralıksız metni boşlukla böl (fpdf yatay alan hatasını kesin önler)
        words = cleaned_line.split(' ')
        safe_words = []
        for w in words:
            if len(w) > 25:
                chunks = [w[i:i+25] for i in range(0, len(w), 25)]
                safe_words.extend(chunks)
            else:
                safe_words.append(w)
        safe_line = " ".join(safe_words)
            
        try:
            if safe_line.startswith('#'):
                pdf.set_font("Helvetica", 'B', 14)
                pdf.write(8, safe_line.replace('#', '').strip() + '\n')
                pdf.set_font("Helvetica", size=12)
            else:
                pdf.write(6, safe_line + '\n')
            pdf.ln(2) # Satır aralığı bırak
        except Exception as e:
            print(f"PDF Hatasi atlandi: {e} - Hatali line: {safe_line[:20]}")
            pdf.set_x(pdf.l_margin)
            pdf.ln(6)
                
    pdf_output = "data/reports/pharma_report.pdf"
    if not os.path.exists("data/reports"):
        os.makedirs("data/reports")
        
    pdf.output(pdf_output)
    return pdf_output
