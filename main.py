import uvicorn
from fastapi import FastAPI
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="Local LLM RAG Projesi", version="1.0")

# Ollama'yı yeni ve güncel modülüyle çağırıyoruz
llm = OllamaLLM(model="llama3")

# Vektör veritabanımızı global bir değişken olarak tanımlıyoruz
vektor_veritabani = None

@app.post("/pdf-yukle")
def pdf_yukle():
    global vektor_veritabani
    
    # 1. PDF'i Oku
    loader = PyPDFLoader("ornek.pdf")
    sayfalar = loader.load()
    
    # 2. Metni Parçalara Ayır (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    parcalar = text_splitter.split_documents(sayfalar)
    
    # 3. Metinleri Vektöre Çevir ve ChromaDB'ye Kaydet (Yeni Modül)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vektor_veritabani = Chroma.from_documents(parcalar, embeddings)
    
    return {"mesaj": "PDF başarıyla okundu, parçalandı ve vektör veritabanına kaydedildi!"}

@app.get("/rag-soru-sor")
def rag_soru_sor(soru: str):
    if not vektor_veritabani:
        return {"hata": "Önce /pdf-yukle ucuyla sistemi beslemelisin!"}
        
    prompt = PromptTemplate.from_template(
        "Aşağıdaki bağlamı (context) kullanarak soruyu cevapla. Cevabı bağlamda bulamazsan kendi bilgini kullanma ve sadece 'Bilmiyorum' de.\n\nBağlam: {context}\n\nSoru: {input}\n\nCevap:"
    )
    
    retriever = vektor_veritabani.as_retriever()
    
    # PDF'ten gelen parçaları düz metne çeviren yardımcı fonksiyon
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # Eski 'chains' yerine en modern LCEL (LangChain Expression Language) yapısı
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Soruyu doğrudan yeni zincire gönderiyoruz
    cevap = rag_chain.invoke(soru)
    
    return {"soru": soru, "cevap": cevap}

# Sunucuyu başlatan kritik kısımpython main.py
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)