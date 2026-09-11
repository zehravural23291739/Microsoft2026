# Local LLM RAG Projesi 

Bu proje, yerel (local) bir Büyük Dil Modeli (LLM) kullanarak belge tabanlı soru-cevap sistemi (RAG - Retrieval-Augmented Generation) sunan bir arka uç (backend) uygulamasıdır. Yüklenen PDF belgeleri dış sunuculara gitmeden, tamamen güvenli bir şekilde kendi bilgisayarınızda (offline) işlenir.

##  Kullanılan Teknolojiler

* **Backend Çerçevesi:** FastAPI, Python
* **Yapay Zeka Modeli:** Ollama (Llama 3)
* **LLM Entegrasyonu:** LangChain
* **Vektör Veritabanı:** ChromaDB
* **Metin İşleme:** PyPDFLoader, RecursiveCharacterTextSplitter, HuggingFaceEmbeddings

##  Kurulum ve Çalıştırma Adımları

**1. Projeyi Klonlayın**
```bash
git clone [https://github.com/zehravural23291739/Microsoft2026.git](https://github.com/zehravural23291739/Microsoft2026.git)
cd Microsoft2026
