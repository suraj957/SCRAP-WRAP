# 🧵 Scrap • Wrap — URL Q&A Chatbot

Scrap • Wrap is a lightweight Retrieval-Augmented Generation (RAG) chatbot that allows you to paste any webpage or PDF URL, embed its content into a local FAISS vector store, and chat naturally about it — fully locally or optionally powered by OpenAI.

Built with **Streamlit**, **LangChain**, **FAISS**, and **SQLite** for persistent chat history.

---

## 🚀 Features

- 🔗 **URL ingestion** — Scrape, clean, and embed text from any webpage or PDF  
- ⚙️ **FAISS index persistence per URL**  
- 💬 **ChatGPT-style interface** with memory & multiple conversations  
- 🧠 Supports **local (llama.cpp)** or **remote (OpenAI)** LLMs  
- ⚡ **Streaming responses** in real time  
- 📂 **Persistent chat history** stored in SQLite  
- 🧩 **Instant index load** using `load_index_only()` (no re-scraping)  
- 🧾 Works with both **HTML** and **PDFs**

---

## 🗂️ Folder Structure

url-chatbot/
├── app.py # Streamlit UI + chat logic
├── retriever.py # RAG pipeline + FAISS handling
├── scraper.py # Scraper (Trafilatura, BeautifulSoup, PDF)
├── storage.py # SQLite storage for conversations/messages
├── vectorstore/ # FAISS indexes (auto-created)
├── .env # Model + backend config
├── requirements.txt # Python dependencies
└── README.md


---

## 🧰 Prerequisites

- Python **3.9 – 3.11**
- Works on **macOS**, **Linux**, or **Windows (WSL)**
- For local models → **llama.cpp GGUF file**
- For remote models → **OpenAI API key**

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/url-chatbot.git
cd url-chatbot

# 2. Create a virtual environment
python -m venv .venv
# Activate:
#   Windows: .venv\Scripts\activate
#   macOS/Linux: source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Create folders
mkdir -p vectorstore models

## 🔑 .env Configuration

### 🧠 Local model (llama.cpp)

```env
LLM_BACKEND=llamacpp
MODEL_PATH=models/mistral-7b-instruct-v0.1.Q4_K_M.gguf

EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
STORE_DIR=vectorstore

# Performance tuning (optional)
N_THREADS=8
N_BATCH=512
N_GPU_LAYERS=20
USE_MMAP=true
USE_MLOCK=false
