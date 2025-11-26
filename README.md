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


---

## 🧰 Prerequisites

- Python **3.9 – 3.11**
- Works on **macOS**, **Linux**, or **Windows (WSL)**
- For local models → **llama.cpp GGUF file**
- For remote models → **OpenAI API key**

---

## ⚙️ Installations

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

---

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

---

LLM_BACKEND=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini

EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
STORE_DIR=vectorstore

---

## 🏃‍♀️ Running the App

```bash
streamlit run app.py

---
## 💬 How to Use

### **1. Embed a URL**
- Paste any webpage or PDF link in the sidebar  
- Click **Embed**  
- The content is scraped, cleaned, chunked, embedded, and saved locally in `vectorstore/`

---

### **2. Ask Questions**
- Type your question in the chat input field  
- The model returns **streaming responses** in real-time  
- Every message is automatically stored in `chat.db`

---

### **3. Manage Chat History**
- Create **new chat threads**  
- Switch between different conversations  
- Rename or delete previous chats  
- Each chat is maintained separately and linked to its own URL index

---

## ⚡ Performance Tips

| Area | Recommendation |
|------|---------------|
| **Local LLM** | Use smaller `.gguf` models (e.g., `Phi-3.5-mini-instruct.Q4_K_M.gguf`) |
| **Threads** | Set `N_THREADS` equal to your CPU core count |
| **GPU Offload** | Set `N_GPU_LAYERS=20–40` if Metal/CUDA is available |
| **Context Length** | Reduce `n_ctx` to `2048` for faster inference |
| **Chunking** | Use `chunk_size=500` and `chunk_overlap=80` |
| **Skip Re-scrape** | Use `load_index_only()` for repeat Q&A |
| **OpenAI Backend** | `gpt-4o-mini` is **3–5× faster** than local llama.cpp models |

---

## 🧱 Key Components

| File | Description |
|------|-------------|
| **app.py** | Streamlit chat UI with persistent sessions & streaming responses |
| **scraper.py** | HTML/PDF scraper using Trafilatura, BeautifulSoup, and PyPDF |
| **retriever.py** | Semantic embeddings, FAISS vector search, and llama.cpp/OpenAI inference |
| **storage.py** | SQLite database for storing conversations and messages |

---

## 🧠 Useful Commands
```bash
# Rebuild all indexes
rm -rf vectorstore && mkdir vectorstore

# Reset all chat history
rm chat.db

# Upgrade dependencies
pip install -r requirements.txt --upgrade

---

## 🧩 Requirements Summary
```shell
streamlit
beautifulsoup4
requests
python-dotenv
langchain
langchain-community
langchain-core
sentence-transformers
faiss-cpu
llama-cpp-python>=0.2.57
pypdf
trafilatura

## 🧾 License
MIT © 2025 — Built for local knowledge retrieval and experimentation.
Made with ❤️ using Streamlit + LangChain.