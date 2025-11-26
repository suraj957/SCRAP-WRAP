# retriever.py
import os
import json
from hashlib import sha256
from typing import List, Optional

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Version-safe import for LangChain callbacks
try:
    from langchain.callbacks.base import BaseCallbackHandler
except Exception:
    from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

# -----------------------------
# Config
# -----------------------------
STORE_DIR   = os.getenv("STORE_DIR", "vectorstore")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_BACKEND = os.getenv("LLM_BACKEND", "llamacpp")  # "llamacpp" | "openai"
MODEL_PATH  = os.getenv("MODEL_PATH", "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

os.makedirs(STORE_DIR, exist_ok=True)

# Embeddings model
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# -----------------------------
# Helpers for persistent storage
# -----------------------------
def _url_key(url: str) -> str:
    """Return a short hash for the URL (used as filename)."""
    return sha256(url.encode("utf-8")).hexdigest()[:16]

def _store_path(url: str) -> str:
    return os.path.join(STORE_DIR, f"{_url_key(url)}.faiss")

def _meta_path(url: str) -> str:
    return os.path.join(STORE_DIR, f"{_url_key(url)}.json")

def _save_meta(url: str, meta: dict):
    with open(_meta_path(url), "w", encoding="utf-8") as f:
        json.dump(meta, f)

def _load_meta(url: str) -> dict:
    p = _meta_path(url)
    return json.load(open(p, "r", encoding="utf-8")) if os.path.exists(p) else {}

# -----------------------------
# LLM loader (with streaming support)
# -----------------------------
def _get_llm(callbacks: Optional[List[BaseCallbackHandler]] = None):
    """
    Return an LLM instance for the selected backend.
    Supports:
      - llama.cpp local models (.gguf)
      - OpenAI API (requires OPENAI_API_KEY)
    """
    if LLM_BACKEND == "llamacpp":
        # Perf knobs via env (optional)
        n_threads    = int(os.getenv("N_THREADS", str(os.cpu_count() or 4)))
        n_batch      = int(os.getenv("N_BATCH", "256"))
        n_gpu_layers = int(os.getenv("N_GPU_LAYERS", "0"))
        use_mlock    = os.getenv("USE_MLOCK", "false").lower() == "true"
        use_mmap     = os.getenv("USE_MMAP",  "true").lower() == "true"

        return LlamaCpp(
            model_path=MODEL_PATH,
            temperature=0.2,   # focused answers
            max_tokens=768,
            top_p=0.95,
            n_ctx=4096,
            n_threads=n_threads,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            use_mlock=use_mlock,
            use_mmap=use_mmap,
            verbose=False,
            callbacks=callbacks or []
        )
    elif LLM_BACKEND == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            streaming=True,  # enable token-by-token streaming
            callbacks=callbacks or []
        )
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")

# -----------------------------
# Build or load FAISS index
# -----------------------------
def build_or_load_index(url: str, docs: List[Document], force_rebuild: bool = False) -> FAISS:
    """
    Build a FAISS vector store for the given docs and URL, or load it if already cached.
    """
    path = _store_path(url)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = splitter.split_documents(docs)

    content_len = sum(len(d.page_content) for d in docs)
    old_meta = _load_meta(url)
    need_rebuild = force_rebuild or not os.path.exists(path) or old_meta.get("length") != content_len

    if need_rebuild:
        db = FAISS.from_documents(splits, embedding)
        db.save_local(path)
        _save_meta(url, {"length": content_len, "chunks": len(splits)})
        return db
    else:
        return FAISS.load_local(path, embeddings=embedding, allow_dangerous_deserialization=True)

def load_index_only(url: str) -> FAISS:
    """Load FAISS index from disk without needing docs."""
    path = _store_path(url)
    if not os.path.exists(path):
        raise FileNotFoundError("Index not found. Click Embed in sidebar first.")
    return FAISS.load_local(path, embeddings=embedding, allow_dangerous_deserialization=True)

# -----------------------------
# Prompt & QA
# -----------------------------
QA_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant. Answer ONLY using the provided context.
If the answer is not in the context, say you don't know.

Question: {question}

Context:
{context}

Answer concisely in 4–6 sentences with accurate details."""
)

def get_answer(
    db: FAISS,
    url: str,
    question: str,
    callbacks: Optional[List[BaseCallbackHandler]] = None
) -> str:
    """
    Retrieve relevant chunks, run the LLM, and return the answer text.
    """
    retriever = db.as_retriever(search_kwargs={"k": 6})
    docs = retriever.get_relevant_documents(question)

    # Keep the context reasonably small for speed
    MAX_PER_DOC = 600
    context = "\n\n".join(d.page_content[:MAX_PER_DOC] for d in docs)[:1800]

    llm = _get_llm(callbacks=callbacks)
    chain = LLMChain(llm=llm, prompt=QA_PROMPT, callbacks=callbacks or [])
    return chain.run({"question": question, "context": context}).strip()

# -----------------------------
# Index cleanup
# -----------------------------
def clear_index(url: str):
    """Delete the FAISS index and metadata for the given URL."""
    for p in (_store_path(url), _meta_path(url)):
        if os.path.exists(p):
            os.remove(p)