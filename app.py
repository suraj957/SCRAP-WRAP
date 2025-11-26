import os
import streamlit as st
from dotenv import load_dotenv
from scraper import scrape_url
from retriever import build_or_load_index, get_answer, clear_index
from storage import (
    init_db, list_conversations, new_conversation, get_conversation,
    get_messages, add_message, rename_conversation, delete_conversation
)

# --- Streaming handler (robust across LangChain versions) ---
try:
    from langchain.callbacks.base import BaseCallbackHandler
except Exception:
    from langchain_core.callbacks import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        self._buf = []
        self._ticks = 0

    # LLMs (e.g., LlamaCpp) and many chat models call this
    def on_llm_new_token(self, token: str, **kwargs):
        self._buf.append(token); self._ticks += 1
        if self._ticks % 5 == 0:
            self.text += "".join(self._buf)
            self._buf = []
            self.container.markdown(self.text)

    # Some newer ChatModel streams emit chunks here instead
    def on_chat_model_stream(self, chunk, **kwargs):
        try:
            # chunk is usually an AIMessageChunk; accumulate its .content
            token = getattr(chunk, "content", "") or ""
        except Exception:
            token = ""
        if token:
            self.on_llm_new_token(token, **kwargs)

    def flush(self):
        if self._buf:
            self.text += "".join(self._buf)
            self._buf = []
            self.container.markdown(self.text)

# -------------------

load_dotenv()
init_db()

st.set_page_config(page_title="Scrap • Wrap", page_icon="🧵", layout="wide")
st.title("Scrap • Wrap — URL Q&A (Local)")

# Session state
if "current_cid" not in st.session_state:
    st.session_state.current_cid = None
if "url" not in st.session_state:
    st.session_state.url = ""

# Sidebar: chat management + indexing
with st.sidebar:
    st.markdown("### Chats")
    chats = list_conversations()
    chat_labels = [f"{c['title']} — {c['url']}" if c['url'] else c['title'] for c in chats]
    selected = st.selectbox("Select a chat", options=["(New chat)"] + chat_labels, index=0)

    if selected == "(New chat)":
        url = st.text_input("Enter URL", value="", placeholder="https://example.com/article")
        title = st.text_input("Chat title (optional)", value="")
        if st.button("Create chat", use_container_width=True):
            cid = new_conversation(url=url, title=title.strip() or "New chat")
            st.session_state.current_cid = cid
            st.session_state.url = url
            st.rerun()
    else:
        idx = chat_labels.index(selected)
        chat = chats[idx]
        st.session_state.current_cid = chat["id"]
        st.session_state.url = chat["url"]

        # Rename & delete
        col_ren, col_del = st.columns([3,1])
        with col_ren:
            new_title = st.text_input("Rename", value=chat["title"], key="rename")
        with col_del:
            if st.button("🗑", type="secondary"):
                delete_conversation(chat["id"])
                st.session_state.current_cid = None
                st.rerun()
        if st.button("Save title"):
            rename_conversation(chat["id"], new_title.strip() or "Untitled")
            st.rerun()

    st.divider()
    st.markdown("### Index")
    url = st.text_input("URL for this chat", value=st.session_state.url, placeholder="https://example.com/article", key="active_url")
    colA, colB = st.columns(2)
    with colA:
        ingest_clicked = st.button("Embed", use_container_width=True)
    with colB:
        clear_clicked = st.button("Clear", type="secondary", use_container_width=True)

    if clear_clicked and url:
        clear_index(url)
        st.success("Cleared index for this URL.")

    if ingest_clicked:
        if not url:
            st.error("Please paste a URL.")
        else:
            with st.spinner("Scraping & indexing…"):
                docs = scrape_url(url)
                if not docs or not docs[0].page_content.strip():
                    st.error("No extractable text from that URL.")
                else:
                    build_or_load_index(url, docs, force_rebuild=True)
                    st.success("Ingested & indexed!")
            st.session_state.url = url

# Main chat area
if st.session_state.current_cid:
    msgs = get_messages(st.session_state.current_cid)
    for m in msgs:
        with st.chat_message(m["role"]):
            st.write(m["content"])
else:
    st.info("Create or select a chat from the sidebar to begin.")

prompt = st.chat_input(
    "Type your question and press Enter…",
    disabled=(not st.session_state.current_cid or not st.session_state.url)
)

if prompt and st.session_state.current_cid:
    url = st.session_state.url
    add_message(st.session_state.current_cid, "user", prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        final_text = ""
        with st.spinner("Thinking…"):
            # ⚡ do NOT scrape here; assume you've ingested already
            try:
                from retriever import load_index_only
                db = load_index_only(st.session_state.url)  # fast load
            except Exception:
                # fallback: load-or-build once (will be fast after first ingest)
                db = build_or_load_index(st.session_state.url, scrape_url(st.session_state.url))
            handler = StreamHandler(placeholder)
            try:
                answer = get_answer(db, st.session_state.url, prompt, callbacks=[handler])
            except TypeError:
                answer = get_answer(db, st.session_state.url, prompt)
            handler.flush()
            final_text = handler.text.strip() or answer.strip() or "_(No text returned.)_"
            placeholder.markdown(final_text)
        add_message(st.session_state.current_cid, "assistant", final_text)
    st.rerun()

