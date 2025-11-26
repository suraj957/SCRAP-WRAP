import os
import sqlite3
import uuid
import datetime
from typing import List, Dict, Optional

# Database file path (can override via .env if needed)
DB_PATH = os.getenv("DB_PATH", "chat.db")

# -----------------------------
# Internal connection helper
# -----------------------------
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # so we can fetch as dict-like rows
    return conn

# -----------------------------
# Initialize database tables
# -----------------------------
def init_db():
    """Create the conversations and messages tables if they don't exist."""
    conn = _connect()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT,
        url TEXT,
        created_at TEXT,
        updated_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT,
        role TEXT,                -- 'user' | 'assistant'
        content TEXT,
        created_at TEXT,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id)
    )
    """)

    conn.commit()
    conn.close()

# -----------------------------
# Conversation operations
# -----------------------------
def new_conversation(url: str, title: Optional[str] = None) -> str:
    """Create a new conversation and return its ID."""
    cid = str(uuid.uuid4())
    now = datetime.datetime.utcnow().isoformat()
    title = title or "New chat"

    conn = _connect()
    conn.execute(
        "INSERT INTO conversations (id, title, url, created_at, updated_at) VALUES (?,?,?,?,?)",
        (cid, title, url, now, now)
    )
    conn.commit()
    conn.close()
    return cid

def list_conversations(limit: int = 50) -> List[Dict]:
    """Return a list of recent conversations."""
    conn = _connect()
    rows = conn.execute("""
        SELECT id, title, url, created_at, updated_at
        FROM conversations
        ORDER BY updated_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_conversation(cid: str) -> Optional[Dict]:
    """Fetch a single conversation by ID."""
    conn = _connect()
    row = conn.execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
    conn.close()
    return dict(row) if row else None

def rename_conversation(cid: str, title: str):
    """Rename a conversation."""
    now = datetime.datetime.utcnow().isoformat()
    conn = _connect()
    conn.execute(
        "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
        (title, now, cid)
    )
    conn.commit()
    conn.close()

def delete_conversation(cid: str):
    """Delete a conversation and all its messages."""
    conn = _connect()
    conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
    conn.execute("DELETE FROM conversations WHERE id=?", (cid,))
    conn.commit()
    conn.close()

# -----------------------------
# Message operations
# -----------------------------
def add_message(cid: str, role: str, content: str):
    """Add a message to a conversation."""
    now = datetime.datetime.utcnow().isoformat()
    conn = _connect()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?,?,?,?)",
        (cid, role, content, now)
    )
    conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, cid))
    conn.commit()
    conn.close()

def get_messages(cid: str) -> List[Dict]:
    """Get all messages for a conversation in order."""
    conn = _connect()
    rows = conn.execute("""
        SELECT role, content, created_at
        FROM messages
        WHERE conversation_id=?
        ORDER BY id ASC
    """, (cid,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
