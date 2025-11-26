# scraper.py
import io
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from langchain_core.documents import Document

def _clean_text(s: str) -> str:
    """Remove excessive whitespace and join lines cleanly."""
    return " ".join(s.split())

def _requests_get(url: str, timeout=25):
    """Fetch a URL with a browser-like User-Agent."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ScrapWrapBot/1.0)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r

# -----------------------------
# PDF scraping
# -----------------------------
def _scrape_pdf(url: str):
    """Extract text from a PDF URL."""
    try:
        import pypdf
        raw = _requests_get(url).content
        reader = pypdf.PdfReader(io.BytesIO(raw))
        text = " ".join([(p.extract_text() or "") for p in reader.pages])
        return _clean_text(text) if text else None
    except Exception:
        return None

# -----------------------------
# HTML scraping with Trafilatura
# -----------------------------
def _scrape_trafilatura(url: str):
    """Use Trafilatura for boilerplate removal."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(
                downloaded,
                include_tables=False,
                include_comments=False
            )
            if extracted:
                return _clean_text(extracted)
    except Exception:
        pass
    return None

# -----------------------------
# HTML scraping with BeautifulSoup
# -----------------------------
def _scrape_bs4(url: str):
    """Fallback: basic HTML parsing with BeautifulSoup."""
    r = _requests_get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    # Prefer <main> or <article>, else gather all <p>
    candidates = []
    for sel in ["main", "article"]:
        node = soup.select_one(sel)
        if node:
            candidates.append(" ".join(p.get_text(" ", strip=True) for p in node.find_all("p")))

    if not candidates:
        candidates.append(" ".join(p.get_text(" ", strip=True) for p in soup.find_all("p")))

    text = max(candidates, key=len) if candidates else ""
    return _clean_text(text)

# -----------------------------
# Public entrypoint
# -----------------------------
def scrape_url(url: str):
    """
    Given a URL, return a list with a single LangChain Document.
    Automatically detects PDF vs HTML and picks the right scraper.
    """
    parsed = urlparse(url)

    # PDF detection
    if parsed.path.lower().endswith(".pdf"):
        txt = _scrape_pdf(url)
        if txt:
            return [Document(page_content=txt, metadata={"source": url})]

    # Try Trafilatura first (best for news/articles)
    txt = _scrape_trafilatura(url)
    if not txt:
        # Fallback to BeautifulSoup
        txt = _scrape_bs4(url)

    return [Document(page_content=txt or "", metadata={"source": url})]
