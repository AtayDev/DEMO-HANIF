import os, hashlib, json, time, requests, shelve
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import pdfplumber
import docx
import tiktoken
import chromadb
from chromadb.config import Settings

# =========================
# Config & paths
# =========================
load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./storage/chroma")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./storage/uploads")
FLASK_SECRET = os.getenv("FLASK_SECRET", "dev-secret")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")

# Context / generation budgets
CONTEXT_MAX_TOKENS = int(os.getenv("CONTEXT_MAX_TOKENS", "1200"))
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "256"))

# Parallel embedding workers
EMBED_THREADS = int(os.getenv("EMBED_THREADS", "4"))

Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".pdf", ".docx", ".txt"}

# Manifest (avoid re-indexing identical files)
MANIFEST_PATH = Path(CHROMA_DIR) / "manifest.json"
if not MANIFEST_PATH.exists():
    MANIFEST_PATH.write_text("{}", encoding="utf-8")

# On-disk embedding cache
CACHE_PATH = str(Path(CHROMA_DIR) / "emb_cache.db")

# Flask app
app = Flask(__name__)
app.secret_key = FLASK_SECRET

# Persistent Chroma client
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# Tokenizer
_enc = tiktoken.get_encoding("cl100k_base")

# =========================
# Small utils
# =========================
def load_manifest() -> Dict[str, Dict]:
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_manifest(m: Dict[str, Dict]) -> None:
    MANIFEST_PATH.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS

def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# =========================
# File readers & chunking
# =========================
def read_pdf(path: Path) -> Tuple[str, List[str]]:
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            pages.append(t)
    return "\n".join(pages), pages

def read_docx(path: Path) -> str:
    d = docx.Document(str(path))
    paras = [p.text for p in d.paragraphs]
    return "\n".join([p for p in paras if p.strip()])

def read_txt(path: Path) -> str:
    return path.read_text(errors="ignore")

def chunk_tokenwise(text: str, max_tokens=900, overlap=150) -> List[str]:
    toks = _enc.encode(text)
    chunks, start, n = [], 0, len(toks)
    while start < n:
        end = min(n, start + max_tokens)
        chunk = _enc.decode(toks[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks

def map_pages_to_chunks(per_page_texts: List[str], max_tokens=900, overlap=150):
    if not per_page_texts:
        return [], []
    chunks, spans = [], []
    for page_idx, page_text in enumerate(per_page_texts, start=1):
        page_chunks = chunk_tokenwise(page_text, max_tokens=max_tokens, overlap=overlap)
        for ch in page_chunks:
            chunks.append(ch)
            spans.append((page_idx, page_idx))
    return chunks, spans

def trim_to_token_limit(texts: List[str], max_total_tokens: int) -> List[str]:
    kept, used = [], 0
    for t in texts:
        nt = len(_enc.encode(t))
        if used + nt > max_total_tokens:
            if not kept and nt > max_total_tokens:
                # take a head if the first block is too big
                head = _enc.decode(_enc.encode(t)[:max_total_tokens])
                kept.append(head)
            break
        kept.append(t)
        used += nt
    return kept

# =========================
# Ollama (embeddings & chat)
# =========================
def _embed_one(text: str) -> List[float]:
    # Retry on transient failures
    for attempt in range(3):
        try:
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
                timeout=120
            )
            if r.status_code != 200:
                raise RuntimeError(r.text)
            return r.json()["embedding"]
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError("Embedding failed after retries")

def embed_ollama(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    out = [None] * len(texts)
    with shelve.open(CACHE_PATH) as db:
        # gather non-cached jobs
        jobs = []
        with ThreadPoolExecutor(max_workers=EMBED_THREADS) as ex:
            for i, t in enumerate(texts):
                key = hashlib.sha256((OLLAMA_EMBEDDING_MODEL + "||" + t).encode("utf-8")).hexdigest()
                if key in db:
                    out[i] = db[key]
                else:
                    jobs.append((i, key, ex.submit(_embed_one, t)))
            for i, key, fut in jobs:
                v = fut.result()
                db[key] = v
                out[i] = v
    return out  # type: ignore

def chat_ollama(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "num_predict": ANSWER_MAX_TOKENS,
            "temperature": 0.1,
            "top_k": 30,
            "num_ctx": 2048,
            "repeat_penalty": 1.05
        }
    }
    for attempt in range(3):
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=180)
        if r.status_code == 200:
            return r.json().get("message", {}).get("content", "").strip()
        if attempt == 2:
            raise RuntimeError(f"Ollama chat error: {r.text}")
        time.sleep(0.8 * (attempt + 1))
    return ""

# =========================
# Chroma helpers
# =========================
def collection_for(client_id: str):
    name = f"client__{client_id}"
    try:
        return chroma_client.get_collection(name=name)
    except Exception:
        return chroma_client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

def upsert_chunks(client_id: str, sha: str, filename: str,
                  chunks: List[str], spans: List[Tuple[Optional[int], Optional[int]]]):
    col = collection_for(client_id)
    ids = [f"{sha}::chunk::{i}" for i in range(len(chunks))]
    metas = []
    for i, (ps, pe) in enumerate(spans):
        metas.append({
            "filename": filename,
            "sha": sha,
            "chunk_index": i,
            "page_start": ps,
            "page_end": pe
        })
    # explicit embeddings (parallelized + cached)
    embs = embed_ollama(chunks)
    col.upsert(ids=ids, documents=chunks, embeddings=embs, metadatas=metas)

def build_context_blocks(hits: Dict) -> List[str]:
    docs = hits.get("documents") or []
    metas = hits.get("metadatas") or []

    if docs and isinstance(docs[0], list):
        docs = docs[0]
    if metas and isinstance(metas[0], list):
        metas = metas[0]

    blocks = []
    for i in range(min(len(docs), len(metas))):
        doc  = (docs[i] or "").strip()
        meta = metas[i] or {}
        if not doc:
            continue
        fn = meta.get("filename", "unknown")
        ps = meta.get("page_start")
        pe = meta.get("page_end")
        pg = (f"pages {ps}" if ps == pe else f"pages {ps}-{pe}") if (ps and pe) else "pages n/a"
        idx = meta.get("chunk_index", "n/a")
        blocks.append(f"{doc}\n[Source: {fn}, {pg}, chunk {idx}]")
    return blocks

def format_prompt(question: str, context_blocks: List[str]) -> Tuple[str, str]:
    trimmed = trim_to_token_limit(context_blocks, CONTEXT_MAX_TOKENS)
    context_text = "\n\n---\n\n".join(trimmed)
    system = (
        "You are a careful assistant. Answer ONLY with facts from the provided Context.\n"
        "If the Context is insufficient, say you don't know.\n"
        "Be concise. Include inline citations like [Source: filename, pages, chunk N] when relevant."
    )
    user = f"Question:\n{question}\n\nContext:\n{context_text}\n\nAnswer:"
    return system, user

# =========================
# RAG core (EXHAUSTIVE SEARCH BY DEFAULT)
# =========================
def rag_answer(client_id: str, question: str) -> Dict:
    col = collection_for(client_id)

    # 1) embed question
    qvec = embed_ollama([question])[0]

    # 2) EXHAUSTIVE: ask Chroma for ALL chunks in this client's collection
    try:
        total = col.count()  # number of vectors (chunks)
    except Exception:
        total = 50  # fallback if count unavailable
    n_results = max(1, total)

    hits = col.query(
        query_embeddings=[qvec],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    print("--------HITS------>", hits)

    if not hits.get("documents") or not hits["documents"][0]:
        return {"answer": "I couldn't find anything in the uploaded documents.", "contexts": []}

    # 3) light rerank by distance (cosine -> lower is better)
    docs  = hits["documents"][0]
    metas = hits["metadatas"][0]
    dists = hits.get("distances", [[0] * len(docs)])[0]
    order = sorted(range(len(docs)), key=lambda i: dists[i])
    docs  = [docs[i] for i in order]
    metas = [metas[i] for i in order]
    re_hits = {"documents": [docs], "metadatas": [metas]}

    # 4) build prompt (with citations) + trim to context budget
    blocks = build_context_blocks(re_hits)
    system, user = format_prompt(question, blocks)

    # 5) generate
    answer = chat_ollama(system, user)
    return {"answer": answer, "contexts": blocks[: min(5, len(blocks))]}

# =========================
# Routes
# =========================
@app.get("/")
def index():
    client = request.args.get("client", "default")
    files = [p.name for p in Path(UPLOAD_DIR).glob("*") if p.is_file()]
    return render_template("index.html", client=client, files=files)

@app.post("/upload")
def upload():
    client = request.form.get("client", "default")
    f = request.files.get("file")
    if not f or f.filename == "":
        flash("No file selected", "error"); return redirect(url_for("index", client=client))
    if not allowed_file(f.filename):
        flash("Unsupported file type. Use PDF, DOCX, or TXT.", "error"); return redirect(url_for("index", client=client))

    filename = secure_filename(f.filename)
    dest = Path(UPLOAD_DIR) / filename
    f.save(dest)
    sha = file_sha256(dest)

    # Skip if already indexed with same embedding model
    manifest = load_manifest()
    already = manifest.get(sha)
    if already and already.get("filename") == filename and already.get("embed_model") == OLLAMA_EMBEDDING_MODEL:
        flash(f"Skipped: {filename} already indexed (sha: {sha[:8]}…)", "ok")
        return redirect(url_for("index", client=client))

    try:
        suffix = dest.suffix.lower()
        if suffix == ".pdf":
            _, per_page = read_pdf(dest)
            chunks, spans = map_pages_to_chunks(per_page, 900, 150)
            if not chunks:  # fallback if PDF text empty
                text = dest.read_text(errors="ignore")
                chunks = chunk_tokenwise(text, 900, 150)
                spans = [(None, None)] * len(chunks)
        elif suffix == ".docx":
            text = read_docx(dest)
            chunks = chunk_tokenwise(text, 900, 150)
            spans = [(None, None)] * len(chunks)
        else:  # .txt
            text = read_txt(dest)
            chunks = chunk_tokenwise(text, 900, 150)
            spans = [(None, None)] * len(chunks)

        if not chunks:
            flash("The document appears empty after parsing.", "error")
            return redirect(url_for("index", client=client))

        upsert_chunks(client, sha, filename, chunks, spans)

        manifest[sha] = {
            "filename": filename,
            "embed_model": OLLAMA_EMBEDDING_MODEL,
            "chunks": len(chunks),
            "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_manifest(manifest)

        flash(f"Uploaded & indexed: {filename} ({len(chunks)} chunks)", "ok")
    except Exception as e:
        flash(f"Failed to ingest: {e}", "error")
    return redirect(url_for("index", client=client))

@app.post("/api/chat")
def api_chat():
    client = request.args.get("client", "default")
    data = request.get_json(silent=True) or {}
    q = (data.get("q") or "").strip()
    if not q:
        return jsonify({"answer": "Please provide a question."})
    try:
        result = rag_answer(client, q)  # exhaustive by default
        return jsonify({"answer": result["answer"], "contexts": result.get("contexts", [])})
    except Exception as e:
        return jsonify({"answer": f"Error: {e}"}), 500

if __name__ == "__main__":
    # Run: python app.py  → http://localhost:8000
    app.run(host="0.0.0.0", port=8000, debug=True)
