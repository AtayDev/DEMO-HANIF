import os, hashlib, time
from pathlib import Path
from typing import List, Tuple, Dict

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import pdfplumber
import docx
import tiktoken

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from datetime import datetime


def tprint(*args, **kwargs):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ", *args, **kwargs)


# =========================
# Config
# =========================
load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./storage/chroma")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./storage/uploads")
FLASK_SECRET = os.getenv("FLASK_SECRET", "dev-secret")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "global_knowledge")

LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "gemini").lower()

# Provider keys and models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_EMBED_MODEL = os.getenv("DEEPSEEK_EMBED_MODEL", "deepseek-embedding")
DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")

# Chunking / RAG settings
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))
FETCH_K = int(os.getenv("FETCH_K", "20"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.4"))
CONTEXT_MAX_TOKENS = int(os.getenv("CONTEXT_MAX_TOKENS", "1200"))
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "256"))

Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
ALLOWED_EXTS = {".pdf", ".docx", ".txt"}

app = Flask(__name__)
app.secret_key = FLASK_SECRET
_enc = tiktoken.get_encoding("cl100k_base")


# =========================
# Embedding + Chat factories
# =========================
def make_embeddings():
    if LLM_PROVIDER == "openai":
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=OPENAI_EMBED_MODEL)
    elif LLM_PROVIDER == "gemini":
        model = GEMINI_EMBED_MODEL if GEMINI_EMBED_MODEL.startswith("models/") else f"models/{GEMINI_EMBED_MODEL}"
        return GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model=model)
    elif LLM_PROVIDER == "deepseek":
        return OpenAIEmbeddings(api_key=DEEPSEEK_API_KEY, model=DEEPSEEK_EMBED_MODEL, base_url=f"{DEEPSEEK_BASE_URL}/v1")
    raise RuntimeError(f"Unsupported provider {LLM_PROVIDER}")


def make_chat():
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_CHAT_MODEL, temperature=0.1)
    elif LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model=GEMINI_CHAT_MODEL, temperature=0.1)
    elif LLM_PROVIDER == "deepseek":
        return ChatOpenAI(api_key=DEEPSEEK_API_KEY, base_url=f"{DEEPSEEK_BASE_URL}/v1",
                          model=DEEPSEEK_CHAT_MODEL, temperature=0.1)
    raise RuntimeError(f"Unsupported provider {LLM_PROVIDER}")


# Lazy load
_EMB, _LLM = None, None
def EMB():
    global _EMB
    if _EMB is None: _EMB = make_embeddings()
    return _EMB

def LLM():
    global _LLM
    if _LLM is None: _LLM = make_chat()
    return _LLM


# =========================
# File utilities
# =========================
def allowed_file(fn: str) -> bool:
    return Path(fn).suffix.lower() in ALLOWED_EXTS


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pdf(path: Path) -> Tuple[str, List[str]]:
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            pages.append(t)
    return "\n".join(pages), pages


def read_docx(path: Path) -> str:
    d = docx.Document(str(path))
    return "\n".join([p.text for p in d.paragraphs if p.text.strip()])


def read_txt(path: Path) -> str:
    return path.read_text(errors="ignore")


def split_tokens(text: str, max_tokens=600, overlap=120) -> List[str]:
    toks = _enc.encode(text)
    chunks, i, n = [], 0, len(toks)
    while i < n:
        j = min(n, i + max_tokens)
        seg = _enc.decode(toks[i:j]).strip()
        if seg:
            chunks.append(seg)
        i = max(0, j - overlap)
        if j >= n: break
    return chunks


# =========================
# Vectorstore (global)
# =========================
def vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=EMB(),
        persist_directory=CHROMA_DIR
    )


def add_documents(vs: Chroma, docs: List[Document], batch_size: int = 64):
    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        ids = [f"{d.metadata['sha']}::{int(time.time()*1e6)}::{i}" for i, d in enumerate(batch)]
        vs.add_documents(batch, ids=ids)
    vs.persist()


# =========================
# Chunking per file type
# =========================
def chunk_file(path: Path, filename: str, sha: str) -> List[Document]:
    suffix = path.suffix.lower()
    docs = []
    if suffix == ".pdf":
        _, pages = read_pdf(path)
        for pageno, text in enumerate(pages, 1):
            for i, chunk in enumerate(split_tokens(text, CHUNK_TOKENS, CHUNK_OVERLAP)):
                docs.append(Document(page_content=chunk, metadata={"filename": filename, "sha": sha,
                                                                  "page_start": pageno, "page_end": pageno,
                                                                  "chunk_index": i}))
    elif suffix == ".docx":
        text = read_docx(path)
        for i, chunk in enumerate(split_tokens(text, CHUNK_TOKENS, CHUNK_OVERLAP)):
            docs.append(Document(page_content=chunk, metadata={"filename": filename, "sha": sha,
                                                              "page_start": -1, "page_end": -1,
                                                              "chunk_index": i}))
    else:
        text = read_txt(path)
        for i, chunk in enumerate(split_tokens(text, CHUNK_TOKENS, CHUNK_OVERLAP)):
            docs.append(Document(page_content=chunk, metadata={"filename": filename, "sha": sha,
                                                              "page_start": -1, "page_end": -1,
                                                              "chunk_index": i}))
    return docs


# =========================
# Prompt / RAG
# =========================
PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a careful assistant. Only answer using facts from the provided context."),
    ("human", "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:")
])


def trim_to_token_limit(texts: List[str], max_total_tokens: int):
    kept, used = [], 0
    for t in texts:
        nt = len(_enc.encode(t))
        if used + nt > max_total_tokens:
            break
        kept.append(t)
        used += nt
    return kept


def blocks_from_docs(docs: List[Document]) -> List[str]:
    blocks = []
    for doc in docs:
        md = doc.metadata
        fn = md.get("filename", "?")
        pg = md.get("page_start", -1)
        blk = f"{doc.page_content}\n[Source: {fn}, page {pg}]"
        blocks.append(blk)
    return blocks


def rag_answer(question: str) -> Dict:
    vs = vectorstore()
    retriever = vs.as_retriever(search_type="mmr",
                                search_kwargs={"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": MMR_LAMBDA})
    docs = retriever.invoke(question)
    if not docs:
        return {"answer": "No relevant information found.", "contexts": []}

    blocks = blocks_from_docs(docs)
    context_text = "\n\n---\n\n".join(trim_to_token_limit(blocks, CONTEXT_MAX_TOKENS))
    chain = PROMPT | LLM() | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context_text}).strip()
    return {"answer": answer, "contexts": blocks[:min(5, len(blocks))]}


# =========================
# Routes
# =========================
@app.get("/")
def index():
    files = [p.name for p in Path(UPLOAD_DIR).glob("*")]
    return render_template("index.html", files=files)


@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f or f.filename == "":
        flash("No file selected", "error")
        return redirect(url_for("index"))
    if not allowed_file(f.filename):
        flash("Invalid file type", "error")
        return redirect(url_for("index"))

    filename = secure_filename(f.filename)
    dest = Path(UPLOAD_DIR) / filename
    f.save(dest)
    sha = file_sha256(dest)
    tprint(f"Processing file {filename} (sha={sha[:8]}…)")

    try:
        docs = chunk_file(dest, filename, sha)
        vs = vectorstore()
        add_documents(vs, docs)
        flash(f"Uploaded & indexed: {filename} ({len(docs)} chunks)", "ok")
    except Exception as e:
        flash(f"Error indexing file: {e}", "error")

    return redirect(url_for("index"))


@app.post("/api/chat")
def api_chat():
    q = (request.get_json(silent=True) or {}).get("q", "").strip()
    if not q:
        return jsonify({"answer": "Please provide a question."})
    try:
        result = rag_answer(q)
        return jsonify(result)
    except Exception as e:
        return jsonify({"answer": f"Error: {e}", "contexts": []}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
