"""
PurchaseBuddy Receipt Scanner API
----------------------------------
POST /scan   — upload a receipt image, get structured JSON back

Query param:
  model=gemini   (default) — Gemini 2.0 Flash via OpenRouter (paid, most accurate)
  model=groq     — Groq Llama 4 Scout (free, fast)
  model=nvidia   — NVIDIA Nemotron 12B (free, OCR-specialized)
  model=gemma    — Google Gemma 4 31B (free, larger/smarter but slower)

Usage:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import io
import tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

# Import scan functions from existing scanner files
from scan_gemini import scan_with_gemini, post_process as gemini_post_process
from scan_groq import scan_with_groq, post_process as groq_post_process
from scan_nvidia import scan_with_nvidia, post_process as nvidia_post_process
from scan_gemma import scan_with_gemma, post_process as gemma_post_process

app = FastAPI(
    title="PurchaseBuddy Receipt Scanner",
    description="Upload a receipt image and get structured JSON back.",
    version="1.0.0",
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

SCANNERS = {
    "gemini": (scan_with_gemini, gemini_post_process),
    "groq":   (scan_with_groq,   groq_post_process),
    "nvidia": (scan_with_nvidia, nvidia_post_process),
    "gemma":  (scan_with_gemma,  gemma_post_process),
}


@app.get("/")
def root():
    return {"status": "ok", "message": "PurchaseBuddy Receipt Scanner is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scan")
async def scan_receipt(
    file: UploadFile = File(...),
    model: str = Query(default="gemini", enum=["gemini", "groq", "nvidia", "gemma"]),
):
    """
    Upload a receipt image and get structured JSON back.

    - **file**: Receipt image (JPG, PNG, WEBP)
    - **model**: Which AI model to use (gemini, groq, nvidia, gemma)
    """
    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Check API key availability
    if model == "groq" and not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    if model in ("gemini", "nvidia", "gemma") and not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    # Save uploaded file to a temp file
    image_bytes = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        scan_fn, post_fn = SCANNERS[model]

        if model == "gemini":
            receipt = scan_fn(tmp_path)
            receipt = post_fn(receipt, tmp_path)
        else:
            receipt = scan_fn(tmp_path)
            receipt = post_fn(receipt)

        return JSONResponse(content=receipt.model_dump())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        Path(tmp_path).unlink(missing_ok=True)
