"""
Receipt Scanner Pipeline
------------------------
Step 1: OCR  — Tesseract reads the image and outputs raw text (no understanding)
Step 2: AI   — receives raw text, understands it, outputs structured JSON

Two options for Step 2:
  - Claude  (paid, better accuracy, needs API key)
  - Ollama  (free, runs locally, needs Ollama installed + model pulled)
"""

import base64
import json
from pathlib import Path
from typing import Optional

import anthropic
import ollama
import pytesseract
from PIL import Image
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Data models — this is the JSON shape Claude will produce
# ---------------------------------------------------------------------------

class ReceiptItem(BaseModel):
    name: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None
    category: Optional[str] = None  # e.g. "clothing", "food", "electronics"


class Receipt(BaseModel):
    store_name: str | None = None
    store_address: Optional[str] = None
    date: Optional[str] = None          # ISO 8601 e.g. "2026-03-28"
    transaction_number: Optional[str] = None   # e.g. "190238124"
    customer_name: Optional[str] = None        # e.g. "Sa ا****"
    invoice_id: Optional[str] = None           # e.g. "202599001198752870"
    items: list[ReceiptItem]
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    tax_percentage: Optional[float] = None
    total: Optional[float] = None
    currency: Optional[str] = "USD"
    payment_method: Optional[str] = None
    return_window_days: Optional[int] = None   # e.g. 14 for Zara, 30 for H&M
    warranty_info: Optional[str] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Step 1: OCR — extract raw text from receipt image
# ---------------------------------------------------------------------------

def extract_text_from_image(image_path: str) -> str:
    """
    Run Tesseract OCR on a receipt image.
    Returns a raw string — just characters, no understanding of what they mean.

    Requires Tesseract installed on your system:
      Windows: https://github.com/UB-Mannheim/tesseract/wiki
      macOS:   brew install tesseract
      Linux:   sudo apt install tesseract-ocr
    """
    image = Image.open(image_path)

    # Tesseract config: treat as a single block of text (good for receipts)
    custom_config = r"--oem 3 --psm 6"
    raw_text = pytesseract.image_to_string(image, config=custom_config)

    return raw_text.strip()


# ---------------------------------------------------------------------------
# Step 2: Claude — parse raw OCR text into structured JSON
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a universal receipt/invoice/bill parsing assistant.
You receive raw OCR text from any scanned document — it could be:
- A retail receipt (Zara, H&M, grocery store, restaurant,coffee shop)
- A bill payment (electricity, water, telecom, insurance)
- An online order confirmation (Amazon, eBay.Noon)
- A payment aggregator receipt (Apple Pay, Google Pay, PayPal,Fawry,AhlyMomken,Masary,ValU,Raseedy,Tabby, MagicPay, etc.)
- A service invoice (car repair, home services, medical bill)
- Insurance documents (policies, claims)
- A handwritten invoice
- In ANY language (Arabic, English, French, Turkish,Spanish,Italian,German, etc.)

The text may have typos, weird spacing, or garbled characters because OCR is imperfect.
Your job is to understand what the document says and extract data into a clean JSON structure.

Rules:
- Extract every line item you can identify (name, quantity, unit price, total price)
- If a field is unclear or missing, use null — NEVER guess or make up values
- For dates, output ISO 8601 format (YYYY-MM-DD)
- Detect the currency from context (symbols like $, £, ج.م, ر.س or words like EGP, USD, SAR,AED)
- For return_window_days: only set this if explicitly mentioned (e.g. "14 days", "30 day returns")
- For warranty_info: include if mentioned on the document
- For transaction_number: look for "transaction #", "رقم الفاتورة", "رقم مرجعي", "order #","providernumber","transaction #", "رقم العملية", "ref no", "confirmation #", etc.
- For customer_name: look for "name", "إسم", "customer", "العميل", "account", "حساب", etc.
- Keep names in their original language — do NOT translate product or store names
- Be strict with numbers — a misread digit in a price causes real problems
- For bill payments with no individual items, put the main service/payment as a single item"""


def parse_receipt_with_claude(raw_ocr_text: str) -> Receipt:
    """
    Send the raw OCR text to Claude.
    Claude understands the text and maps it to a structured Receipt object.
    """
    client = anthropic.Anthropic()

    response = client.messages.parse(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Parse this receipt OCR text into structured data:\n\n{raw_ocr_text}"
            }
        ],
        output_format=Receipt,
    )

    return response.parsed_output


# ---------------------------------------------------------------------------
# Free alternative: Ollama (local AI, no API key, no cost)
# Requires: https://ollama.com installed + `ollama pull llama3.2` run once
# ---------------------------------------------------------------------------

def parse_receipt_with_ollama(raw_ocr_text: str, model: str = "llama3.2") -> Receipt:
    """
    Same job as parse_receipt_with_claude() but runs 100% locally for free.
    Ollama must be running (it starts automatically after install on Windows).

    First time setup:
      ollama pull llama3.2
    """
    prompt = f"""{SYSTEM_PROMPT}

Parse this receipt OCR text and return ONLY valid JSON matching this exact structure, no explanation:
{{
  "store_name": string,
  "store_address": string or null,
  "date": string (YYYY-MM-DD) or null,
  "transaction_number": string or null,
  "customer_name": string or null,
  "invoice_id": string or null,
  "items": [
    {{
      "name": string,
      "quantity": number,
      "unit_price": number,
      "total_price": number,
      "category": string or null
    }}
  ],
  "subtotal": number or null,
  "tax": number or null,
  "total": number,
  "currency": string,
  "payment_method": string or null,
  "return_window_days": number or null,
  "warranty_info": string or null,
  "notes": string or null
}}

Receipt text:
{raw_ocr_text}"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )

    raw_json = response["message"]["content"]
    data = json.loads(raw_json)
    return Receipt(**data)


# ---------------------------------------------------------------------------
# Vision fallback: send the image directly to Claude (skips OCR entirely)
# Use this when Tesseract gives bad results on low-quality receipts
# ---------------------------------------------------------------------------

def parse_receipt_with_vision(image_path: str) -> Receipt:
    """
    Skip OCR entirely — send the raw image to Claude's vision.
    Claude reads and understands the receipt in one step.

    Use this for:
    - Handwritten receipts
    - Poor lighting / skewed photos
    - When Tesseract output is too garbled
    """
    client = anthropic.Anthropic()

    image_data = Path(image_path).read_bytes()
    b64_image = base64.standard_b64encode(image_data).decode("utf-8")

    # Detect media type from extension
    suffix = Path(image_path).suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")

    response = client.messages.parse(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Parse this receipt image into structured data.",
                    },
                ],
            }
        ],
        output_format=Receipt,
    )

    return response.parsed_output


# ---------------------------------------------------------------------------
# Full pipeline: OCR → Claude → Receipt
# ---------------------------------------------------------------------------

def scan_receipt(image_path: str, use_vision_fallback: bool = False) -> dict:
    """
    Full pipeline:
      1. Run Tesseract OCR on the image (or skip to vision if use_vision_fallback=True)
      2. Send output to Claude to extract structured JSON
      3. Return the result as a dict

    Args:
        image_path: Path to the receipt image (JPG, PNG, etc.)
        use_vision_fallback: If True, skip OCR and send image directly to Claude
    """
    if use_vision_fallback:
        print("Using Claude vision directly (no OCR)...")
        receipt = parse_receipt_with_vision(image_path)
        raw_text = "(image sent directly to Claude vision)"
    else:
        print("Step 1: Running OCR...")
        raw_text = extract_text_from_image(image_path)
        print(f"\n--- Raw OCR Output ---\n{raw_text}\n----------------------\n")

        print("Step 2: Claude parsing OCR text...")
        receipt = parse_receipt_with_claude(raw_text)

    result = {
        "raw_ocr_text": raw_text,
        "parsed_receipt": receipt.model_dump(),
    }

    return result


# ---------------------------------------------------------------------------
# Demo: parse a receipt from raw text (no image needed to test)
# ---------------------------------------------------------------------------

def demo_parse_raw_text(ocr_text: str, backend: str = "ollama") -> Receipt:
    """
    Test the AI parsing step directly with raw text.
    Useful for development without a real receipt image.

    backend: "ollama" (free, local) or "claude" (paid, more accurate)
    """
    if backend == "ollama":
        print("Ollama parsing text (local, free)...")
        return parse_receipt_with_ollama(ocr_text)
    else:
        print("Claude parsing text...")
        return parse_receipt_with_claude(ocr_text)
