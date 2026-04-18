"""
Scan a real receipt image (supports Arabic + English)
-----------------------------------------------------
Usage:
  1. Drop your receipt image in this folder (or use full path)
  2. Change IMAGE_PATH below to your file name
  3. Run: python scan_receipt_image.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import json
import pytesseract
from PIL import Image
import ollama
from receipt_scanner import Receipt
from mmain import SYSTEM_PROMPT,SYSTEM_PROMPT_Shorter
import re

# ---------------------------------------------------------------------------
# CONFIG — change these
# ---------------------------------------------------------------------------

IMAGE_PATH = "kaireceipt.jpeg"   # <-- put your receipt image name here

# Tesseract location (installed on D:)
pytesseract.pytesseract.tesseract_cmd = r"D:\Programs\Tesseract\tesseract.exe"


# ---------------------------------------------------------------------------
# Step 1: OCR — read text from receipt image (Arabic + English) -- Tesseract OCR
# ---------------------------------------------------------------------------

def extract_text(image_path: str) -> str:
    """Run Tesseract OCR with Arabic + English support."""
    image = Image.open(image_path)

    # ara+eng = read both Arabic and English text
    # --psm 6 = assume a single block of text (good for receipts)
    raw_text = pytesseract.image_to_string(image, lang="ara+eng", config="--oem 3 --psm 6")

    return raw_text.strip()

import re

def clean_ocr_text(text: str) -> str:
    # Fix common OCR number issues
    text = text.replace(",", "")  # remove thousands commas
    
    # Fix patterns like 1.699.001 → 1699.00
    text = re.sub(r"(\d)\.(\d{3})\.(\d{3})", r"\1\2.\3", text)

    # Remove weird symbols
    text = re.sub(r"[^\w\s\.\-\n\(\)]", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ---------------------------------------------------------------------------
# Step 2: Ollama — parse raw OCR text into structured JSON (free, local) -- LLM Language Model
# ---------------------------------------------------------------------------

def parse_with_ollama(raw_text: str) -> Receipt:
    """Send OCR text to local Ollama for structured parsing."""
    #prompt = f"""{SYSTEM_PROMPT_Shorter}
    prompt = f"""{SYSTEM_PROMPT}

IMPORTANT: The receipt may be in Arabic. Keep original Arabic names as-is in the JSON output.
Provide an English translation in the "notes" field. Extract all prices, quantities, dates, and store info accurately.
For fields like transaction_number and invoice_id, look for Arabic labels like رقم العملية, رقم مرجعي, رقم الفاتورة.

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
  "subtotal": float | None = None,
  "tax_amount": float | None = None,
  "tax_percentage": float | None = None,
  "total": float,
  "currency": string,
  "payment_method": string or null,
  "return_window_days": number or null,
  "warranty_info": string or null,
  "notes": string or null
}}

Receipt text:
{raw_text}"""

    response = ollama.chat(
        model="llama3.2",
        #model="llama3.1:8b",   # or even 3b if available
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )

    raw_json = response["message"]["content"]
    data = json.loads(raw_json)
    return Receipt(**data)


def validate_total(receipt: Receipt):
    subtotal = receipt.subtotal or 0
    tax = receipt.tax_amount or 0

    expected_total = subtotal + tax

    # If total is missing or wrong → fix it
    if not receipt.total or abs(receipt.total - expected_total) > 5:
        receipt.total = expected_total

    return receipt

def normalize_tax(receipt: Receipt):
    subtotal = receipt.subtotal or 0

    # Case 1: Only percentage exists → calculate amount
    if receipt.tax_percentage and not receipt.tax_amount and subtotal:
        receipt.tax_amount = subtotal * (receipt.tax_percentage / 100)

    # Case 2: Only amount exists → calculate percentage
    elif receipt.tax_amount and not receipt.tax_percentage and subtotal:
        receipt.tax_percentage = (receipt.tax_amount / subtotal) * 100

    return receipt

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"=== PurchaseBuddy — Scanning: {IMAGE_PATH} ===\n")

    # Step 1: OCR
    print("Step 1: Running OCR (Arabic + English)...")
    raw_text = extract_text(IMAGE_PATH)
    raw_text = clean_ocr_text(raw_text)
    
    print(f"\n--- Raw OCR Output ---\n{raw_text}\n----------------------\n")

    # Step 2: AI parsing
    print("Step 2: Ollama parsing into JSON...")
    receipt = parse_with_ollama(raw_text)
    
    receipt = validate_total(receipt)
    receipt = normalize_tax(receipt)

    # Output
    print("\n--- Parsed Receipt ---")
    print(json.dumps(receipt.model_dump(), indent=2, ensure_ascii=False))

    print("\n--- Summary ---")
    print(f"Store:         {receipt.store_name}")
    print(f"Date:          {receipt.date}")
    print(f"Total:         {receipt.currency} {receipt.total}")
    print(f"Items:         {len(receipt.items)}")
    if receipt.transaction_number:
        print(f"Transaction #: {receipt.transaction_number}")
    if receipt.customer_name:
        print(f"Customer:      {receipt.customer_name}")
    if receipt.invoice_id:
        print(f"Invoice ID:    {receipt.invoice_id}")
    if receipt.return_window_days:
        print(f"Return deadline: {receipt.return_window_days} days from {receipt.date}")

    print("\nItems:")
    for item in receipt.items:
        name = item.name or "N/A"
        qty = f"{item.quantity}x " if item.quantity else ""
        price = f"{receipt.currency} {item.total_price}" if item.total_price else ""
        print(f"  - {name}: {qty}{price}")


    