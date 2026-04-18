"""
Receipt Scanner — Google Gemini Vision (Recommended)
-----------------------------------------------------
Sends the receipt image DIRECTLY to Gemini — no separate OCR step needed.
Gemini reads the image and returns structured JSON in one API call.

Why this is better than scan_google_vision.py:
  - No OCR errors (Gemini sees the actual image, not garbled text)
  - No Ollama needed (no local model)
  - Handles Arabic + English naturally
  - Much smarter parsing (follows complex rules reliably)
  - One API call instead of two

Setup (one time):
  1. Go to aistudio.google.com → Sign in → "Get API key"
  2. Create API key (free, no billing required for development)
  3. Add to your .env file:
       GEMINI_API_KEY=your_key_here

Free tier limits (Google AI Studio):
  - 1,500 requests/day, 15 requests/minute
  - No credit card required

Production (unlimited):
  - Enable billing on the same Google account
  - Gemini 2.0 Flash costs ~$0.0001 per receipt image
  - 10,000 users scanning 1 receipt each ≈ $1 total

Usage:
  python scan_gemini.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import base64
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import requests
from receipt_scanner import Receipt, ReceiptItem

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG — change IMAGE_PATH to your receipt image
# ---------------------------------------------------------------------------

#IMAGE_PATH = "kaireceipt.jpeg"
# IMAGE_PATH = "zaracecipt.jpeg"
#IMAGE_PATH = "talabatreceipt.jpeg"
# IMAGE_PATH = "receiptmagicpay.png"
# IMAGE_PATH = "starbucksreceipt.jpeg"
#IMAGE_PATH = "seoudireceipt1.jpeg"
#IMAGE_PATH = "seoudireceipt2.jpeg"
#IMAGE_PATH = "pharmacy1.jpeg"
IMAGE_PATH = "Oscar1.jpeg"   # <-- change to your receipt image

GEMINI_MODEL = "google/gemini-2.0-flash-001"   # Gemini 2.0 Flash on OpenRouter

# ---------------------------------------------------------------------------
# Prompt — sent alongside the image. Much simpler than scan_google_vision.py
# because Gemini understands receipts natively without needing workaround rules.
# ---------------------------------------------------------------------------

GEMINI_PROMPT = """You are a receipt parsing assistant. Look at this receipt image and extract all data into JSON.

RULES:
- Return ONLY valid JSON — no explanation, no markdown, no code fences
- Every field you output MUST be visible in the image. Do NOT invent or guess values.
- If a field is not present or unclear, use null
- store_name: use the English name if both Arabic and English exist
- date: output as YYYY-MM-DD format only
- currency: detect from symbols (EGP, LE, $, £, SAR, AED, etc.)
- items: extract only real purchased products. Do NOT include:
    * Delivery fees, service fees, VAT rows, tax rows → put fees in tax_amount instead
    * Barcodes, phone numbers, account numbers, auth codes
    * Payment method lines, loyalty points
- For café/restaurant receipts: add-ons like "Coconut Milk", "Extra Shot" are modifiers — append to the item name, do NOT create a separate item
- For supermarket receipts: item format is usually name + weight/qty + unit price + total. Use total_price per item.
- For food delivery (Talabat, Uber Eats): store_name = restaurant name, customer_name = delivery recipient
- total: use the final total shown on the receipt (labeled TOTAL, Total with Tax, المبلغ الكلي, etc.). NEVER compute it.
- payment_method: extract from payment section (Visa, Cash, Debit Card, etc.)
- return_window_days: only set if explicitly mentioned (e.g. "14 days", "30 day return policy")

Output this exact JSON structure:
{
  "store_name": string or null,
  "store_address": string or null,
  "date": string (YYYY-MM-DD) or null,
  "transaction_number": string or null,
  "customer_name": string or null,
  "invoice_id": string or null,
  "items": [
    {
      "name": string,
      "quantity": number or null,
      "unit_price": number or null,
      "total_price": number or null,
      "category": string or null
    }
  ],
  "subtotal": number or null,
  "tax_amount": number or null,
  "tax_percentage": number or null,
  "total": number or null,
  "currency": string or null,
  "payment_method": string or null,
  "return_window_days": number or null,
  "warranty_info": string or null,
  "notes": string or null
}"""


# ---------------------------------------------------------------------------
# Step 1: Send image to Gemini and get structured JSON back
# ---------------------------------------------------------------------------

def scan_with_gemini(image_path: str) -> Receipt:
    """
    Send receipt image directly to Gemini Vision via OpenRouter.
    Returns a parsed Receipt object.
    No OCR step needed — Gemini reads and understands the image directly.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found in .env file")

    # Read image and detect MIME type
    image_bytes = Path(image_path).read_bytes()
    suffix = Path(image_path).suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}
    mime_type = mime_map.get(suffix, "image/jpeg")
    b64_image = base64.b64encode(image_bytes).decode()

    # OpenRouter uses the OpenAI-compatible API format
    payload = {
        "model": GEMINI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                    },
                    {"type": "text", "text": GEMINI_PROMPT},
                ],
            }
        ],
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")

    raw = response.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown code fences if model wraps the JSON
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    data = json.loads(raw)
    if "items" in data and data["items"]:
        data["items"] = [i for i in data["items"] if i.get("name")]
    return Receipt(**data)


# ---------------------------------------------------------------------------
# Step 2: Python post-processing — sanity fixes Gemini occasionally gets wrong
# ---------------------------------------------------------------------------

def post_process(receipt: Receipt, image_path: str) -> Receipt:
    """
    Light post-processing on top of Gemini's output.
    Gemini is much more accurate than Ollama so these are minimal safety nets.
    """
    # Fix known OCR store name typos (rare with Gemini but kept as safety net)
    OCR_STORE_FIXES = {"ZABA": "ZARA", "ZEBA": "ZARA", "AOIDAS": "ADIDAS"}
    if receipt.store_name:
        receipt.store_name = OCR_STORE_FIXES.get(
            receipt.store_name.strip().upper(), receipt.store_name
        )

    # Fix currency: if Gemini output "LE", normalize to "EGP"
    if receipt.currency and receipt.currency.upper() in ("LE", "L.E", "L.E."):
        receipt.currency = "EGP"

    # Reject item total_prices that look like barcodes (> 50,000)
    for item in receipt.items:
        if item.total_price and item.total_price > 50_000:
            item.total_price = None
        if item.unit_price and item.unit_price > 50_000:
            item.unit_price = None

    # Reject total if it looks like a barcode (large integer with no cents)
    if receipt.total and receipt.total > 9_999 and receipt.total == int(receipt.total):
        receipt.total = None

    # Fill total if still missing
    if not receipt.total:
        if receipt.subtotal and receipt.tax_amount:
            receipt.total = round(receipt.subtotal + receipt.tax_amount, 2)
        elif receipt.items:
            items_sum = sum(
                i.total_price for i in receipt.items
                if i.total_price and i.total_price <= 50_000
            )
            if items_sum > 0:
                receipt.total = round(items_sum, 2)

    return receipt


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"=== PurchaseBuddy — Gemini Vision Scanner: {IMAGE_PATH} ===\n")

    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not found.")
        print("Get a free key at: openrouter.ai → API Keys")
        print("Then add to your .env file:")
        print("  OPENROUTER_API_KEY=your_key_here")
        exit(1)

    # Scan
    print("Sending image to Gemini Vision...")
    receipt = scan_with_gemini(IMAGE_PATH)
    receipt = post_process(receipt, IMAGE_PATH)

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
        print(f"Return by:     {receipt.return_window_days} days from {receipt.date}")

    print("\nItems:")
    for item in receipt.items:
        name = item.name or "N/A"
        qty = f"{item.quantity}x " if item.quantity else ""
        price = f"{receipt.currency} {item.total_price}" if item.total_price else ""
        print(f"  - {name}: {qty}{price}")
