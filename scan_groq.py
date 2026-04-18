"""
Receipt Scanner — Groq Vision (Free & Fast)
--------------------------------------------
Uses Groq's free API with Llama 3.2 Vision to scan receipts.
Sends the image directly — no separate OCR step needed.

Why Groq:
  - 100% free, no credit card required
  - 14,400 requests/day (very generous for personal use)
  - Extremely fast inference (faster than OpenRouter)
  - Stable — models don't disappear like OpenRouter free tier
  - Handles Arabic + English receipts

Setup (one time):
  1. Go to console.groq.com → Sign up (free)
  2. Go to API Keys → Create API Key
  3. Add to your .env file:
       GROQ_API_KEY=your_key_here

Usage:
  python scan_groq.py
"""

import sys
import io
import base64
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import requests
from receipt_scanner import Receipt

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG — change IMAGE_PATH to your receipt image
# ---------------------------------------------------------------------------

IMAGE_PATH = "kaireceipt.jpeg"
# IMAGE_PATH = "zaracecipt.jpeg"
# IMAGE_PATH = "talabatreceipt.jpeg"
# IMAGE_PATH = "receiptmagicpay.png"
# IMAGE_PATH = "starbucksreceipt.jpeg"
# IMAGE_PATH = "seoudireceipt1.jpeg"

# Groq free vision models (both free, 90B is smarter but slower)
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # fast + smart, free on Groq

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT = """You are a receipt parsing assistant. Look at this receipt image and extract all data into JSON.

RULES:
- Return ONLY valid JSON — no explanation, no markdown, no code fences
- Every field you output MUST be visible in the image. Do NOT invent or guess values.
- If a field is not present or unclear, use null
- store_name: use the English name if both Arabic and English exist
- date: output as YYYY-MM-DD format only
- currency: detect from symbols or words (EGP, LE, $, £, SAR, AED, etc.)
- total: use the final total labeled TOTAL, Total with Tax, المبلغ الكلي, etc. NEVER compute it.
- items: extract only real purchased products. Do NOT include:
    * Delivery fees, service fees, VAT rows → put in tax_amount instead
    * Barcodes, phone numbers, account numbers, auth codes
- For café receipts: add-ons (Coconut Milk, Extra Shot) are modifiers — append to item name, not a separate item
- For supermarket receipts: use total_price per item (weight × unit price)
- For food delivery (Talabat, Uber Eats): store_name = restaurant, customer_name = delivery recipient
- payment_method: Visa, Cash, Debit Card, etc.
- return_window_days: only if explicitly mentioned (e.g. "14 days return")
- CRITICAL: Do NOT hallucinate. Only use what you actually see in the image.

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
# Scan
# ---------------------------------------------------------------------------

def scan_with_groq(image_path: str) -> Receipt:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in .env file")

    image_bytes = Path(image_path).read_bytes()
    suffix = Path(image_path).suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_map.get(suffix, "image/jpeg")
    b64_image = base64.b64encode(image_bytes).decode()

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "temperature": 0,
    }

    response = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(f"Groq error {response.status_code}: {response.text}")

    raw = response.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown code fences if model wraps the JSON
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    data = json.loads(raw)
    if "items" in data and data["items"]:
        data["items"] = [i for i in data["items"] if i.get("name")]
    return Receipt(**data)


# ---------------------------------------------------------------------------
# Post-processing — sanity fixes
# ---------------------------------------------------------------------------

def post_process(receipt: Receipt) -> Receipt:
    # Normalize currency
    if receipt.currency and receipt.currency.upper() in ("LE", "L.E", "L.E."):
        receipt.currency = "EGP"

    # Reject item prices that look like barcodes
    for item in receipt.items:
        if item.total_price and item.total_price > 50_000:
            item.total_price = None
        if item.unit_price and item.unit_price > 50_000:
            item.unit_price = None

    # Reject total if it looks like a barcode (large whole number)
    if receipt.total and receipt.total > 9_999 and receipt.total == int(receipt.total):
        receipt.total = None

    # Fill total if missing
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
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print(f"=== PurchaseBuddy — Groq Vision Scanner: {IMAGE_PATH} ===\n")

    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not found.")
        print("Get a free key at: console.groq.com → API Keys")
        print("Then add to your .env file:")
        print("  GROQ_API_KEY=your_key_here")
        exit(1)

    print("Sending image to Groq Vision (free)...")
    receipt = scan_with_groq(IMAGE_PATH)
    receipt = post_process(receipt)

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
    if receipt.return_window_days:
        print(f"Return by:     {receipt.return_window_days} days from {receipt.date}")

    print("\nItems:")
    for item in receipt.items:
        name = item.name or "N/A"
        qty = f"{item.quantity}x " if item.quantity else ""
        price = f"{receipt.currency} {item.total_price}" if item.total_price else ""
        print(f"  - {name}: {qty}{price}")
