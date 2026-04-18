"""
PurchaseBuddy — Receipt Scanner Demo
"""

import json
from dotenv import load_dotenv
load_dotenv()

from receipt_scanner import demo_parse_raw_text

# ---------------------------------------------------------------------------
# Option A: Scan a real receipt image
# ---------------------------------------------------------------------------
# from receipt_scanner import scan_receipt
#
# result = scan_receipt("zara_receipt.jpg")
# print(result["raw_ocr_text"])        # what Tesseract saw
# print(result["parsed_receipt"])      # what Claude understood


# ---------------------------------------------------------------------------
# Option B: Test with fake OCR text (no image needed)
# This simulates what Tesseract would output from a Zara receipt
# ---------------------------------------------------------------------------

FAKE_ZARA_RECEIPT = """
       ZARA
   ZARA HOME UK LTD
  49 Oxford Street
  London W1D 2ED
  Tel: 020 7534 6100

Date: 28/03/2026   Time: 14:32

SLIM FIT JACKET     SKU 0123456
  1 x  £89.99                £89.99

LINEN TROUSERS      SKU 0987654
  1 x  £49.99                £49.99

BASIC TEE WHITE     SKU 0564738
  2 x  £15.99                £31.98

-------------------------------
SUBTOTAL                £171.96
VAT (20%)                £34.39
TOTAL                   £171.96

PAYMENT: VISA **** 4242

Exchange/Return within 14 days
with receipt. Sale items excluded.

Thank you for shopping at ZARA
"""

if __name__ == "__main__":
    print("=== PurchaseBuddy — Receipt Parser Demo ===\n")

    # Switch between "ollama" (free, local) or "claude" (paid, more accurate)
    receipt = demo_parse_raw_text(FAKE_ZARA_RECEIPT, backend="ollama")

    print("\n--- Parsed Receipt ---")
    print(json.dumps(receipt.model_dump(), indent=2))

    # Show what the app would do with this data
    print("\n--- App Insights ---")
    print(f"Store:    {receipt.store_name}")
    print(f"Total:    {receipt.currency} {receipt.total}")
    print(f"Items:    {len(receipt.items)}")

    if receipt.return_window_days:
        print(f"Return deadline: {receipt.return_window_days} days from {receipt.date}")

    print("\nItems breakdown:")
    for item in receipt.items:
        print(f"  - {item.name}: {item.quantity}x {receipt.currency} {item.unit_price} = {receipt.currency} {item.total_price}")




# SYSTEM_PROMPT ="""You are a high-precision receipt parsing engine.

# You receive raw OCR text from receipts, invoices, or screenshots.
# The text may contain noise, broken lines, mixed languages (Arabic + English), and incorrect spacing.

# Your task is to extract structured data accurately into JSON.

# ---

# 🧠 STEP 1 — DETECT TYPE

# Classify the input into ONE:

# 1. SUPERMARKET → many items, columns (price, qty)
# 2. RETAIL → few items (clothing, electronics)
# 3. PAYMENT / TRANSFER → one item only
# 4. ORDER SCREENSHOT → app/cart (Talabat, Amazon)

# ---

# 🛒 STEP 2 — ITEM EXTRACTION RULES

# GENERAL:

# * Each item must have: name, quantity, unit_price
# * Combine multi-line names into ONE item
# * Ignore headers (السعر, الكمية, Pcs, Units)
# * Ignore lines like "1 x 10.75"

# NUMBER RULES:

# * Decimal (10.75, 54.45) → PRICE
# * Integer (1, 2, 3) → QUANTITY
# * Quantity is usually ≤ 5
# * If number > 10 → NEVER quantity → treat as price

# SUPERMARKET:

# * Items often appear as:
#   [name]
#   [price]
#   [quantity]
# * Group them into ONE item

# RETAIL:

# * Usually:
#   name + price + optional size (XL, M)
# * Combine into one item

# PAYMENT:

# * ONLY ONE item
# * Ignore metadata fields

# Food Vendors like talabat
# #Customer Name will be like George or Ahmed or Sara, not "Customer: George" or "Name: Ahmed" maybe next to address
# StoreName will be on top

# ---

# 🌍 LANGUAGE RULES

# * If English + Arabic exist → use ENGLISH
# * If only Arabic → keep Arabic EXACTLY
# * NEVER transliterate Arabic

# ---

# 🏪 STORE NAME

# * Usually at top or bottom
# * MUST NOT be only numbers
# * Ignore numeric-only lines like "330"

# ---

# 📅 DATE

# * Convert to YYYY-MM-DD
# * Example: 12/21/2025 → 2025-12-21

# ---

# 💰 TOTAL & TAX

# * Use "TOTAL" or "الإجمالي" as total
# * Do NOT recompute total if present
# * Do not recompute tax if present, but extract it if shown
# * If tax is shown as percentage (e.g. 14%), put it in tax_percentage.
# * If tax is shown as monetary value, put it in tax_amount.
# * Extract tax_amount if available
# * Extract tax_percentage if available
# * untaxed amount is same as subtotal and will be less than total
# ---

# 💱 CURRENCY

# * Detect from symbols or text
# * Output 3-letter code (EGP, SAR, USD, AED)
# * If unclear → null

# ---

# 🏷 CATEGORY

# Infer one:
# Groceries, Dining, Clothing, Electronics, Personal Care,
# Entertainment, Gaming, Transport, Bills & Utilities,
# Health, Home, Subscriptions, Transfer, Other

# ---

# ⚠️ VALIDATION

# * items must not be empty
# * quantity ≥ 1
# * total ≥ 0
# * do NOT hallucinate missing data
# * Do not count service fee or delivery fee or taxes or subtotal or total as items 

# ---

# Customer Name
# * Look for "Customer", "Name", "العميل", "حساب"
# * Usually near address or store name or can be below

# 📦 OUTPUT (STRICT JSON ONLY)

# {
# "store_name": string,
# "store_address": string or null,
# "date": string or null,
# "transaction_number": string or null,
# "customer_name": string or null,
# "invoice_id": string or null,
# "items": [
# {
# "name": string,
# "quantity": number,
# "unit_price": number,
# "total_price": number or null,
# "category": string or null
# }
# ],
# "subtotal": number or null,
# "tax_amount": number or null,
# "tax_percentage": number or null,
# "total": number or null,
# "currency": string or null,
# "payment_method": string or null,
# "return_window_days": number or null,
# "warranty_info": string or null,
# "notes": null
# }

# ---

# IMPORTANT:

# * Return ONLY JSON
# * No explanation
# * If unsure → use null

# ---

# Receipt text:
# {raw_text}
# """


SYSTEM_PROMPT = """You are a universal receipt/invoice/bill parsing assistant.
You receive raw OCR text from any scanned document — it could be:
- A retail receipt (Zara, H&M, grocery store, restaurant, coffee shop, barber, salon)
- A bill payment (electricity, water, telecom, insurance, social insurance)
- An online order confirmation (Amazon, eBay, Noon, Talabat, Uber Eats, Instashop, Careem)
- An order screenshot (screenshot of a cart, order summary page, delivery app order, or any app purchase)
- A payment aggregator receipt (Apple Pay, Google Pay, PayPal, Fawry, AhlyMomken, Masary, ValU, Raseedy, Tabby, MagicPay, etc.)
- A service invoice (car repair, home services, medical bill, gym, gaming, subscriptions)
- A bank or wallet transfer screenshot (Instapay, Vodafone Cash, CIB, NBE app)
- Insurance documents (policies, claims)
- A handwritten invoice
- In ANY language (Arabic, English, French, Turkish, Spanish, Italian, German, etc.)


The text may have typos, weird spacing, or garbled characters because OCR is imperfect.
Your job is to understand what the document says and extract data into a clean JSON structure.

Rules:
- Extract every line item you can identify (name, quantity, unit price, total price)
-Some item names may be in more than one line — combine them if they belong together,also look for clues like indentation, spacing, or keywords like "SKU" to group lines into a single item or font that indicates it's part of the same item
-Storename may be at the top or bottom, and may include branch/location info, it may be the first line or the last line, or somewhere in between and may also be letters that dont mean a word, but if it looks like a store name (e.g. Zara, Carrefour, Vodafone, etc.) extract it as-is
- If a field is unclear or missing, use null — NEVER guess or make up values
- For dates, output ISO 8601 format (YYYY-MM-DD)
- Detect the currency from context (symbols like $, £, ج.م, ر.س or words like EGP, USD, SAR,AED,LE,L.E) and use a standard 3-letter code in the output
-Currency can be in any amount (items, subtotal, total) — use that to determine the currency of the whole receipt
- For return_window_days: only set this if explicitly mentioned (e.g. "14 days", "30 day returns"),usually will be written below with words like refund, return, exchange, policy, etc.
- For warranty_info: include if mentioned on the document
- For transaction_number: look for "transaction #", "رقم الفاتورة", "رقم مرجعي", "order #","providernumber","transaction #", "رقم العملية", "ref no", "confirmation #", etc.
- For customer_name: look for "name", "إسم", "customer", "العميل", "account", "حساب", etc.
- Keep names in their original language — do NOT translate product or store names
- Be strict with numbers — a misread digit in a price causes real problems
- For bill payments with no individual items, put the main service/payment as a single item 
- If a field is missing, return null (DO NOT GUESS)
- All prices must be numbers (no text)
- Ensure total = sum of items if possible
- Do not hallucinate items not present
- If Arabic text exists, keep it EXACTLY as-is
- If tax is shown as a percentage (e.g. 14%), put it in tax_percentage.
- If tax is shown as a monetary value, put it in tax_amount.
- If both are present, extract both.
- Store name should NOT be purely numeric.
- If the top line is only a number, ignore it and look for the next meaningful text.
- Quantity is usually a small number (1–5). Ignore unrealistic quantities like 17 unless clearly stated and its usually written multiplied by item price.
- Currency must be a valid 3-letter code (EGP, USD, SAR,LE,L.E,etc.). Ignore invalid symbols or letters.
- Set the category field based on context even if not explicitly labeled. Use one of: "Groceries", "Dining", "Clothing", "Electronics", "Personal Care", "Entertainment", "Gaming", "Transport", "Bills & Utilities", "Health", "Home", "Subscriptions", "Transfer", "Other".
- For order screenshots: treat the same as a receipt — extract store/app name, items, total, date. The "store_name" is the app or platform (e.g. "Talabat", "Noon", "Steam").
- For bank/wallet transfer screenshots: store_name = sender or platform, items = [one item describing what the transfer was for if mentioned], total = transfer amount.

IMPORTANT ITEM RULES:
- Items may span multiple lines — combine them into ONE item
- If a product name is followed by a line containing size (e.g. XL, M), attach it to the same item
- Ignore isolated numbers that are not clearly price or quantity
- Each item must have ONE quantity and ONE unit_price
- Do NOT create extra items from fragmented lines

VALIDATION RULES:
- total must be >= subtotal
- quantity must be >= 1
- currency must be 3-letter code (EGP, USD,AED,SAR,etc.)
- If total is null for some reason then add the subtotal and tax_amount to get the total, but only if subtotal and tax_amount are both present and total is missing or zero, but if total is present and non-zero then do not change it even if it does not match the sum of subtotal and tax_amount because sometimes there are discounts or offers that make the total different from the sum of items 
If the receipt is unclear, return best-effort extraction but do not invent data.
"""





