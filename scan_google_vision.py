"""
Receipt Scanner — Google Cloud Vision OCR
------------------------------------------
Google Vision is significantly better than Tesseract for:
- Low quality / blurry photos
- Curved or skewed receipts
- Mixed Arabic + English text
- Handwritten text

Setup (one time):
  1. Go to console.cloud.google.com
  2. Create a project → Enable "Cloud Vision API"
  3. Go to APIs & Services → Credentials → Create Service Account
  4. Download the JSON key file → save it to D:/Programs/Google/vision-key.json
  5. Add to your .env file:
       GOOGLE_APPLICATION_CREDENTIALS=D:/Programs/Google/vision-key.json

Usage:
  Change IMAGE_PATH below to your receipt image, then:
  python scan_google_vision.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import ollama
from google.cloud import vision
from receipt_scanner import Receipt
from mmain import SYSTEM_PROMPT

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

#IMAGE_PATH = "kaireceipt.jpeg"   # <-- change to your receipt image
#IMAGE_PATH = "receiptmagicpay.png"   # <-- change to your receipt image
#IMAGE_PATH = "talabatreceipt.jpeg"   # <-- change to your receipt image
#IMAGE_PATH = "seoudireceipt1.jpeg"   # <-- change to your receipt image
#IMAGE_PATH = "zaracecipt.jpeg"   # <-- change to your receipt image
#IMAGE_PATH = "starbucksreceipt.jpeg"   # <-- change to your receipt image
#IMAGE_PATH = "seoudireceipt2.jpeg"   # <-- change to your receipt image
#IMAGE_PATH = "kai2.jpeg"   # <-- change to your receipt image
#IMAGE_PATH = "starbucks2.jpeg"   # <-- change to your receipt image
#IMAGE_PATH = "pharmacy1.jpeg"   # <-- change to your receipt image
#IMAGE_PATH = "Oscar1.jpeg"   # <-- change to your receipt image
#IMAGE_PATH = "Oscar2.jpeg"   # <-- change to your receipt image
#IMAGE_PATH = "Seoudi3.jpeg"   # <-- change to your receipt image

# I only care more about the storename,payment method, item names and total and return window 
# and warrany and quantity, 
# #no need to fetch the fees,taxes
# or unit price
# i can also track prices of items so when user buys it again, i check for inflation if its price increased 

# ---------------------------------------------------------------------------
# Step 1: Google Vision OCR — far better than Tesseract
# ---------------------------------------------------------------------------

def extract_text_google(image_path: str) -> str:
    """
    Use Google Cloud Vision to extract text from a receipt image.

    Advantages over Tesseract:
    - Handles blurry, dark, or angled photos
    - Understands mixed Arabic/English on the same line
    - No need to install anything locally##
    - Free: 1000 images/month at no cost

    Requires GOOGLE_APPLICATION_CREDENTIALS in .env pointing to your key file.
    """
    client = vision.ImageAnnotatorClient()

    image_bytes = Path(image_path).read_bytes()
    image = vision.Image(content=image_bytes)

    # document_text_detection is better than text_detection for dense receipts
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"Google Vision error: {response.error.message}")

    full_text = response.full_text_annotation.text
    return full_text.strip()


# ---------------------------------------------------------------------------
# Step 1b: Clean OCR text — fix number formats before sending to AI
# ---------------------------------------------------------------------------

ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def clean_ocr_text(text: str) -> str:
    # Convert Arabic-Indic numerals: ٢٥٩٠ → 2590
    text = text.translate(ARABIC_INDIC)
    # Fix European number format: 3.025.44 → 3025.44, 3.449.00 → 3449.00
    text = re.sub(r"(\d)\.(\d{3})\.(\d{2,3})", r"\1\2.\3", text)
    # Remove thousands commas in numbers: 3,449.00 → 3449.00
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    # Remove noise lines: standalone 1-3 char fragments, pure punctuation, lone symbols
    NOISE = re.compile(r"^([*\[\]()\-_.#@/\\|]+|[a-zA-Z]{1,2}|[٠-٩\d]{1,2})$")
    # Normalize spaces within each line but preserve newlines
    lines = [re.sub(r" +", " ", line).strip() for line in text.splitlines()]
    lines = [l for l in lines if l and not NOISE.match(l)]
    # Remove duplicate receipt sections (e.g. Zara Egypt prints English + Arabic copy)
    seen = set()
    deduped = []
    for line in lines:
        if len(line) > 8 and line in seen:
            break  # duplicate section starts here, stop
        seen.add(line)
        deduped.append(line)
    return "\n".join(line for line in deduped if line)



# ---------------------------------------------------------------------------
# Step 2: Ollama — same as before, parse raw text → structured JSON
# ---------------------------------------------------------------------------

def parse_with_ollama(raw_text: str) -> Receipt:
    """Parse OCR text into structured JSON using local Ollama."""
    prompt = f"""{SYSTEM_PROMPT}

# CRITICAL RULES — follow exactly:
# - You a smart receipt parsing assistant. You receive raw OCR text from a receipt image. The text may have typos, weird spacing, or garbled characters because OCR is imperfect. Your job is to understand what the receipt says and extract data into a clean JSON structure.
# - For store_name and item names: if BOTH English and Arabic versions exist in the text, always use the ENGLISH version. Only use Arabic if there is no English equivalent present.
# - For Arabic-only text (no English version exists): keep it exactly as-is in Arabic script. NEVER transliterate Arabic into Latin letters.
# - For customer_name: look for lines like "إسم : VALUE" or "العميل : VALUE" or a name appearing on the line above the word "Customer" — extract the VALUE.
# - For invoice_id: use "رقم مرجعي" value if present.
# - For transaction_number: use "رقم العملية" value if present.
# - For date in DD/MM/YYYY format: the FIRST number is the DAY, the SECOND is the MONTH. Example: 12/10/2025 = October 12 = 2025-10-12.
# - For total: look for "المبلغ الكلي" or "TOTAL" — use that value directly as total.
# - For service/transaction fees (تكلفة الخدمة, service fee, processing fee, رسوم الخدمة): put the amount in tax_amount, NOT as an item.
# - For payment aggregator receipts (MagicPay, Fawry, Masary, etc.): there is ONE item — the service being paid for (e.g. the biller name + amount). Lines like إسم, رقم الحساب, الرقم التاميني, قيمة الفاتورة, رقم مرجعي are metadata fields — do NOT create items from them.
# - For items: combine multi-line product names into ONE item. Do NOT include service fees as items.
#   A line like "1.00 x 1699.00 LE/Units" is a quantity confirmation — it is NOT a separate item, skip it.
#   A product name continues on the next line if that line has no price and no standalone number.
#   Example: "Cream Knitted Crew  1699.00  1" + "Neck Sweater (XL)" + "1.00 x 1699.00 LE/Units" = ONE item named "Cream Knitted Crew Neck Sweater (XL)", quantity=1, unit_price=1699, total_price=1699.
# - NEVER transliterate Arabic words into English letters.
# - Storename can be in english and arabic together usually under each other so extract english better if it exists, but if not extract arabic as-is. Example: "Zara - مصر الجديدة" → store_name = "Zara - مصر الجديدة"
# - for supermarket receipts like seoudi,oscar, carrefour, etc, it can have item then price of that item then its quantity and total of item amount * quantity, also the count number or quantity will be in integer numbers next to the item
# - for supermarket receipt it can have Unit(الوحدة), Price(السعر), quantity (الكمية) Total(الإجمالي) columns and the item name can be in one line and its price and quantity and total in the next line so you have to combine them together to form one item, also for supermarket receipt it can have a line with the word "المجموع" or "Total" that has the total amount of the receipt so you can use that as the total of the receipt instead of summing up the items because sometimes there are discounts or offers that make the total different from the sum of items, also for supermarket receipt it can have a line with the word "الضريبة" or "Tax" that has the tax amount of the receipt so you can use that as the tax_amount of the receipt instead of calculating it from the items because sometimes there are different tax rates for different items
# - for supermarket receipt it can have a line below called quantity or count or piece or pics or "عدد"
# for supermaket receipt fetch the total amount not the unit price and use it to compute, ignore unit price and just use total price for each item, and if quantity is present use it but if quantity is missing assume it is 1, also for supermarket receipt if there is a line with the word "الخصم" or "discount" that has the discount amount of the receipt so you can subtract that from the total to get the final total, also for supermarket receipt if there is a line with the word "طريقة الدفع" or "payment method" that has the payment method of the receipt so you can use that to fill the payment_method field in the JSON output
# - for food delivery receipts (Talabat, Uber Eats, Careem Food, etc.):
#   * store_name = the RESTAURANT name (shown at top, above the order items). If both Arabic and English exist, use English.
#   * customer_name = the DELIVERY RECIPIENT name (shown under "Delivery Address"). NOT the restaurant name.
#   * items = ONLY the food items ordered. "Delivery fee" and "Service fee" are NOT items   

# Parse this receipt OCR text and return ONLY valid JSON matching this exact structure, no explanation:
# {{
#   "store_name": string,
#   "store_address": string or null,
#   "date": string (YYYY-MM-DD) or null,
#   "transaction_number": string or null,
#   "customer_name": string or null,
#   "invoice_id": string or null,
#   "items": [
#     {{
#       "name": string,
#       "quantity": number or null,
#       "unit_price": number or null,
#       "total_price": number or null,
#       "category": string or null
#     }}
#   ],
#   "subtotal": number or null,
#   "tax_amount": number or null,
#   "tax_percentage": number or null,
#   "total": number or null,
#   "currency": string or null,
#   "payment_method": string or null,
#   "return_window_days": number or null,
#   "warranty_info": string or null,
#   "notes": string or null
# }}   

EXTRA RULES:
- TOTAL: Find the line containing "TOTAL" or "المبلغ الكلي". The total is the number on that same line. If no number is on the same line, use the number on the line immediately ABOVE "TOTAL". Use it directly — NEVER add, subtract, or compute the total from other values.
- If a line has TWO numbers like "318.07 2590.00" and it is near a VAT/tax label, the SECOND number is the total. Use it directly as total.
- subtotal and tax_amount: extract only if clearly labeled. If unclear, leave as null. Do NOT compute them.
- If multiple dates exist, prefer the card/transaction date (format DD-MM-YYYY next to "Date:" label) over the date at the top of the receipt which may have OCR digit errors.
- BARCODE ITEM LINES: A line starting with a long number (8 or more digits) followed by a product name is an item. Example: "0806243040644 TROUSERS" → item name = "TROUSERS". Extract the text after the barcode as the item name.
- TAX SUMMARY ROWS (NOT items): Any line that contains "VAT", "TAX", "%" followed by a number, OR starts with a 2-4 letter tax code (like SMC, VAT, TAX) — is a tax summary row, NOT an item. Examples: "SMC 14.00 1 2590.00", "13.51 14% VAT", "VAT 14%" — all are tax rows, skip them as items. NEVER use the number from a tax row as an item price.
- PRICE ON NEXT LINE: In café/POS receipts, an item name may appear alone on one line and its price on the very next line. Example:
    "1 GR LATTE"   ← item name (quantity=1 prefix)
    "110.00"       ← this is the price for GR LATTE
    "Coconut Milk" ← modifier/add-on (no price), append to previous item name
    "13.51 14% VAT"← tax row, skip
  Result: ONE item — name="GR LATTE with Coconut Milk", quantity=1, unit_price=110.00, total_price=110.00
- ITEM ADD-ONS / MODIFIERS: A line with no price that comes after an item (or its price line) is a customization — append it to that item's name with a space. Do NOT create a separate item.
- QUANTITY PREFIX: If an item name starts with a number like "1 GR LATTE" or "2 Croissant", that leading number is the quantity — strip it from the name. name="GR LATTE", quantity=1.
- NEVER create items from: SMS, phone numbers, mobile numbers, card numbers, auth codes, RRN, application IDs, QR codes, barcodes (8+ digit numbers alone), website URLs, email addresses, or purely numeric lines with more than 8 digits.
- CRITICAL — NO HALLUCINATION: Every item name you output MUST appear word-for-word in the OCR text below. Do NOT invent, guess, translate, or carry over items from memory or training data. If a word is not in the OCR text, do not use it. If you cannot identify a clear item name from the text, skip it.
- customer_name: look for a name BELOW the word "Customer" on the next line. A string like "gorg 01229411" means customer_name="gorg" and 01229411 is a loyalty/account number — split them.
- store_address should be a physical address. Email or website (http/www/@) is NOT a store address — set it to null.
- FOR FOOD DELIVERY SCREENSHOTS (Talabat, Uber Eats, Careem Food, etc.):
  * store_name = the RESTAURANT name (shown at top, above the order items). If both Arabic and English exist, use English.
  * customer_name = the DELIVERY RECIPIENT name (shown under "Delivery Address"). NOT the restaurant name.
  * items = ONLY the food items ordered. "Delivery fee" and "Service fee" are NOT items.
  * total = the final amount labeled "Total" — use directly, do not add fees.
  * payment_method = extract from "Payment method" line (e.g. "Debit/Credit Card", "Cash", etc.).
- ITEM COMBINING: A product name that continues on the next line MUST be merged into ONE item. The rule: if a line has no price and the previous line had a price, the current line is a name continuation — append it to the previous item name.
  Example from a clothing store receipt:
    Line 1: "Cream Knitted Crew 1699.00 1"  → item starts, name="Cream Knitted Crew", price=1699, qty=1
    Line 2: "Neck Sweater (XL)"             → no price → append to previous → name="Cream Knitted Crew Neck Sweater (XL)"
    Line 3: "1.00 x 1699.00 LE/Units"       → quantity confirmation, skip
    Line 4: "Mocha Knitted Quarter 1750.00 1" → NEW item starts
    Line 5: "Zip Sweater (XL)"              → no price → append → name="Mocha Knitted Quarter Zip Sweater (XL)"
    Line 6: "1.00 x 1750.00 LE/Units"       → skip
  Result: exactly 2 items.

Receipt text:
{raw_text}"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )

    raw_json = response["message"]["content"]
    data = json.loads(raw_json)
    return Receipt(**data)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"=== PurchaseBuddy — Google Vision Scanner: {IMAGE_PATH} ===\n")

    # Check credentials
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds or not Path(creds).exists():
        print("ERROR: Google Vision credentials not found.")
        print("Add this to your .env file:")
        print("  GOOGLE_APPLICATION_CREDENTIALS=D:/Programs/Google/vision-key.json")
        print("And make sure the file exists.")
        exit(1)

    # Step 1: Google Vision OCR
    print("Step 1: Google Vision OCR...")
    raw_text = extract_text_google(IMAGE_PATH)
    raw_text = clean_ocr_text(raw_text)
    print(f"\n--- Raw OCR Output ---\n{raw_text}\n----------------------\n")

    # Step 2: Ollama parsing
    print("Step 2: Ollama parsing into JSON...")
    receipt = parse_with_ollama(raw_text)

    # Fix 1: OCR store name corrections (common misreads)
    OCR_STORE_FIXES = {"ZABA": "ZARA", "AOIDAS": "ADIDAS", "NKIE": "NIKE", "H$M": "H&M", "ZEBA": "ZARA"}
    if receipt.store_name:
        receipt.store_name = OCR_STORE_FIXES.get(receipt.store_name.strip().upper(), receipt.store_name)
    else:
        # store_name is null — pull first meaningful line from raw text
        for line in raw_text.splitlines():
            line = line.strip()
            if line and len(line) >= 3 and not re.fullmatch(r'[\d\s.,:/\-]+', line):
                receipt.store_name = OCR_STORE_FIXES.get(line.upper(), line)
                break

    # Fix 2a: Short-form date like "7 Apr 26" → 2026-04-07
    MONTH_MAP = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,
                 'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    short_date = re.search(r'\b(\d{1,2})\s+([A-Za-z]{3})\s+(\d{2})\b', raw_text)
    if short_date:
        sd, smon, syr = short_date.groups()
        mon_num = MONTH_MAP.get(smon.lower())
        if mon_num:
            receipt.date = f"{2000 + int(syr)}-{mon_num:02d}-{int(sd):02d}"

    # Fix 2b: Date year is in the future (OCR digit misread) — fall back to card transaction date
    if receipt.date:
        year_match = re.search(r'(\d{4})', str(receipt.date))
        if year_match and int(year_match.group(1)) > 2026:
            card_date = re.search(r'\b(\d{2})-(\d{2})-(\d{4})\b', raw_text)
            if card_date:
                d, m, y = card_date.groups()
                receipt.date = f"{y}-{m}-{d}"

    # Fix 2c: Override currency with explicit code found in raw text (Ollama often guesses wrong)
    for code in ['EGP', 'SAR', 'AED', 'KWD', 'GBP', 'EUR', 'USD', 'QAR']:
        if re.search(r'\b' + code + r'\b', raw_text):
            receipt.currency = code
            break

    # Fix 3: Extract total directly from raw text (Ollama often computes wrong values).
    # Strategy: collect all plausible numbers in a small window around the TOTAL label,
    # then take the MAXIMUM — the total is always the largest amount (>= subtotal, >= any item).
    # min_plausible filters out VAT percentages and small codes near the label.
    # Cap at 50,000 so a barcode misread as an item price doesn't break total extraction
    item_prices = [i.total_price for i in receipt.items if i.total_price and i.total_price <= 50_000]
    max_item_price = max(item_prices, default=0)
    min_plausible = max(max_item_price, 1)

    def plausible_nums(text_line):
        results = []
        for m in re.finditer(r'\d+(?:\.\d+)?', text_line):
            n = float(m.group())
            if n < min_plausible or n > 1_000_000:
                continue
            # Reject bare integers > 999 — likely a code/barcode, not a price
            # Real prices above 999 always have a decimal (e.g. 1699.00, 2590.00)
            if n > 999 and '.' not in m.group():
                continue
            results.append(n)
        return results

    ocr_total = None
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    def find_total_near_label(pattern):
        for i, line in enumerate(lines):
            if re.search(pattern, line, re.IGNORECASE):
                window = lines[max(0, i - 2):i] + [line] + lines[i + 1:min(len(lines), i + 9)]
                candidates = []
                for wline in window:
                    candidates.extend(plausible_nums(wline))
                if candidates:
                    return max(candidates)
        return None

    # 1. Prefer explicit TOTAL label (not Subtotal)
    ocr_total = find_total_near_label(r'(?<![Ss]ub)\bTOTAL\b|Total with Tax|إجمالي مع الضريبة|المبلغ الكلي|الإجمالي')
    # 2. Fall back to Subtotal label if no TOTAL found (some receipts only have Subtotal)
    if ocr_total is None:
        ocr_total = find_total_near_label(r'\bSubtotal\b|\bSUBTOTAL\b|المجموع')

    # Reject Ollama's total if it looks like a barcode (no decimal and > 9999)
    if receipt.total and receipt.total > 9999 and receipt.total == int(receipt.total):
        receipt.total = None

    if ocr_total is not None:
        receipt.total = ocr_total
    elif not receipt.total:
        items_sum = sum(i.total_price for i in receipt.items if i.total_price and i.total_price <= 50_000)
        if items_sum > 0:
            receipt.total = round(items_sum, 2)

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



# CRITICAL RULES — follow exactly:
# - You a smart receipt parsing assistant. You receive raw OCR text from a receipt image. The text may have typos, weird spacing, or garbled characters because OCR is imperfect. Your job is to understand what the receipt says and extract data into a clean JSON structure.
# - For store_name and item names: if BOTH English and Arabic versions exist in the text, always use the ENGLISH version. Only use Arabic if there is no English equivalent present.
# - For Arabic-only text (no English version exists): keep it exactly as-is in Arabic script. NEVER transliterate Arabic into Latin letters.
# - For customer_name: look for lines like "إسم : VALUE" or "العميل : VALUE" or a name appearing on the line above the word "Customer" — extract the VALUE.
# - For invoice_id: use "رقم مرجعي" value if present.
# - For transaction_number: use "رقم العملية" value if present.
# - For date in DD/MM/YYYY format: the FIRST number is the DAY, the SECOND is the MONTH. Example: 12/10/2025 = October 12 = 2025-10-12.
# - For total: look for "المبلغ الكلي" or "TOTAL" — use that value directly as total.
# - For service/transaction fees (تكلفة الخدمة, service fee, processing fee, رسوم الخدمة): put the amount in tax_amount, NOT as an item.
# - For payment aggregator receipts (MagicPay, Fawry, Masary, etc.): there is ONE item — the service being paid for (e.g. the biller name + amount). Lines like إسم, رقم الحساب, الرقم التاميني, قيمة الفاتورة, رقم مرجعي are metadata fields — do NOT create items from them.
# - For items: combine multi-line product names into ONE item. Do NOT include service fees as items.
#   A line like "1.00 x 1699.00 LE/Units" is a quantity confirmation — it is NOT a separate item, skip it.
#   A product name continues on the next line if that line has no price and no standalone number.
#   Example: "Cream Knitted Crew  1699.00  1" + "Neck Sweater (XL)" + "1.00 x 1699.00 LE/Units" = ONE item named "Cream Knitted Crew Neck Sweater (XL)", quantity=1, unit_price=1699, total_price=1699.
# - NEVER transliterate Arabic words into English letters.
# - Storename can be in english and arabic together usually under each other so extract english better if it exists, but if not extract arabic as-is. Example: "Zara - مصر الجديدة" → store_name = "Zara - مصر الجديدة"
# - for supermarket receipts like seoudi,oscar, carrefour, etc, it can have item then price of that item then its quantity and total of item amount * quantity, also the count number or quantity will be in integer numbers next to the item
# - for supermarket receipt it can have Unit(الوحدة), Price(السعر), quantity (الكمية) Total(الإجمالي) columns and the item name can be in one line and its price and quantity and total in the next line so you have to combine them together to form one item, also for supermarket receipt it can have a line with the word "المجموع" or "Total" that has the total amount of the receipt so you can use that as the total of the receipt instead of summing up the items because sometimes there are discounts or offers that make the total different from the sum of items, also for supermarket receipt it can have a line with the word "الضريبة" or "Tax" that has the tax amount of the receipt so you can use that as the tax_amount of the receipt instead of calculating it from the items because sometimes there are different tax rates for different items
# - for supermarket receipt it can have a line below called quantity or count or piece or pics or "عدد"

# Parse this receipt OCR text and return ONLY valid JSON matching this exact structure, no explanation:
# {{
#   "store_name": string,
#   "store_address": string or null,
#   "date": string (YYYY-MM-DD) or null,
#   "transaction_number": string or null,
#   "customer_name": string or null,
#   "invoice_id": string or null,
#   "items": [
#     {{
#       "name": string,
#       "quantity": number or null,
#       "unit_price": number or null,
#       "total_price": number or null,
#       "category": string or null
#     }}
#   ],
#   "subtotal": number or null,
#   "tax_amount": number or null,
#   "tax_percentage": number or null,
#   "total": number or null,
#   "currency": string or null,
#   "payment_method": string or null,
#   "return_window_days": number or null,
#   "warranty_info": string or null,
#   "notes": string or null
# }}