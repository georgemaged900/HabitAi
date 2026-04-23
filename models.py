from typing import Optional
from pydantic import BaseModel


class ReceiptItem(BaseModel):
    name: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None
    category: Optional[str] = None


class Receipt(BaseModel):
    store_name: Optional[str] = None
    store_address: Optional[str] = None
    date: Optional[str] = None
    transaction_number: Optional[str] = None
    customer_name: Optional[str] = None
    invoice_id: Optional[str] = None
    items: list[ReceiptItem] = []
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    tax_percentage: Optional[float] = None
    total: Optional[float] = None
    currency: Optional[str] = "USD"
    payment_method: Optional[str] = None
    return_window_days: Optional[int] = None
    warranty_info: Optional[str] = None
    notes: Optional[str] = None
