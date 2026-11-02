from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json


@dataclass
class Transaction:
    date: str           # YYYY-MM-DD
    amount: float       # absolute value
    direction: str      # 收入/支出/不计收支
    merchant: str
    description: str
    channel: str        # alipay/wechat/credit_card
    owner: str = ""
    month: str = ""     # YYYY-MM, auto-derived
    l2: str = ""
    is_fixed: str = "非固定"      # 固定/可变/非固定
    is_necessary: str = "非必要"  # 必要/可压缩/非必要
    note: str = ""
    exclude_from_budget: bool = False
    monthly: str = ""
    payment_method: str = ""  # internal: used for cross-channel dedup, not synced

    def __post_init__(self):
        if not self.month and self.date:
            self.month = self.date[:7]
        if not self.monthly and self.month:
            self.monthly = self.month

    def to_dict(self):
        d = asdict(self)
        d.pop("payment_method", None)
        return d

def excel_serial_to_date(serial) -> str:
    """Convert Excel serial number, datetime object, or string to YYYY-MM-DD."""
    if isinstance(serial, datetime):
        return serial.strftime("%Y-%m-%d")
    if isinstance(serial, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
            try:
                return datetime.strptime(serial.strip(), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return serial.strip()
    if isinstance(serial, (int, float)):
        base = datetime(1899, 12, 30)
        return (base + timedelta(days=serial)).strftime("%Y-%m-%d")
    return str(serial)


def parse_amount(value) -> float:
    """Parse amount from string or number, return absolute value."""
    if isinstance(value, (int, float)):
        return abs(value)
    s = str(value).strip().replace(",", "").replace("¥", "").replace("￥", "")
    try:
        return abs(float(s))
    except ValueError as exc:
        raise ValueError(f"Invalid amount: {value!r}") from exc


def transactions_to_json(transactions: list[Transaction]) -> str:
    return json.dumps([t.to_dict() for t in transactions], ensure_ascii=False, indent=2)
