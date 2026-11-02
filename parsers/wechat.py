from openpyxl import load_workbook
from .base import Transaction, excel_serial_to_date, parse_amount


def parse_xlsx(filepath: str) -> list[Transaction]:
    """Parse WeChat XLSX. Row 2: banner, Row 3: headers, Row 4+: data."""
    wb = load_workbook(filepath, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if len(rows) < 4:
        return []

    # Find header row (contains 交易时间)
    header_idx = None
    for i, row in enumerate(rows):
        if row and row[0] and str(row[0]).strip() == '交易时间':
            header_idx = i
            break

    if header_idx is None:
        return []

    headers = [str(h).strip() if h else "" for h in rows[header_idx]]
    col = {h: i for i, h in enumerate(headers) if h}

    transactions = []
    for row in rows[header_idx + 1:]:
        if not row or not row[0]:
            continue
        first = str(row[0]).strip()
        if first.startswith('-') or not first:
            continue

        date = excel_serial_to_date(row[col.get('交易时间', 0)])
        direction = str(row[col.get('收/支', 4)] or "").strip()
        # WeChat amount column is "金额(元)"
        amount_col = col.get('金额(元)', col.get('金额', 5))
        amount = parse_amount(row[amount_col])
        merchant = str(row[col.get('交易对方', 2)] or "").strip()
        description = str(row[col.get('商品', 3)] or "").strip()
        description = "" if description == "/" else description
        note = str(row[col.get('备注', -1)] or "").strip() if '备注' in col else ""
        note = "" if note == "/" else note
        payment_method = str(row[col.get('支付方式', 6)] or "").strip() if '支付方式' in col else ""
        payment_method = "" if payment_method == "/" else payment_method

        transactions.append(Transaction(
            date=date,
            amount=amount,
            direction=direction,
            merchant=merchant,
            description=description,
            channel="wechat",
            note=note,
            payment_method=payment_method,
        ))

    return transactions
