import xlrd
from .base import Transaction


def parse_xls(filepath: str) -> list[Transaction]:
    """Parse credit card XLS (OLE2). Row 0: headers, Row 1+: data."""
    wb = xlrd.open_workbook(filepath)
    ws = wb.sheet_by_index(0)

    if ws.nrows < 2:
        return []

    headers = [str(ws.cell_value(0, j)).strip() for j in range(ws.ncols)]
    col = {h: i for i, h in enumerate(headers) if h}

    transactions = []
    for i in range(1, ws.nrows):
        row = [ws.cell_value(i, j) for j in range(ws.ncols)]

        date = str(row[col.get('交易日期', 0)]).strip()
        if not date or date == '交易日期':
            continue

        amount_str = row[col.get('交易金额', 6)]
        amount_val = float(str(amount_str).strip()) if amount_str else 0.0

        # Negative amount = refund/credit
        if amount_val < 0:
            direction = "收入"
        else:
            direction = "支出"

        merchant = str(row[col.get('交易描述', 2)] or "").strip()
        card_last4 = str(row[col.get('卡末四位', 3)] or "").strip()
        note = str(row[col.get('备注', 8)] or "").strip()

        transactions.append(Transaction(
            date=date,
            amount=abs(amount_val),
            direction=direction,
            merchant=merchant,
            description=merchant,  # credit card has no separate description
            channel="credit_card",
            owner=f"信用卡({card_last4})" if card_last4 else "",
            note=note,
        ))

    return transactions
