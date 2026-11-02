import csv
import re
from openpyxl import load_workbook
from .base import Transaction, excel_serial_to_date, parse_amount


def _extract_owner_from_csv_lines(lines: list[str]) -> str:
    for line in lines[:10]:
        # Alipay metadata lines may have trailing empty commas (e.g. "姓名：示例用户,,,,,").
        m = re.search(r'姓名[：:]\s*([^,\n]+)', line)
        if m:
            return m.group(1).strip()
    return ""


def parse_csv(filepath: str) -> list[Transaction]:
    """Parse Alipay CSV (GBK or UTF-8 encoded)."""
    for enc in ('utf-8', 'gbk'):
        try:
            with open(filepath, 'r', encoding=enc) as f:
                lines = f.readlines()
            if any('交易时间' in line for line in lines[:30]):
                break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        with open(filepath, 'r', encoding='gbk', errors='replace') as f:
            lines = f.readlines()

    owner = _extract_owner_from_csv_lines(lines)

    # Find the header line (starts with 交易时间)
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('交易时间,'):
            header_idx = i
            break
    if header_idx is None:
        return []

    transactions = []
    reader = csv.reader(lines[header_idx:])
    headers = [h.strip() for h in next(reader)]

    col = {h: i for i, h in enumerate(headers) if h}

    for row in reader:
        if not row or len(row) < len(col):
            continue
        # Skip empty/separator rows
        if not row[0].strip() or row[0].strip().startswith('-'):
            continue

        date_str = row[col.get('交易时间', 0)].strip()
        direction = row[col.get('收/支', 5)].strip()

        # Completely exclude 不计收支 transactions
        if direction == '不计收支':
            continue

        amount = parse_amount(row[col.get('金额', 6)])
        merchant = row[col.get('交易对方', 2)].strip()
        description = row[col.get('商品说明', 4)].strip()
        note = row[col.get('备注', -1)].strip() if '备注' in col else ""
        payment_method = row[col.get('收/付款方式', 7)].strip() if '收/付款方式' in col else ""

        # Parse date: "2026-03-31 05:02:44" or "6/30/2026 21:56"
        if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            date = date_str[:10]
        else:
            m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str)
            date = f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}" if m else date_str

        transactions.append(Transaction(
            date=date,
            amount=amount,
            direction=direction,
            merchant=merchant,
            description=description,
            channel="alipay",
            owner=owner,
            note=note,
            payment_method=payment_method,
        ))

    return transactions


def parse_xlsx(filepath: str) -> list[Transaction]:
    """Parse Alipay XLSX. Row 1: banner, Row 2: headers, Row 3+: data."""
    wb = load_workbook(filepath, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if len(rows) < 3:
        return []

    # Find header row (contains 交易时间)
    header_idx = None
    owner = ""
    for i, row in enumerate(rows):
        row_str = str(row)
        m = re.search(r'姓名[：:]\s*(.+?)[\s\'"]', row_str)
        if m:
            owner = m.group(1).strip()
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
        direction = str(row[col.get('收/支', 5)] or "").strip()

        # Completely exclude 不计收支 transactions
        if direction == '不计收支':
            continue

        amount = parse_amount(row[col.get('金额', 6)])
        merchant = str(row[col.get('交易对方', 2)] or "").strip()
        description = str(row[col.get('商品说明', 4)] or "").strip()
        note = str(row[col.get('备注', -1)] or "").strip() if '备注' in col else ""
        payment_method = str(row[col.get('收/付款方式', 7)] or "").strip() if '收/付款方式' in col else ""

        transactions.append(Transaction(
            date=date,
            amount=amount,
            direction=direction,
            merchant=merchant,
            description=description,
            channel="alipay",
            owner=owner,
            note=note,
            payment_method=payment_method,
        ))

    return transactions
