"""Monthly bills: prepare locally, review in this session, then sync."""
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import click

from configuration import load_config
from parsers.base import Transaction, transactions_to_json
from parsers.detector import detect_source
from parsers import alipay, wechat, credit_card

L2_CATEGORIES = (
    "房租", "水电气电话费", "在家吃饭", "外出吃饭", "日用品", "宠物",
    "交通出行", "锻炼/运动", "日常活动", "旅游", "医疗健康", "人情/礼物", "杂项/意外", "其他",
)


@click.group()
def cli():
    """Monthly bills: prepare → review → sync."""


def parse_file(filepath: str, source_type: str | None = None) -> list[Transaction]:
    """Parse a single file, auto-detecting source if not specified."""
    if source_type is None:
        source_type = detect_source(filepath)
    if source_type is None:
        raise click.ClickException(f"Cannot detect source type for: {os.path.basename(filepath)}")

    match source_type:
        case 'alipay_csv':
            return alipay.parse_csv(filepath)
        case 'alipay_xlsx':
            return alipay.parse_xlsx(filepath)
        case 'wechat':
            return wechat.parse_xlsx(filepath)
        case 'credit_card':
            return credit_card.parse_xls(filepath)
        case _:
            raise click.ClickException(f"Unsupported source type: {source_type}")


def _load_from_json(filepath: str) -> list[Transaction]:
    try:
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Expected a transaction array")
        aliases = load_config().get("owner_aliases", {})
        transactions = [Transaction(**rec) for rec in data]
        for transaction in transactions:
            transaction.owner = aliases.get(transaction.owner, transaction.owner)
        return transactions
    except (ValueError, TypeError) as exc:
        raise click.ClickException(f"Invalid transaction JSON: {exc}") from exc


@cli.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--owner", required=True, help="Owner name from private/config.json")
@click.option("--output", "-o", type=click.Path(dir_okay=False), help="Default: artifacts/<month>/<owner>.json")
def prepare(files, owner, output):
    """Parse and deduplicate bills. No AI calls; never overwrites an existing file."""
    owners = load_config().get("owners", [])
    if not owner.strip() or (owners and owner not in owners):
        raise click.ClickException("Invalid --owner; check private/config.json")
    transactions = []
    for filepath in files:
        try:
            batch = parse_file(filepath)
            if not batch:
                raise ValueError("No transactions found; check the bill format")
            for transaction in batch:
                transaction.owner = owner
            transactions.extend(batch)
            click.echo(f"{Path(filepath).name}: {len(batch)} records", err=True)
        except Exception as exc:
            raise click.ClickException(f"Cannot parse {Path(filepath).name}: {exc}") from exc
    transactions, dropped = _dedup_cross_channel(transactions)
    months = sorted({t.month for t in transactions})
    if output is None and len(months) != 1:
        raise click.ClickException("Multiple months found; specify --output explicitly")
    target = Path(output) if output else Path("artifacts") / months[0] / f"{owner}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as stream:
            stream.write(transactions_to_json(transactions))
    except FileExistsError as exc:
        raise click.ClickException(f"Already exists: {target}; choose another --output") from exc
    click.echo(f"{target}: {len(transactions)} records; {dropped} cross-channel duplicates removed")


@cli.command()
@click.argument("json_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--summary-only", is_flag=True, help="Only totals; omit transaction rows")
@click.option("--pending-only", is_flag=True, help="Only rows with missing L2 classification")
def review(json_file, summary_only, pending_only):
    """Compact summary + TSV rows. Row IDs are zero-based JSON array indexes."""
    transactions = _load_from_json(json_file)
    totals = defaultdict(Decimal)
    for t in transactions:
        if t.direction == "支出" and not t.exclude_from_budget:
            totals[(t.month, t.owner)] += Decimal(str(t.amount))
    click.echo(f"Records: {len(transactions)}; missing L2: {sum(not t.l2 for t in transactions)}")
    for (month, owner), amount in sorted(totals.items()):
        click.echo(f"{month} {owner}: budget expense {amount:.2f}")
    if summary_only:
        return
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(["id", "date", "owner", "amount", "direction", "channel", "merchant", "description", "L2", "fixed", "necessary", "excluded", "note"])
    for index, t in enumerate(transactions):
        if pending_only and t.l2:
            continue
        writer.writerow([index, t.date, t.owner, t.amount, t.direction, t.channel, t.merchant,
                         t.description if t.description != t.merchant else "", t.l2,
                         t.is_fixed, t.is_necessary, int(t.exclude_from_budget), t.note])


def _dedup_cross_channel(transactions: list[Transaction]) -> tuple[list[Transaction], int]:
    """Drop matching card entries, respecting owner, direction and multiplicity."""
    def key(t):
        return t.owner, t.date, t.amount, t.direction

    backed = Counter(key(t) for t in transactions
                     if t.channel in ("alipay", "wechat") and "信用卡" in t.payment_method)
    result = []
    dropped = 0
    for t in transactions:
        if t.channel == "credit_card" and backed[key(t)] > 0:
            backed[key(t)] -= 1
            dropped += 1
        else:
            result.append(t)
    result.sort(key=lambda t: t.date, reverse=True)
    return result, dropped


CHANNEL_MAP = {
    'alipay': '支付宝',
    'wechat': '微信',
    'credit_card': '信用卡',
}


def _to_base_record(transaction: Transaction) -> dict:
    """Convert a Transaction to a Feishu Base record field map."""
    return {
        'Date': f"{transaction.date} 00:00:00",
        'Amount': transaction.amount,
        'Direction': transaction.direction,
        'Merchant': transaction.merchant,
        'Description': transaction.description,
        'Channel': CHANNEL_MAP.get(transaction.channel, transaction.channel),
        'Owner': transaction.owner,
        'L2': transaction.l2,
        'IsFixed': transaction.is_fixed,
        'IsNecessary': transaction.is_necessary,
        'Note': transaction.note,
        'ExcludeFromBudget': bool(transaction.exclude_from_budget),
        'SyncedAt': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def _call_lark_record_batch_create(
    base_token: str,
    table_id: str,
    records: list[dict],
) -> dict:
    """Call lark-cli base +record-batch-create with a batch of records."""
    # lark-cli's Base shortcut expects a column-oriented payload, rather than
    # the legacy OpenAPI ``create_records`` shape.
    fields = list(records[0]) if records else []
    payload = {
        'fields': fields,
        'rows': [
            [record.get(field) if record.get(field) != "" else None for field in fields]
            for record in records
        ],
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    # lark-cli requires --json @file.json paths to be relative to cwd.
    temp_name = f".lark-batch-{os.getpid()}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}.json"
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(payload_json, encoding='utf-8')
        cmd = [
            'lark-cli', 'base', '+record-batch-create',
            '--base-token', base_token,
            '--table-id', table_id,
            '--as', 'user',
            '--json', f'@{temp_name}',
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError:
            response = {
                'ok': False,
                'error': {'message': f'Invalid JSON from lark-cli: {result.stdout[:500]}'},
            }

        if result.returncode != 0 or not response.get('ok'):
            err_msg = result.stderr.strip() if result.stderr else str(response)
            raise RuntimeError(f"lark-cli failed (exit {result.returncode}): {err_msg}")

        return response
    finally:
        if temp_path.exists():
            temp_path.unlink()


@cli.command()
@click.argument('json_file', type=click.Path(exists=True, dir_okay=False))
@click.option('--base-token', envvar='FEISHU_APP_TOKEN', default=None, help='Feishu Base token')
@click.option('--table-id', envvar='FEISHU_TABLE_ID', default=None, help='Table ID')
@click.option('--batch-size', type=click.IntRange(1, 200), default=200, show_default=True, help='Records per batch (max 200)')
@click.option('--dry-run', is_flag=True, help='Preview requests without writing to Base')
def sync(json_file, base_token, table_id, batch_size, dry_run):
    """Write reviewed transactions to Transaction via lark-cli (append only)."""
    config = load_config()
    base_token = base_token or config.get("base_token")
    table_id = table_id or config.get("tables", {}).get("transactions")
    owners = config.get("owners", [])
    if not base_token:
        raise click.ClickException("Set base_token in private/config.json, --base-token or FEISHU_APP_TOKEN")
    if not table_id:
        raise click.ClickException("Set tables.transactions in private/config.json, --table-id or FEISHU_TABLE_ID")
    transactions = _load_from_json(json_file)
    click.echo(f"Loaded {len(transactions)} transactions from {os.path.basename(json_file)}", err=True)

    if not transactions:
        click.echo("No transactions to sync.", err=True)
        return

    for index, t in enumerate(transactions):
        try:
            datetime.strptime(t.date, "%Y-%m-%d")
            if not isinstance(t.owner, str) or not t.owner.strip() or (owners and t.owner not in owners):
                raise ValueError("invalid owner")
            if isinstance(t.amount, bool) or not math.isfinite(t.amount):
                raise ValueError("invalid amount")
            if t.direction not in ("收入", "支出", "不计收支"):
                raise ValueError("invalid direction")
            if not isinstance(t.exclude_from_budget, bool):
                raise ValueError("exclude_from_budget must be boolean")
            if t.direction == "支出" and not t.exclude_from_budget:
                if t.l2 not in L2_CATEGORIES:
                    raise ValueError("missing or invalid L2; review first")
                if t.is_fixed not in ("固定", "可变", "非固定") or t.is_necessary not in ("必要", "可压缩", "非必要"):
                    raise ValueError("invalid fixed/necessary classification")
        except (ValueError, TypeError) as exc:
            raise click.ClickException(f"Row {index}: {exc}") from exc
    records = [_to_base_record(t) for t in transactions]

    source = Path(json_file)
    if dry_run:
        target = source.with_name(f"{source.stem}-sync-payload.json")
        target.write_text(json.dumps({"base_token": base_token, "table_id": table_id, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        click.echo(f"Preview: {len(records)} records → {target}; no network calls")
        return

    receipt = source.with_name(f"{source.stem}-sync-receipt.jsonl")
    try:
        stream = receipt.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise click.ClickException(f"Sync receipt exists: {receipt}; check Feishu before retrying") from exc
    total = 0
    with stream:
        stream.write(json.dumps({"status": "started", "base_token": base_token, "table_id": table_id, "count": len(records)}) + "\n")
        stream.flush()
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                response = _call_lark_record_batch_create(base_token, table_id, batch)
            except Exception as exc:
                raise click.ClickException(f"Sync stopped after {total} confirmed records: {exc}. Check {receipt} and Feishu; do not blindly rerun.") from exc
            stream.write(json.dumps({"offset": i, "count": len(batch), "response": response}, ensure_ascii=False) + "\n")
            stream.flush()
            total += len(batch)
            click.echo(f"Synced {total}/{len(records)}", err=True)
    click.echo(f"Created {total} records; receipt: {receipt}")


if __name__ == "__main__":
    cli()
