# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "requests",
#     "openpyxl",
#     "openai",
#     "python-dotenv",
#     "httpx[socks]",
# ]
# ///

import os
import re
import time
import json
import traceback
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import requests
from openai import AzureOpenAI

# ----------------------------
# REQUIRED ENV
# ----------------------------
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")

# From your Feishu Base URL:
BITABLE_APP_TOKEN = os.getenv("BITABLE_APP_TOKEN", "")
BITABLE_TABLE_ID = os.getenv("BITABLE_TABLE_ID", "")

# Azure OpenAI
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2025-04-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2")

# Behavior
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
DRY_RUN = False

DEFAULT_OWNER = os.getenv("DEFAULT_OWNER", "共同")
DEFAULT_CHANNEL = os.getenv("DEFAULT_CHANNEL", "支付宝")

# Your Feishu Transactions table columns are EXACTLY these names:
FIELDS = {
    "Date": "Date",
    "Amount": "Amount",
    "Direction": "Direction",
    "Merchant": "Merchant",
    "Description": "Description",
    "Channel": "Channel",
    "Owner": "Owner",
    "L1": "L1",
    "L2": "L2",
    "IsFixed": "IsFixed",
    "IsNecessary": "IsNecessary",
    "Month": "Month",
    "Note": "Note",
    "ExcludeFromBudget": "ExcludeFromBudget",
}

# ----------------------------
# Helpers
# ----------------------------
def must_env(name: str, val: str):
    if not val:
        raise RuntimeError(f"Missing env var: {name}")

def safe_text(x: Any) -> str:
    return "" if pd.isna(x) else str(x).strip()

def normalize_amount(x: Any) -> float:
    """
    Input examples: '¥63.55', '￥0.56', '63.55'
    Output: float positive
    """
    if pd.isna(x):
        return 0.0
    s = str(x).strip().replace("￥", "").replace("¥", "").replace(",", "")
    s = re.sub(r"[^\d\.\-]", "", s)
    if s in ("", "-"):
        return 0.0
    return abs(float(s))

def normalize_direction(x: Any) -> str:
    s = safe_text(x)
    if "收入" in s:
        return "收入"
    if "支出" in s:
        return "支出"
    if "转" in s or "不计收支" in s:
        return "转账"
    return s or "支出"

def month_str(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y-%m")

# ----------------------------
# Feishu API
# ----------------------------
def get_tenant_access_token(app_id: str, app_secret: str) -> str:
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    r = requests.post(url, json={"app_id": app_id, "app_secret": app_secret}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != 0:
        raise RuntimeError(f"tenant_access_token error: {data}")
    return data["tenant_access_token"]

def bitable_batch_create(token: str, app_token: str, table_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_create"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"records": [{"fields": it} for it in items]}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def chunks(xs: List[Any], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

# ----------------------------
# Category model
# ----------------------------
L1_CANDIDATES = ["生存成本", "生活运营", "旅行", "升级消费", "投资", "收入", "其他"]

L2_CANDIDATES = [
    # 生存成本
    "居住-房租", "居住-水电网", "通讯订阅-手机网费", "通讯订阅-软件订阅", "保险", "宠物-基础", "宠物-医疗",
    # 生活运营
    "餐饮-外食", "餐饮-外卖咖啡", "超市-食材", "日用品", "交通-打车公交", "社交-聚会请客",
    # 旅行
    "旅行-预提", "旅行-机酒", "旅行-当地交通", "旅行-餐饮门票",
    # 升级消费
    "数码设备", "家具家电", "运动/摄影装备", "高端体验",
    # 投资 / 收入
    "定投/入金", "保险理财", "工资", "奖金", "股票/期权", "投资收益",
    # 兜底
    "其他",
]

L1_L2_MAP = {
    "生存成本": ["居住-房租", "居住-水电网", "通讯订阅-手机网费", "通讯订阅-软件订阅", "保险", "宠物-基础", "宠物-医疗"],
    "生活运营": ["餐饮-外食", "餐饮-外卖咖啡", "超市-食材", "日用品", "交通-打车公交", "社交-聚会请客"],
    "旅行":     ["旅行-预提", "旅行-机酒", "旅行-当地交通", "旅行-餐饮门票"],
    "升级消费": ["数码设备", "家具家电", "运动/摄影装备", "高端体验"],
    "投资":     ["定投/入金", "保险理财"],
    "收入":     ["工资", "奖金", "股票/期权", "投资收益"],
    "其他":     ["其他"],
}

_llm_client: Optional[AzureOpenAI] = None

def get_llm_client() -> AzureOpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    return _llm_client

def classify_transaction(merchant: str, desc: str, direction: str, tx_type: str = "") -> Dict[str, Any]:
    """Use LLM to classify a single transaction row into Feishu fields."""

    default = {
        "L1": "其他",
        "L2": "其他",
        "IsFixed": "非固定",
        "IsNecessary": "非必要",
        "ExcludeFromBudget": direction == "转账",
    }

    if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT):
        return default

    l1_l2_desc = "\n".join(f"  {l1}: {', '.join(l2s)}" for l1, l2s in L1_L2_MAP.items())

    prompt = f"""你是家庭记账分类器。根据下面一条支付宝交易记录，输出 JSON 填充飞书多维表格的字段。

--- 交易信息 ---
交易类型: {tx_type}
交易对方: {merchant}
商品/描述: {desc}
收支方向: {direction}

--- 需要你填写的字段 ---
1. L1 (一级分类): 必须是以下之一: {", ".join(L1_CANDIDATES)}
2. L2 (二级分类): 必须是以下之一，且与 L1 对应:
{l1_l2_desc}
3. IsFixed (是否固定支出): "固定" 或 "非固定"
   - 固定: 每月必然发生且金额相对稳定 (房租、水电、订阅、保险等)
   - 非固定: 偶发或金额波动大
4. IsNecessary (是否必要支出): "必要" 或 "非必要"
   - 必要: 维持基本生活必须 (居住、基本餐饮、交通通勤、保险等)
   - 非必要: 可削减的消费 (外卖咖啡、社交聚会、高端体验等)
5. ExcludeFromBudget (是否排除预算): true 或 false
   - true: 转账、退款、内部划转等不计入收支预算
   - false: 正常收支

--- 要求 ---
- 只输出合法 JSON，不要输出任何解释
- JSON 格式: {{"L1": "...", "L2": "...", "IsFixed": "...", "IsNecessary": "...", "ExcludeFromBudget": true/false}}
"""
    try:
        client = get_llm_client()
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a JSON-only responder. Output valid JSON and nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        result = json.loads(raw)

        # Validate and fallback for each field
        if result.get("L2") not in L2_CANDIDATES:
            result["L2"] = "其他"
        if result.get("L1") not in L1_CANDIDATES:
            result["L1"] = "其他"
        if result.get("IsFixed") not in ("固定", "非固定"):
            result["IsFixed"] = "非固定"
        if result.get("IsNecessary") not in ("必要", "非必要"):
            result["IsNecessary"] = "非必要"
        if not isinstance(result.get("ExcludeFromBudget"), bool):
            result["ExcludeFromBudget"] = direction == "转账"

        return result
    except Exception:
        traceback.print_exc()

    return default

# ----------------------------
# Parse Alipay exported files
# ----------------------------
# Alipay CSV column mapping (Chinese column names to internal names)
ALIPAY_MAP = {
    "交易创建时间": "Date",
    "付款时间": "PayTime",
    "最近修改时间": "ModifyTime",
    "交易来源地": "Source",
    "类型": "TxType",
    "交易对方": "Merchant",
    "商品名称": "Description",
    "金额（元）": "Amount",
    "收/支": "Direction",
    "交易状态": "Status",
    "服务费（元）": "ServiceFee",
    "成功退款（元）": "Refund",
    "备注": "Note",
    "资金状态": "FundStatus",
}

def load_file(path: str) -> pd.DataFrame:
    """Load Alipay CSV file with proper encoding and header detection."""

    # Try different encodings
    for encoding in ["gbk", "gb2312", "utf-8", "utf-8-sig"]:
        try:
            # First, read the file to find the header row
            with open(path, "r", encoding=encoding) as f:
                lines = f.readlines()

            # Find the header row (contains column names like "交易号" or "交易创建时间")
            header_row = None
            for idx, line in enumerate(lines):
                if "交易创建时间" in line or "交易号" in line:
                    header_row = idx
                    break

            if header_row is None:
                # Try to find by looking for common patterns
                for idx, line in enumerate(lines):
                    if "收/支" in line and "金额" in line:
                        header_row = idx
                        break

            if header_row is None:
                raise RuntimeError("Could not find Alipay CSV header row")

            # Read CSV starting from the header row
            df = pd.read_csv(
                path,
                encoding=encoding,
                skiprows=header_row,
                dtype=str,
                on_bad_lines='skip'
            )

            # Clean column names (remove whitespace, tabs)
            df.columns = [col.strip().replace('\t', '') for col in df.columns]

            # Filter out summary rows at the end (rows starting with ---)
            df = df[~df.iloc[:, 0].astype(str).str.startswith('---')]
            df = df[~df.iloc[:, 0].astype(str).str.contains('笔记录', na=False)]

            return df

        except (UnicodeDecodeError, UnicodeError):
            continue

    raise RuntimeError(f"Could not read file {path} with any known encoding")

def prepare_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Debug: show what columns we found
    print(f"Columns in file: {df.columns.tolist()}")

    # Keep only known columns (handle variations in column names)
    col_mapping = {}
    for csv_col in df.columns:
        clean_col = csv_col.strip()
        # Handle variations in column names
        if "交易创建时间" in clean_col or "创建时间" in clean_col:
            col_mapping[csv_col] = "Date"
        elif clean_col == "类型":
            col_mapping[csv_col] = "TxType"
        elif "交易对方" in clean_col:
            col_mapping[csv_col] = "Merchant"
        elif "商品名称" in clean_col or "商品" in clean_col:
            col_mapping[csv_col] = "Description"
        elif "收/支" in clean_col:
            col_mapping[csv_col] = "Direction"
        elif "金额" in clean_col and "服务费" not in clean_col and "退款" not in clean_col:
            col_mapping[csv_col] = "Amount"
        elif "备注" in clean_col:
            col_mapping[csv_col] = "Note"
        elif "交易状态" in clean_col:
            col_mapping[csv_col] = "Status"

    print(f"Column mapping: {col_mapping}")

    if not col_mapping:
        raise RuntimeError(f"No matching columns found. Available: {df.columns.tolist()}")

    # Select and rename columns
    cols_to_keep = [c for c in col_mapping.keys() if c in df.columns]
    df = df[cols_to_keep].copy()
    df.rename(columns=col_mapping, inplace=True)
    print(f"Columns after rename: {df.columns.tolist()}")

    # Ensure required columns exist
    for col in ["Date", "Amount", "Direction", "Merchant"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column: {col}")

    # Add missing optional columns
    if "Description" not in df.columns:
        df["Description"] = ""
    if "Note" not in df.columns:
        df["Note"] = ""
    if "TxType" not in df.columns:
        df["TxType"] = ""
    if "Status" not in df.columns:
        df["Status"] = ""

    # Normalize
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = df["Amount"].apply(normalize_amount)
    df["Direction"] = df["Direction"].apply(normalize_direction)
    df["Merchant"] = df["Merchant"].apply(safe_text)
    df["Description"] = df["Description"].apply(safe_text)
    df["Note"] = df["Note"].apply(safe_text)
    df["TxType"] = df["TxType"].apply(safe_text)
    df["Status"] = df["Status"].apply(safe_text)

    # Filter out transactions with 0 amount (often pre-auth or cancelled)
    df = df[df["Amount"] > 0]

    # Filter out refund success / cancelled transactions if needed
    # df = df[~df["Status"].str.contains("退款成功|交易关闭", na=False)]

    total = len(df)
    out: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(df.iterrows()):
        if pd.isna(row["Date"]):
            continue

        dt: pd.Timestamp = row["Date"]
        direction = row["Direction"]
        amount = float(row["Amount"])
        merchant = row["Merchant"]
        desc = row["Description"]
        tx_type = row["TxType"]

        # LLM classifies all judgment fields for this row
        print(f"[{i+1}/{total}] Classifying: {merchant} | {desc} | {direction}")
        classification = classify_transaction(merchant, desc, direction, tx_type=tx_type)
        print(f"  -> {classification}")

        fields = {
            FIELDS["Date"]: int(dt.timestamp() * 1000),
            FIELDS["Amount"]: amount,
            FIELDS["Direction"]: direction,
            FIELDS["Merchant"]: merchant,
            FIELDS["Description"]: desc,
            FIELDS["Channel"]: DEFAULT_CHANNEL,
            FIELDS["Owner"]: DEFAULT_OWNER,
            FIELDS["L1"]: classification["L1"],
            FIELDS["L2"]: classification["L2"],
            FIELDS["IsFixed"]: classification["IsFixed"],
            FIELDS["IsNecessary"]: classification["IsNecessary"],
            FIELDS["Month"]: month_str(dt),
            FIELDS["Note"]: row["Note"],
            FIELDS["ExcludeFromBudget"]: classification["ExcludeFromBudget"],
        }

        out.append(fields)

    return out

# ----------------------------
# Main
# ----------------------------
def main():
    must_env("FEISHU_APP_ID", FEISHU_APP_ID)
    must_env("FEISHU_APP_SECRET", FEISHU_APP_SECRET)
    must_env("BITABLE_APP_TOKEN", BITABLE_APP_TOKEN)
    must_env("BITABLE_TABLE_ID", BITABLE_TABLE_ID)

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="Alipay CSV exported file path")
    args = p.parse_args()

    df = load_file(args.file)
    records = prepare_records(df)

    print(f"Prepared {len(records)} records.")
    print(json.dumps(records[:3], ensure_ascii=False, indent=2))

    if DRY_RUN:
        print("DRY_RUN=true: not writing to Feishu.")
        return

    token = get_tenant_access_token(FEISHU_APP_ID, FEISHU_APP_SECRET)

    for batch in chunks(records, BATCH_SIZE):
        resp = bitable_batch_create(token, BITABLE_APP_TOKEN, BITABLE_TABLE_ID, batch)
        if resp.get("code") != 0:
            raise RuntimeError(f"Batch create failed: {resp}")
        print(f"Inserted {len(batch)} records.")
        time.sleep(0.2)

    print("Done.")

if __name__ == "__main__":
    main()
