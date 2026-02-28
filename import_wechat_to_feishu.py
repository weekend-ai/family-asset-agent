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
# DRY_RUN = os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes", "y")
DRY_RUN = False

DEFAULT_OWNER = os.getenv("DEFAULT_OWNER", "共同")
DEFAULT_CHANNEL = os.getenv("DEFAULT_CHANNEL", "微信")

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
    if "转" in s:
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

    prompt = f"""你是家庭记账分类器。根据下面一条微信交易记录，输出 JSON 填充飞书多维表格的字段。

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
# Parse WeChat exported files
# ----------------------------
WECHAT_MAP = {
    "交易时间": "Date",
    "交易类型": "TxType",
    "交易对方": "Merchant",
    "商品": "Description",
    "收/支": "Direction",
    "金额(元)": "Amount",
    "支付方式": "Channel",
    "备注": "Note",
}

def load_file(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        try:
            return pd.read_csv(path, dtype=str, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, dtype=str, encoding="gbk")

    # WeChat export files have metadata rows at the top
    # Find the row containing "微信支付账单明细列表"
    df_raw = pd.read_excel(path, header=None)
    header_row = None

    for idx, row in df_raw.iterrows():
        if row.astype(str).str.contains("微信支付账单明细列表", na=False).any():
            # The actual header is one row below this marker
            header_row = idx + 1
            break

    if header_row is None:
        raise RuntimeError(
            "Could not find WeChat export header marker. "
            "Is this a valid WeChat export file?"
        )

    # Re-read with the correct header row
    return pd.read_excel(path, dtype=str, skiprows=header_row, header=0)

def prepare_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Debug: show what columns we found
    print(f"Columns in file: {df.columns.tolist()}")

    # Keep only known columns
    cols = [c for c in WECHAT_MAP.keys() if c in df.columns]
    print(f"Matched columns: {cols}")

    if not cols:
        raise RuntimeError(f"No matching columns found. Expected: {list(WECHAT_MAP.keys())}")

    df = df[cols].copy()
    df.rename(columns={k: v for k, v in WECHAT_MAP.items()}, inplace=True)
    print(f"Columns after rename: {df.columns.tolist()}")

    # Normalize
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = df["Amount"].apply(normalize_amount)
    df["Direction"] = df["Direction"].apply(normalize_direction)
    df["Merchant"] = df["Merchant"].apply(safe_text)
    df["Description"] = df.get("Description", "").apply(safe_text)
    df["Channel"] = df.get("Channel", DEFAULT_CHANNEL).apply(safe_text)
    df["Note"] = df.get("Note", "").apply(safe_text)
    df["TxType"] = df.get("TxType", "").apply(safe_text)

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
            FIELDS["Channel"]: row["Channel"] or DEFAULT_CHANNEL,
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
    p.add_argument("--file", required=True, help="WeChat CSV/XLSX exported file path")
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
