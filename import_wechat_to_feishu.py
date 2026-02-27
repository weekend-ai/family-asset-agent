# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "requests",
#     "openpyxl",
#     "openai",
#     "python-dotenv",
# ]
# ///

import os
import re
import time
import json
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
DRY_RUN = os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes", "y")

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
    # Feishu formula field Month exists, but we also fill it for convenience.
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

def l1_from_l2(l2: str) -> str:
    if l2 in {"居住-房租","居住-水电网","通讯订阅-手机网费","通讯订阅-软件订阅","保险","宠物-基础","宠物-医疗"}:
        return "生存成本"
    if l2 in {"餐饮-外食","餐饮-外卖咖啡","超市-食材","日用品","交通-打车公交","社交-聚会请客"}:
        return "生活运营"
    if l2.startswith("旅行-"):
        return "旅行"
    if l2 in {"数码设备","家具家电","运动/摄影装备","高端体验"}:
        return "升级消费"
    if l2 in {"定投/入金","保险理财"}:
        return "投资"
    if l2 in {"工资","奖金","股票/期权","投资收益"}:
        return "收入"
    return "其他"

def classify_l2(merchant: str, desc: str, direction: str, tx_type: str = "") -> str:

    prompt = f"""你是家庭记账分类器。根据"交易对方/商品描述/交易类型"把交易归入二级分类 L2。
候选 L2：
{", ".join(L2_CANDIDATES)}

交易类型：{tx_type}
交易对方：{merchant}
商品/描述：{desc}
收支方向：{direction}

要求：
- 只输出一个 L2，必须来自候选列表
- 不要输出解释
"""
    try:
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "Return only one label from the candidate list."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        label = (resp.choices[0].message.content or "").strip()
        if label in L2_CANDIDATES:
            return label
    except Exception:
        pass

    return "其他"

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
    return pd.read_excel(path, dtype=str)

def prepare_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Keep only known columns
    cols = [c for c in WECHAT_MAP.keys() if c in df.columns]
    df = df[cols].copy()
    df.rename(columns={k: v for k, v in WECHAT_MAP.items()}, inplace=True)

    # Normalize
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = df["Amount"].apply(normalize_amount)
    df["Direction"] = df["Direction"].apply(normalize_direction)
    df["Merchant"] = df["Merchant"].apply(safe_text)
    df["Description"] = df.get("Description", "").apply(safe_text)
    df["Channel"] = df.get("Channel", DEFAULT_CHANNEL).apply(safe_text)
    df["Note"] = df.get("Note", "").apply(safe_text)
    df["TxType"] = df.get("TxType", "").apply(safe_text)

    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        if pd.isna(row["Date"]):
            continue

        dt: pd.Timestamp = row["Date"]
        direction = row["Direction"]
        amount = float(row["Amount"])
        merchant = row["Merchant"]
        desc = row["Description"]
        tx_type = row["TxType"]

        # Guess L2/L1
        l2 = classify_l2(merchant, desc, direction, tx_type=tx_type)

        # If it's a transfer, exclude by default (you can change this policy)
        exclude = True if direction == "转账" else False

        fields = {
            FIELDS["Date"]: dt.strftime("%Y-%m-%d %H:%M:%S"),
            FIELDS["Amount"]: amount,
            FIELDS["Direction"]: direction,
            FIELDS["Merchant"]: merchant,
            FIELDS["Description"]: desc,
            FIELDS["Channel"]: row["Channel"] or DEFAULT_CHANNEL,
            FIELDS["Owner"]: DEFAULT_OWNER,
            FIELDS["L2"]: l2,
            FIELDS["Note"]: row["Note"],
            FIELDS["ExcludeFromBudget"]: exclude,
        }

        # Optional: fill these if your Feishu columns are single-select and already have options
        # If not set up yet, better leave them blank to avoid write errors.
        # fields[FIELDS["IsFixed"]] = "固定"
        # fields[FIELDS["IsNecessary"]] = "必要"

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
    print(json.dumps(records[:5], ensure_ascii=False, indent=2))

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