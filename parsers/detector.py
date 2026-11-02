import os
import re


def _clean_filename(basename: str) -> str:
    """Strip OpenClaw UUID suffix and get clean filename."""
    return re.sub(r'---[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(\..*)?$', r'\1', basename)


def detect_source(filepath: str) -> str | None:
    """Detect file source from filename pattern. Returns: 'alipay_csv', 'alipay_xlsx', 'wechat', 'credit_card', or None."""
    clean = _clean_filename(os.path.basename(filepath))

    if re.match(r'支付宝(?:交易明细|账单)', clean):
        if clean.endswith('.csv'):
            return 'alipay_csv'
        if clean.endswith('.xlsx'):
            return 'alipay_xlsx'
    if re.match(r'微信支付账单', clean):
        return 'wechat'
    if re.match(r'信用卡账单明细', clean):
        return 'credit_card'
    return None
