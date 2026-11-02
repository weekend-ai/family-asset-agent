import json
import subprocess
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
from openpyxl import Workbook

import parser
import configuration
from parsers.base import Transaction, parse_amount, transactions_to_json


def transaction(**changes):
    fields = dict(date="2026-10-01", amount=10.5, direction="支出", merchant="测试商店",
                  description="测试商品", channel="alipay", owner="成员A", l2="日用品")
    fields.update(changes)
    return Transaction(**fields)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.workspace = self.runner.isolated_filesystem()
        self.workspace.__enter__()
        Path("private").mkdir()
        Path("private/config.json").write_text(json.dumps({
            "base_token": "test-base", "tables": {"transactions": "test-table"},
            "owners": ["成员A", "成员B", "共同"], "owner_aliases": {"测试姓名": "成员A"},
        }))
        self.config_patch = patch.object(configuration, "CONFIG_PATH", Path("private/config.json"))
        self.config_patch.start()

    def tearDown(self):
        self.config_patch.stop()
        self.workspace.__exit__(None, None, None)

    def write_transactions(self, transactions):
        Path("reviewed.json").write_text(transactions_to_json(transactions), encoding="utf-8")

    def bill(self):
        csv = Path("支付宝交易明细.csv")
        csv.write_text("姓名：测试用户\n交易时间,交易分类,交易对方,对方账号,商品说明,收/支,金额,收/付款方式,交易状态,交易订单号,商家订单号,备注\n2026-10-01 10:00:00,日用,测试商店,账号,测试商品,支出,10.50,信用卡,交易成功,1,2,\n", encoding="utf-8")
        return str(csv)

    def test_prepare_no_ai_and_no_overwrite(self):
        bill = self.bill()
        with patch.object(parser.subprocess, "run") as external:
            result = self.runner.invoke(parser.cli, ["prepare", "--owner", "成员B", bill])
            self.assertEqual(result.exit_code, 0, result.output)
            external.assert_not_called()
        target = Path("artifacts/2026-10/成员B.json")
        data = json.loads(target.read_text())
        self.assertEqual(data[0]["owner"], "成员B")
        self.assertEqual(data[0]["l2"], "")
        original = target.read_bytes()
        result = self.runner.invoke(parser.cli, ["prepare", "--owner", "成员B", bill])
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(target.read_bytes(), original)

    def test_prepare_does_not_write_partial_bills(self):
        bill = self.bill()
        Path("unknown.csv").write_text("bad input")
        result = self.runner.invoke(parser.cli, ["prepare", "--owner", "成员A", bill, "unknown.csv", "-o", "out.json"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertFalse(Path("out.json").exists())

    def test_xlsx_prepare(self):
        wb = Workbook()
        ws = wb.active
        ws.append(["微信支付账单"])
        ws.append(["账单信息"])
        ws.append(["交易时间", "交易类型", "交易对方", "商品", "收/支", "金额(元)", "支付方式", "备注"])
        ws.append(["2026-10-01 10:00:00", "消费", "测试商店", "测试商品", "支出", 8.5, "零钱", "/"])
        wb.save("微信支付账单.xlsx")
        wb.close()
        result = self.runner.invoke(parser.cli, ["prepare", "--owner", "成员A", "微信支付账单.xlsx"])
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(Path("artifacts/2026-10/成员A.json").read_text())
        self.assertEqual(data[0]["amount"], 8.5)

    def test_dedup_preserves_refunds_owners_and_counts(self):
        backed = transaction(payment_method="信用卡")
        cc = transaction(channel="credit_card")
        refund = transaction(channel="credit_card", direction="收入")
        other_owner = transaction(channel="credit_card", owner="成员B")
        result, dropped = parser._dedup_cross_channel([backed, cc, cc, refund, other_owner])
        self.assertEqual(dropped, 1)
        self.assertEqual(len(result), 4)
        self.assertIn(refund, result)
        self.assertIn(other_owner, result)

    def test_review_net_expense_and_pending_ids(self):
        self.write_transactions([transaction(), transaction(amount=-2.5, l2=""),
                                 transaction(amount=99, exclude_from_budget=True),
                                 transaction(amount=50, direction="收入")])
        summary = self.runner.invoke(parser.cli, ["review", "reviewed.json", "--summary-only"])
        self.assertEqual(summary.exit_code, 0, summary.output)
        self.assertIn("budget expense 8.00", summary.output)
        pending = self.runner.invoke(parser.cli, ["review", "reviewed.json", "--pending-only"])
        self.assertIn("\n1\t2026-10-01", pending.output)
        self.assertNotIn("\n0\t2026-10-01", pending.output)

    def test_sync_dry_run_is_offline(self):
        self.write_transactions([transaction(amount=-2.5)])
        with patch.object(parser.subprocess, "run") as external:
            result = self.runner.invoke(parser.cli, ["sync", "reviewed.json", "--dry-run"])
            self.assertEqual(result.exit_code, 0, result.output)
            external.assert_not_called()
        self.assertTrue(Path("reviewed-sync-payload.json").exists())
        self.assertFalse(Path("reviewed-sync-receipt.jsonl").exists())

    def test_sync_rejects_unclassified_or_invalid_values_before_network(self):
        for changes in ({"l2": ""}, {"amount": float("nan")}, {"owner": "unknown"},
                        {"exclude_from_budget": "false"}, {"date": "bad"}):
            self.write_transactions([transaction(**changes)])
            with patch.object(parser.subprocess, "run") as external:
                result = self.runner.invoke(parser.cli, ["sync", "reviewed.json"])
                self.assertNotEqual(result.exit_code, 0)
                external.assert_not_called()
            self.assertFalse(Path("reviewed-sync-receipt.jsonl").exists())

    def test_success_receipt_blocks_repeat_sync(self):
        self.write_transactions([transaction(), transaction(amount=20)])
        response = {"ok": True, "data": {"record_id_list": ["rec_test"]}}
        with patch.object(parser, "_call_lark_record_batch_create", return_value=response) as create:
            result = self.runner.invoke(parser.cli, ["sync", "reviewed.json", "--batch-size", "1"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(create.call_count, 2)
            repeat = self.runner.invoke(parser.cli, ["sync", "reviewed.json"])
            self.assertNotEqual(repeat.exit_code, 0)
            self.assertEqual(create.call_count, 2)
        receipt = [json.loads(line) for line in Path("reviewed-sync-receipt.jsonl").read_text().splitlines()]
        self.assertEqual([row["offset"] for row in receipt[1:]], [0, 1])

    def test_partial_sync_retains_confirmed_batch_and_blocks_retry(self):
        self.write_transactions([transaction(), transaction(amount=20)])
        with patch.object(parser, "_call_lark_record_batch_create", side_effect=[{"ok": True}, RuntimeError("timeout")]) as create:
            result = self.runner.invoke(parser.cli, ["sync", "reviewed.json", "--batch-size", "1"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("1 confirmed records", result.output)
            repeat = self.runner.invoke(parser.cli, ["sync", "reviewed.json"])
            self.assertNotEqual(repeat.exit_code, 0)
            self.assertEqual(create.call_count, 2)
        self.assertEqual(len(Path("reviewed-sync-receipt.jsonl").read_text().splitlines()), 2)

    def test_lark_payload_identity_and_cleanup(self):
        captured = {}
        def fake_run(command, **kwargs):
            self.assertEqual(command[command.index("--as") + 1], "user")
            path = Path(command[command.index("--json") + 1][1:])
            captured["payload"] = json.loads(path.read_text())
            return subprocess.CompletedProcess(command, 0, '{"ok":true}', '')
        with patch.object(parser.subprocess, "run", side_effect=fake_run):
            parser._call_lark_record_batch_create("base", "table", [{"Note": "", "Amount": -2}])
        self.assertEqual(captured["payload"], {"fields": ["Note", "Amount"], "rows": [[None, -2]]})
        self.assertEqual(list(Path('.').glob('.lark-batch-*')), [])

    def test_invalid_amount_does_not_become_zero(self):
        with self.assertRaises(ValueError):
            parse_amount("invalid")

    def test_owner_aliases_are_loaded_from_private_config(self):
        self.write_transactions([transaction(owner="测试姓名")])
        self.assertEqual(parser._load_from_json("reviewed.json")[0].owner, "成员A")

    def test_sync_config_and_override_precedence(self):
        self.write_transactions([transaction()])
        result = self.runner.invoke(parser.cli, ["sync", "reviewed.json", "--dry-run"])
        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(Path("reviewed-sync-payload.json").read_text())
        self.assertEqual((payload["base_token"], payload["table_id"]), ("test-base", "test-table"))
        result = self.runner.invoke(parser.cli, ["sync", "reviewed.json", "--dry-run", "--base-token", "cli-base"],
                                    env={"FEISHU_APP_TOKEN": "env-base", "FEISHU_TABLE_ID": "env-table"})
        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(Path("reviewed-sync-payload.json").read_text())
        self.assertEqual((payload["base_token"], payload["table_id"]), ("cli-base", "env-table"))

    def test_missing_config_does_not_use_a_personal_default(self):
        with patch.object(configuration, "CONFIG_PATH", Path("missing-config.json")):
            self.assertEqual(configuration.load_config(), {})
            self.write_transactions([transaction()])
            with patch.object(parser.subprocess, "run") as external:
                result = self.runner.invoke(parser.cli, ["sync", "reviewed.json"])
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("private/config.json", result.output)
                external.assert_not_called()

    def test_invalid_config_stops_without_echoing_private_contents(self):
        Path("private/config.json").write_text('{"owners": "private-invalid-value"}')
        result = self.runner.invoke(parser.cli, ["prepare", "--owner", "成员A", self.bill()])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid private/config.json", result.output)
        self.assertNotIn("private-invalid-value", result.output)

    def test_legacy_json_still_loads(self):
        legacy = asdict(transaction(monthly="2026-10"))
        Path("legacy.json").write_text(json.dumps([legacy]))
        self.assertEqual(parser._load_from_json("legacy.json")[0].monthly, "2026-10")


if __name__ == "__main__":
    unittest.main()
