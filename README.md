# family-finance

每月提供支付宝 / 微信 / 信用卡账单，由当前会话 review 后写入飞书。
本地脚本负责解析、去重、汇总和批量写入，不再额外调用 Kimi/Codex 分类。

## 使用

依赖统一用 `uv sync` 安装。首次使用将 `config.example.json` 复制到 `private/config.json`，
填写 Base/table 标识、使用人与姓名映射；私人业务口径放在 `private/finance-context.md`。
`private/` 不提交 Git，需要单独进行私人备份。入口为 `./finance-parser`（或 `uv run python parser.py`）。

```bash
# 1. 解析：不调用 AI，不写飞书，默认保存到 artifacts/<月份>/<使用人>.json
./finance-parser prepare --owner 成员A '/path/支付宝交易明细.csv' '/path/微信支付账单.xlsx'

# 2. 紧凑明细：供当前会话一次 review、分类，id 对应 JSON 的零基数组下标
./finance-parser review artifacts/2026-10/成员A.json

# 修改本地 JSON 后复核汇总；也可用 --pending-only 仅显示尚未分类的行
./finance-parser review artifacts/2026-10/成员A.json --summary-only

# 3. 本地预览：验证并保存 payload，不联网
./finance-parser sync artifacts/2026-10/成员A.json --dry-run

# 已 review 且用户授权写入后执行
./finance-parser sync artifacts/2026-10/成员A.json
```

`prepare --owner` 必填，使用本地配置 `owners` 中的名称，统一应用到本批账单。
支持支付宝 CSV/XLSX、微信 XLSX、信用卡 XLS，按原始导出文件名识别。
传入明确的文件列表；不要把同一账单的 CSV 和 XLSX 两个版本一起传入。
跨渠道去重匹配使用人、日期、金额和收支方向，保留支付宝/微信的信用卡支付记录。
同渠道重复或重叠导出仍需 review；跨渠道仅凭日期金额匹配也需留意同日同额消费。
任一文件解析失败或为空就停止；输出已存在也停止，可用 `-o` 指定新路径。

## Review 口径

当前会话读取紧凑 TSV，在本地 JSON 批量填写 `l2`、`is_fixed`、`is_necessary`，
处理退款、排除项和疑似重复。金额和原始交易信息保持可追溯，疑点按需查看原始文件。
类别：房租、水电气电话费、在家吃饭、外出吃饭、日用品、宠物、交通出行、锻炼/运动、
日常活动、旅游、医疗健康、人情/礼物、杂项/意外、其他。
固定性：固定 / 可变 / 非固定；必要性：必要 / 可压缩 / 非必要。

预算支出 = `direction=支出` 且 `exclude_from_budget=false` 的金额之和，保留退款净额。
汇总在本地计算，不把完整 JSON、payload、响应反复打印给模型。
`month` / `monthly` 暂保留以兼容历史 JSON；飞书月份由表内公式计算。

## 飞书同步

使用已登录的 `lark-cli`，显式 `--as user`。默认资源读取项目目录的 `private/config.json`，不在源码中硬编码。
`--base-token` / `--table-id`（或 `FEISHU_APP_TOKEN` / `FEISHU_TABLE_ID`）只适用于同结构交易表，
不能直接换成汇总表。参数优先级为命令行 > 环境变量 > 本地配置。`--batch-size` 范围 1–200，默认 200。

写前由当前会话按 `lark-base` 技能检查真实字段及对应月份/使用人的已导入记录，
写后核对条数与预算支出。查询结果及同步响应存文件，向模型只输出摘要。
脚本为追加写入，不提供云端自动去重，也不修改汇总表或私人账户。

每批响应保存到同目录 `*-sync-receipt.jsonl`；该文件存在时拒绝再次同步同一文件。
中途失败、超时或中断，先结合回执与飞书查明哪些批次已写入，只同步缺失记录；
不要删回执后直接重跑。改文件名仍可能重复导入，必须检查云端记录。

## 目录与检查

- `private/`：私人业务口径与资源配置，不提交。
- `AGENTS.md`：通用协作规则，可以提交。
- `configuration.py` / `config.example.json`：配置加载与通用示例。
- `parser.py`：prepare / review / sync。
- `parsers/`：渠道解析器与统一交易结构。
- `artifacts/<月份>/`：本地账单、review、同步记录；历史文件已归档，默认不提交。
- `tests/`：离线回归测试，运行 `.venv/bin/python -m unittest discover -s tests`。

旧 `parse` / `parse-dir` / `combine` / `classify` 命令已收敛到上述流程。
