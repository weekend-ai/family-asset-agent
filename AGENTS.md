# 家庭财政项目协作说明

## 本地上下文与隐私

- 处理家庭账单或财务分析前，读取 `private/finance-context.md` 和 `private/config.json`（若存在）。前者保存用户已确认口径、确认日期和适用月份；后者保存飞书资源标识、使用人与姓名映射。沿用已确认口径，不反复追问。
- `private/` 与 `artifacts/` 不提交 Git；不将私人金额、行程、收入原因、真实姓名或资源标识写入公共源码、文档或测试。
- 不读取或输出 `.env`、密钥或访问令牌。配置示例见 `config.example.json`，缺失本地配置时不要猜资源标识。
- 用户只要求查看或分析时保持只读；是否写入遵循用户本轮授权范围。不自动 Git commit。

## 每月导入工作流

- 执行 `./finance-parser prepare --owner <使用人> <文件...>`。入口为 `prepare / review / sync`，不嵌套调用其他 AI 分类。
- 用 `review <json>` 一次读取紧凑 TSV，在当前会话 review、分类，通过本地脚本按行号批量修改 JSON。汇总由脚本计算，只看摘要与疑点；不反复打印全量 JSON、payload 或同步响应，不为省 token 跳过交易 review。
- 产物放 `artifacts/YYYY-MM/`。`review --summary-only` 复核预算支出，`--pending-only` 查看尚未分类行；分类选项见 README。
- 写前确认真实字段及对应月份/使用人的已导入记录；`sync <json> --dry-run` 仅生成本地预览，`sync <json>` 真实写入，写后核对数量和预算支出。
- 保留 `*-sync-receipt.jsonl`。中断后先核对云端，只补缺失记录，不删回执盲目重跑。不擅自修改汇总表或私人账户。

## 飞书读取与统计

- 阅读当前环境 `lark-base` 技能及其要求的查询分析参考；认证/权限问题按 `lark-shared` 处理。使用 `lark-cli`，用户资源显式 `--as user`。
- 先 `+table-list` / `+field-list` 确认真正结构，再读取汇总表，核对最新有实际数据的月份。资源参数从本地配置获取。
- 分类、按月汇总等问题先读取交易字段，再用 `+data-query` 云端聚合；不要只读默认第一页就推断全局。原始记录按需投影，`has_more=true` 时继续分页，除非明确仅取样例。
- `+record-list --format json` 的 `data.fields`、`data.data`、`data.record_id_list` 按同一返回的列顺序映射，不依赖历史列顺序。
- 预算支出为 `Direction=支出` 且 `ExcludeFromBudget=false` 的 Amount 之和（对应 `ExpenseForDashboard`），尊重排除项与退款净额。家庭结余、累计余额等业务解释以本地口径为准。
- 报告注明数据截至月份与统计口径；余额、数量等动态事实重新读取，不引用旧快照冒充当前结果。
- 沙箱无法访问钥匙串时，按环境规则申请在沙箱外重试同一只读命令；不重新初始化、不降级钥匙串、不擅自更换身份。
