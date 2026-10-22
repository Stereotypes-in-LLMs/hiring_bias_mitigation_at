# 75 行纠正方案与当前使用入口

当前采用 `hiring-evidence-v1`，统一标准和两阶段 prompt 见 [标准](HIRING_DECISION_STANDARD.md)。本次更新把标准落实到输入校验、分类复算和纠正数据导出，分类依据仍是已完成的 80 份盲审记录。

## Setup

问题是冻结 SFT 教师在这 75 行上有多少受审阅证据支持的纠正方向。范围固定为 40 个 pair、75 行。它们来自已有 paired probe：SFT 无属性 16/16 与旧锚点一致，DAPO 两种输入均 0/16 与旧锚点一致。对照是原始采样与 v1 的两份独立文本审阅，没有新增模型生成。

期望：若确为 DAPO 退化，审阅应支持 SFT 方向；若审阅支持 DAPO 或保留 review，则不能据旧锚点强迫 DAPO 跟随教师。导出遇到版本、引文、清单、计数或已保存结果不一致时停止。本步骤不选择模型、不改数据划分、不训练。

## Result

| 分类 | Pair / 行 | 当前动作 |
|---|---:|---|
| DAPO 判断退化 | 3 / 3 | 3930、4653、9117，仅保留 hire 标签纠正候选 |
| 锚点有问题 | 9 / 13 | 进入锚点复核队列，不用旧教师方向纠正 |
| 信息不足或审阅分歧 | 28 / 59 | 进入证据复核队列，不设二元硬目标 |

三条目标的 SFT 无属性合计 48/48 hire，带属性 46/48 hire；DAPO 两种输入各 0/48 hire。原七条候选中的 P25 四行 2405、4411、4556、6152 被排除。P06、P25、P39 的最终审阅结论分歧继续保留。

数据来源为 `teacher75_v1_full/summary.json`、`audit_rows_75.json` 和 `teacher_correction_candidates.json`，完整逐 pair 证据见 [审计报告](TEACHER75_STANDARD_AUDIT.md)。新入口对原始证据重新计算并核对上述结果。

## Reading

冻结 SFT 教师在本子集上有证据支持的纠正候选为 3/75。其余 72 行不进入硬纠正目标。13 行需要复核锚点，59 行需要补必需项证据或解决审阅分歧，不能把未知改成 reject。

当前安排是继续暂停新 RL，先处理证据和锚点。3 条候选可用于以后确认的教师纠正实验，但不足以证明大规模训练会改善 TPR 或偏见。这 75 行是特选 C+D 子集，不能用它估计全部 9,140 行中 A/B 与 C+D 的比例。

所有新行标注 `training_ready=false`。候选的 `target_reason` 来自本轮审阅，原 SFT rationale 未被批准复用。后续若准备训练文本，需要明确选定监督格式、逐条核查理由，确认数据用途与划分，并为新三元任务建立相同 prompt 下的父模型基线。

## Caveats

审阅来自隔离上下文中的模型，不能当作人工金标。校验器验证引文存在、字段、哈希、计数和决策函数，不能证明自然语言推断正确或 JD 清单完整。

新 prompt 的变化是强制两阶段输入和三元输出约定，v1 判断规则及政策哈希保持不变。40 份待调用的新 prompt 已准备，`request_executed=false`；已有 80 份审阅没有被换成新 prompt 版本。已有 DAPO/SFT 采样来自旧 prompt，没有作为新任务定义下的模型结果报告。

## Reproducing

唯一的本次准备配置为 `configs/teacher75_correction_workflow_v1.json`。它指向已完成的 `teacher75_v1_full` 来源，固定 40 pair / 75 行、政策哈希和 15 个来源文件的 SHA-256。来源文件改变时，旧审阅不能直接授权新输入的目标，需要重新确认来源及审阅。源目录中的历史文件保留原貌。

```bash
uv run python scripts/prepare_teacher_corrections.py \
  --config configs/teacher75_correction_workflow_v1.json \
  --output /path/to/a/new/teacher75_v1_application
```

输出包括全部 75 行、3 行候选、锚点复核队列、证据复核队列、每行资格判定及 40 个未调用的审阅请求。`manifest.json` 记录源文件、脚本与产物 SHA-256、政策和 prompt 版本。只接受尚不存在的输出目录；不调用模型、不提交训练任务。
