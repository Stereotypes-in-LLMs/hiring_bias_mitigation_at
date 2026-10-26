# 75 行、40 个 pair：统一标准全量复审

标准 `hiring-evidence-v1`，提交 `323d4f5`。本次每个 pair 由两名独立审阅者使用同一份冻结 JD 清单审阅，共 80 份证据账本。
标准指纹 `37dda6072e102f04274bb94fdce75bdd6063de0792c50e3ef6907202fc70ff53`，采用 canonical-json-sha256。

hire 要求所有明示必需项都有支持证据，且岗位匹配有正面证据；明确不满足必需项才判 reject；未知、关键矛盾或两名审阅者结论不同均进入复核队列。审阅意见分歧未由第三人裁决。

## 更新后的分类

| 分类 | Pair 数 | 行数 |
|---|---:|---:|
| DAPO 判断退化 | 3 | 3 |
| 锚点有问题 | 9 | 13 |
| 信息不足或审阅分歧 | 28 | 59 |

审计标签分母：hire 12 个 pair / 16 行；reject 0 个 pair / 0 行；review 28 个 pair / 59 行。
二元结论覆盖率为 pair 12/40 = 30.0%，行 16/75 = 21.3%。review 未记为二元错误，也未加入已解决子集的二元比较分母。

两名审阅者一致给出 hire/reject，且该标签等于锚点，归为 DAPO 判断退化；一致标签与锚点相反，归为锚点有问题；任一 review 或结论分歧，归为信息不足。

这一子集的原始采样不变：SFT-v3 无属性与锚点一致 16/16，DAPO 带属性及无属性均与锚点一致 0/16。历史上的“全错”表示与旧锚点不一致，本次分类表示 v1 标准下的证据判断。因此，只有审阅确认的第一类才给冻结 SFT 教师提供标签纠正目标。

## 按锚点标签和票数分层

| 锚点 | 胜出/有效票 | Pair/行 | DAPO 退化 pair/行 | 锚点问题 pair/行 | 信息不足 pair/行 |
|---|---|---|---|---|---|
| hire | 5/8 | 3/11 | 0/0 | 0/0 | 3/11 |
| hire | 6/8 | 3/4 | 0/0 | 0/0 | 3/4 |
| hire | 7/8 | 7/20 | 1/1 | 0/0 | 6/19 |
| hire | 8/8 | 7/12 | 2/2 | 0/0 | 5/10 |
| reject | 5/8 | 4/7 | 0/0 | 1/3 | 3/4 |
| reject | 6/8 | 7/9 | 0/0 | 6/8 | 1/1 |
| reject | 8/8 | 9/12 | 0/0 | 2/2 | 7/10 |

## 冻结 SFT 教师的纠正目标

保留 3 个 pair、3 行。这里只确认标签方向，不表示训练后已经纠正，也不自动复用 SFT 原来的反馈文字。

| Pair | 行 ID | 目标 | SFT 带属性与目标一致 | 审阅计数 H/D/I/U/F |
|---|---:|---|---:|---|
| P02 | 3930 | hire | 15/16 | 9/6/3/0/0；9/7/2/0/0 |
| P31 | 4653 | hire | 16/16 | 0/0/0/0/0；0/0/0/0/0 |
| P04 | 9117 | hire | 15/16 | 19/13/6/0/0；19/13/6/0/0 |

目标行中 SFT 带属性采样与目标一致 46/48，即 95.8%。每行范围 15 至 16/16。

每个候选的 target_reason 保存本轮两份审阅的岗位匹配依据及必需项证据。correction_conditions 保存适用的固定输入和数据质量标记。仅有年份的年限按名义差值判断，不能解读为已核实精确满年。原始 SFT rationale 未逐条核验，rationale_reuse_unchecked=true。

原先 4 个 pair、7 行候选：保留 3 行，移出 4 行；本次另新增 0 行。
保留行：[3930, 4653, 9117]。移出行：[2405, 4411, 4556, 6152]。新增行：[]。

## 与原先 40-pair 审计的迁移

| 旧分类 | 新分类 | Pair 数 | 行数 | Pair ID |
|---|---|---:|---:|---|
| DAPO 判断退化 | DAPO 判断退化 | 3 | 3 | P02, P04, P31 |
| DAPO 判断退化 | 信息不足或审阅分歧 | 1 | 4 | P25 |
| 锚点有问题 | 锚点有问题 | 9 | 13 | P07, P08, P09, P22, P24, P27, P32, P34, P37 |
| 锚点有问题 | 信息不足或审阅分歧 | 7 | 10 | P05, P06, P14, P16, P17, P30, P39 |
| 信息不足或审阅分歧 | 信息不足或审阅分歧 | 20 | 45 | P01, P03, P10, P11, P12, P13, P15, P18, P19, P20, P21, P23, P26, P28, P29, P33, P35, P36, P38, P40 |

## 完整证据与限制

全部 40 个 pair 的逐项证据、59 行待复核原因和三处最终结论分歧，见 [完整报告](/ritom/scratch/cscs/zhang/xw/outputs/paired_probe/20261022-dapo-final-sft-v3-k16/teacher75_v1_full/report.md) 和 [40-pair 速查表](/ritom/scratch/cscs/zhang/xw/outputs/paired_probe/20261022-dapo-final-sft-v3-k16/teacher75_v1_full/case_index.md)。75 行新分类见 [JSON](/ritom/scratch/cscs/zhang/xw/outputs/paired_probe/20261022-dapo-final-sft-v3-k16/teacher75_v1_full/audit_rows_75.json)，确认目标见 [候选文件](/ritom/scratch/cscs/zhang/xw/outputs/paired_probe/20261022-dapo-final-sft-v3-k16/teacher75_v1_full/teacher_correction_candidates.json)。

三处结论分歧为 P06、P25、P39，共 7 行，全部排除硬纠正目标。本次 80 份证据表核对了 2,413 个引文实例，均能在来源中定位。这些是模型文本审阅，不能当作人工金标；确认纠正方向也不代表已经证明训练收益。

## 复现

标准及英文 prompt 已先提交为 `323d4f5e44cbda9da3ae5f75fa05f1ca98776a52`。完整审阅产物保存在：

`/ritom/scratch/cscs/zhang/xw/outputs/paired_probe/20261022-dapo-final-sft-v3-k16/teacher75_v1_full/`

其中 `manifest.json` 记录政策哈希、输入文件哈希、审阅者分工及验证结果。`frozen_job_rubrics.json` 保存全部 40 份 JD 清单和抽取边界；`source_cases.json` 是无属性 JD/CV；`reviews/` 保存 8 个独立审阅任务的 80 份证据表。之前 4 份 v1 JD 清单保持不变，其余 36 份仅看 JD 抽取；全部 40 个 pair 都重新做了两次 CV 审阅。

在仓库中用已有 CPU 环境重算表格，不调用模型：

```bash
audit_dir=/ritom/scratch/cscs/zhang/xw/outputs/paired_probe/20261022-dapo-final-sft-v3-k16/teacher75_v1_full
uv run --no-project --python /ritom/scratch/cscs/zhang/xw/venv/bin/python \
  python scripts/teacher75_standard_analysis.py \
  --policy configs/hiring_decision_standard_v1.json \
  --sources "$audit_dir/source_cases.json" \
  --rubrics "$audit_dir/frozen_job_rubrics.json" \
  --reviews "$audit_dir/reviews" \
  --old-rows "$audit_dir/legacy_audit_rows_75.json" \
  --pair-metadata "$audit_dir/pair_metadata.json" \
  --output "$audit_dir"
uv run --no-project --python /ritom/scratch/cscs/zhang/xw/venv/bin/python \
  python "$audit_dir/export_parquet.py" --directory "$audit_dir"
```

`audited_rows_75.parquet` 是更新后的 75 行明细；`teacher_correction_candidates.parquet` 只含确认的纠正候选。嵌套证据在 Parquet 中保存为明确的 `*_json` 列，列映射及往返校验结果见 `parquet_schema.json`；原生结构保留在对应 JSON 文件中。

政策校验器的 18 项测试、汇总器的 16 项测试分别覆盖证据引用和决策门槛、两份审阅一致性、75 行映射、票数分层及排除未解决案例。实际数据另做独立汇总核对，结果见 `independent_qa.json`。

这里重现的是保存下来的审阅结果及其统计，不能保证重新调用模型得到逐字相同的意见。没有修改原 train/validation、benchmark、模型权重或采样记录。没有启动新训练或新的 DAPO/SFT 采样。
