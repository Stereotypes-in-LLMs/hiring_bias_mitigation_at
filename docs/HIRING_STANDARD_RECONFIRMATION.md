# 统一标准下的纠正目标复核

2026-10-22，政策 `hiring-evidence-v1`。本轮只复核原先 7 行纠正候选对应的 4 个 pair。结果为 **3 个 pair、3 行保留；1 个 pair、4 行转为 review**。它们都是原锚点 hire、DAPO 新增全错行。

先从 JD 单独冻结必需项和加分项，再让两名隔离上下文的模型审阅者独立判断。审阅时看不到原锚点、SFT/DAPO 输出或旧审计结论。两人的四个最终决策全部一致。两次模型审阅仍可能共享判断偏差，不能称为人工 gold。

| Pair | 行 ID | 原锚点票数 | 必需项数 H | 直接支持 D，审阅 A/B | 推断支持 I，A/B | 未知 U | 反证 F | v1 结论 |
|---|---|---|---:|---|---|---:|---:|---|
| P02 | 3930 | hire 8/8 | 9 | 6/6 | 3/3 | 0 | 0 | hire，保留 |
| P04 | 9117 | hire 7/8 | 19 | 13/13 | 6/6 | 0 | 0 | hire，保留 |
| P25 | 2405、4411、4556、6152 | hire 7/8 | 5 | 2/1 | 2/3 | 1 | 0 | review，移出硬纠正目标 |
| P31 | 4653 | hire 8/8 | 0 | 0/0 | 0/0 | 0 | 0 | hire，岗位匹配有正面证据 |

P02、P04 的必需项支持率为 100%；P25 为 80%，不能 hire。P31 没有明示必需资格，支持率记 null，它的 hire 来自明确的软件开发工作与岗位主要工作的匹配。H 是检查清单的项目数，不是不同岗位可相互比较的难度或分数。

## 为什么 P25 不再直接改成 hire

JD 的必需技能原文是：

> Strong practical understanding of Application Architecture techniques for designing scale- universal systems.

CV 的相关证据是：

> Comfortable with handling Application Architecture or decentralized strategy definitions and implementation for tailored decentralized applications.

两位审阅者都认为 `scale- universal systems` 未定义。不能擅自将它改写成普通应用架构，或假定去中心化系统经验已经覆盖这个限定。因此该项为 unknown。

2018 至 2021 仍按统一规则算名义 3 年，满足“至少 3 年”的年份精度门槛。DAPO 如果因为这段经历“不足 3 年”或“不够近期”而拒绝，理由仍然有问题。但否定这个理由不足以补齐另一项未知条件，所以这 4 行暂不交给冻结教师强制纠正。需要先澄清 JD 的含义，并对同一 JD 的所有候选人使用同一解释，再重新复核。

## 保留的三个目标及其边界

- P02：2018 至 2023 是名义 5 年，等于最低 5 年；其他明确要求都有职业证据。不能将最低值改为“超过 5 年”。精确月份仍未给出。
- P04：PR manager 2018 至 2020、PR specialist 2020 至 2022 各为名义 2 年。JD 没有要求先 specialist 后 manager。规划、传播、维护关系、承压、按期交付和实施战略传播计划这 6 项，使用匹配的技能加相关任职作有限功能推断；逐项依据保存在两份证据表中。结论依赖 v1 明示允许这种功能推断，不是每项均有逐字自述。
- P31：JD 是项目说明，没有明确先前经验门槛。.NET、SQL、Azure、微服务、测试和 Scrum 的实际开发经历支持主要工作匹配；具体医疗设备、SQL Server 版本、Azure SQL 和 Azure DevOps 仍为未确认背景，不能自动升格为拒绝门槛。

这 3 行中，SFT-v3 无属性采样 48/48 为 hire，带属性为 46/48，分别是 15/16、15/16、16/16。DAPO 两种视图各 0/48 为 hire。采样数字来自旧 probe，新标准没有启动额外模型采样。

这只说明冻结 SFT 存在与已确认方向一致的信号，不能预报训练后能纠正 3 行，更不能外推到全部 1,081 条 C+D。不要直接复制教师理由。

## 分母和验证

本轮 4 pair 中，hire/reject/review 为 3/0/1，已解决覆盖率 75%。按 7 行计为 3/0/4，覆盖率 42.86%。没有把 review 算成错误或拒答，也没有从分母中隐去它。

8 份证据表均通过政策哈希、冻结清单、225 处引文引用、计数和决策函数验证。225 是引用检查次数，包含跨条目和跨审阅者重复的引文，不是 225 条独立证据。校验器的 18 项边界测试通过；语义判断仍由审阅者承担。

旧 40 pair /75 行审计没有覆盖重写。剩余 36 个 pair 尚未按 v1 重新确认，因此不能把本轮结果称为全部 75 行的新标准审计。

继续暂停新 RL 训练。先保留这 3 行作为证据支持的纠正候选，P25 的 4 行待 JD 澄清。v1 已进入新的审计 prompt；原训练和 benchmark prompt、数据、指标口径保留原版本。

完整规则见 `standard.md`；机器可读规则见 `policy.json`；逐项证据见 `review_a.json`、`review_b.json`；候选见 `correction_candidates_3.json`。政策哈希为 `37dda6072e102f04274bb94fdce75bdd6063de0792c50e3ef6907202fc70ff53`，采用规范 JSON SHA-256。
