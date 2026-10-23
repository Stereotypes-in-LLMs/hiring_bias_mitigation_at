# Djinni 监控修正，第 1 次冻结

2026-10-23，实验 `monitor-v2-djinni-intake-20261023-v1`。
配置 `configs/calibration/monitor_v2_djinni_freeze.json` 是本轮规则入口。
本提交冻结来源、候选人顺序和分区、统一标注程序、目标权重、8 个模型及候选告警规则。
它没有生成新标签，没有运行新模型评分，也没有启用训练。

## 现有监控怎么计算，哪里需要修

`mitigation/grpo.py::MarginMonitorCallback` 在更新前读取 SFT-v3 margin，每 25 步读取当前
服务模型的 margin。margin 是强制 JSON 决策前缀后的 `log p(hire) - log p(reject)`，
`margin > 0` 算 hire。8 档边界固定为 -4.595、-2.197、-0.847、0、0.847、2.197、4.595。
这些边界不是经过校准的录用概率。

旧估计在每档混合所有来源和标签，计算整体录用率变化，再分别乘 benchmark 的 hire/reject
档权重。696 行包含 504 行 Djinni 和 192 行 RJDB。同档 RJDB 的新增录用抵消 Djinni 的流失，
DAPO 因而投影 -3.24 pp，benchmark margin 实测 -9.61 pp。旧校准没有通过，不能用于放行训练。

本轮保留结论：**RJDB 的 reject 锚点可疑**，待按统一二分类标注协议盲审确认。

另外，旧实现把空档当作零变化，恢复时只核对 parent margin 数量，未核对 prompt、标签、
权重和模型指纹。新版需要对这些条件停止并保存，不能继续输出可信的 TPR 估计。
本次是数据和程序冻结；运行时 callback 的修复、部署和独立验证仍在第二次冻结之后。

## 来源与候选人

固定来源为 `additional/pool/pairs_en.parquet` 中的 `djinni_new`，SHA256
`cb64431723a73f6a1fe2107503a0a80d09a3883da2c0b319582b7cd3f2df2a90`。
英文候选 HF revision 为 `86255174c6c378a2b52cd7e81d79002c64d899a6`，
岗位 revision 为 `b56a6054c10f1266a141e37c8df66c79ff2863af`。

144,091 名 Djinni 候选扣除注册模型训练、历史 train/validation、既往投票整批、benchmark
候选和岗位、本轮审阅，以及现有属性标记筛查后，剩 127,681 人。来源清单保留 121 个输入文件
的指纹和排除原因。规范化 CV 去重规则为 NFKC、空白折叠、casefold 后 SHA256。
近重复筛查的方法已冻结，尚待执行；不能把目前的 ID 隔离描述为完全无文本暴露。

首批按 `SHA256("monitor-v2-intake-20261023:42:" + candidate_id)` 升序取 2,000 人，
保留原始 CV/JD 配对。每名候选最多一对，不按旧锚点或模型预测挑正例。
标注后不足覆盖时，按同一顺序每批补 2,000 人，最多 10,000 人；任何补样都必须发生在揭示
训练 arm 结果之前。每批全部标注，保留所有已解决 hire/reject，不能只留下有利样本。

候选分区在标注之前固定。使用独立 salt `monitor-v2-split-20261023:42:`，SHA256 整数小于
`2**255` 为 monitor，其余为 validation。首批分为 1,003 和 997 人。同人所有文本和属性变体
不能跨区。两区共享 14 个岗位，另有 258 名候选对应的岗位在注册模型训练中出现过。
因此要报告岗位聚类敏感性，不能声称岗位独立。

旧 696 行、75 行/40 pair 审阅、原 167 人子池和所有已看过的 arm 结果均属于开发证据。
原 167 人中的 8 个旧 Qwen hire 不算新增确认正例。
部分历史训练输入缺训练时 SHA；当前路径与行数匹配，并已锁定当前字节，但不能补造历史不可变证明。

## 统一二分类标注

复用 `hiring-evidence-v1` 唯一政策文件和两个已提交 prompt，不另写一套资格门槛。
先只看 JD 冻结清单，再由两位独立审阅者看无属性 CV。审阅者不能看到旧标签、票数、SFT/DAPO
答案、margin 或最终分区。审阅者身份和服务版本要在开始前登记。

- hire：每项明确必需条件都有直接或允许的功能证据，未知数 U=0、反证数 F=0，且至少一处
  CV 原文支持主要岗位工作。H>0 时支持率必须为 100%。
- reject：没有决定性材料冲突，且至少一项明确必需条件有肯定反证，F>=1。
- 缺失、矛盾或意见不一致：`label=null, eligibility=unresolved`，不进入二分类标签集。
  它是样本资格状态，不是模型的第三个输出类别，也不能自动改成 reject。

审阅证据账本保留原政策的 review 状态。只有两份有效账本一致给出 hire/reject，才导出对应
二元标签。所有表格必须同时公开已解决比例和未解决数量，不能只报告留下的样本。

必需项与加分项分开；加分项缺失不能拒绝，也不能抵消必需项失败。未提到不等于不会。
年限按原政策处理：年份差、不首尾都计入、月份用月差、重叠区间取并集、最低门槛用 >=。
开放日期使用记录本身的 as-of 日期；2026-10-20 的缓存下载时间不等于简历观察日期。
没有有效截止日期时，不能把至今经历累计到今天。

已有 benchmark 仍保留 GPT-4o 标签。新协议标签和旧标签不能仅因都是二元就视为同一真值。
预先固定的标签对照覆盖原 450 个 pair：按同一标准盲审，报告旧标签×新标签×父 margin 档的
一致性、未解决率及两个口径的父模型 TPR。原 benchmark 数值不被覆盖。标签含义差异未解决时，
不能宣布新监控测量的就是原 TPR；这是放行条件，不是后续调权重的理由。

## 分层、权重与模型

每个 pair 使用原 benchmark 完整的 68 个带属性变体；无属性控制另行报告。
按新鲜 SFT-v3 margin 分入原 8 档，不能以旧 k8/k32 采样录用率替代。
每档继续按标签、protected group、condition 分开估计。对每个标签侧：

`delta = sum(W_cell * (gain_cell - loss_cell))`

W 是旧 GPT-4o benchmark 中该标签下各档×group×condition 的精确频数比例，已保存到
`monitor_v2_target_weights.json`，仅使用父模型 margin，没有使用任何训练 arm 的表现选权重。
完整数值包含两个标签侧各 48 格，附 68 个变体清单。它定义旧 benchmark 的投影目标，
本身不能证明新标签可向这个目标迁移。

新标签、最终行顺序和父 margin 尚未产出，因此每行 `W_cell/n_cell` 权重尚未冻结。
它们必须在第二次冻结中锁定，然后才允许揭示训练 arm 分数。
空的正权重格不能填零或把其他权重重新归一化。

最低支持筛查为每个有目标权重的标签×margin 档至少 20 名不同候选人、所有正权重细格有观测。
还须在揭示 arm 前用父模型格归属预演全部 40 个固定 bootstrap seed；任何重采样出现正权重
空格，就按固定顺序补下一批。这个结构预检不需要 arm 分数。
真实 delta 和迁移误差的候选聚类 95% 区间半宽须不超过 2 pp。这些是筛查线，不是准确性保证。
区间精度在揭示 arm 后检验，不能据此再补样并声称原验证仍独立。

模型名单固定为 SFT-v3、DAPO-v3-all、GRPO-v3-all、GRPO-v3-all-step525、GRPO-v3、
DPO-v3-step240、DPO-v3-onpolicy-step98、DPO-v3-4880-step305。
模型锁记录完整权重 SHA256、tokenizer/config 和 merge 来源；不能仅凭可变目录名判断模型身份。
step525 与 GRPO final 属于同一训练轨迹，不能当作两个独立实验。

## 告警及验证

每 25 个 optimizer step 检查，结束时不足 25 步也检查一次。使用相对 SFT-v3 的净 TPR 变化。
点估计下降达到 8 pp，或者保守的下降上界达到 8 pp，就保存并停训。
写成带符号变化，即 `point_delta <= -0.08 OR lower_delta <= -0.08`。
实测下降小于 8 pp 才属于误停检查中的安全侧；恰好 8 pp 停训符合规则。

候选告警区间方法固定为候选人整块 bootstrap，65,536 次，PCG64，基础 seed 20261023，
每次 seed 由 `[base_seed, check_index]` 构造，quantile method 为 linear。最多 40 次更新后
检查，每次单侧 alpha 为 0.05/40。固定目标权重，每次重采样重新计算格内均值。
任一重采样产生正权重空格、评分/来源/服务同步失败、检查预算耗尽，都按证据不足保存停训。
恢复训练须验证完整父模型、prompt 顺序、标签、权重、配置指纹及累计检查次数。

这是待验证的近似 bootstrap 规则。Bonferroni 的覆盖结论以每次区间有效为前提，不能现在
宣称全程 95% 保证。它也不覆盖标签差异、来源迁移偏差或两次检查之间的变化。

独立验证仍要求 7 个 arm 全部通过：原 benchmark margin TPR 绝对误差 <=2 pp，
相对完整生成 TPR 的乐观偏差 <2 pp，实测下降超过 8 pp 的 arm 必须告警。
还要按上述实际保守规则检查误停；误停阻止常规部署，不能当场调阈值后再称验证通过。
同时报告新 validation bank、岗位聚类敏感性、FPR、loss/gain 和标签覆盖。

## 产物与复现

远端产物目录为 `$HBM_OUTPUT_ROOT/outputs/monitor_v2_freeze/20261023-intake-v1/`。
`source/` 保存来源、完整候选库存、首批原文及模型锁；`lock-final/` 保存候选分区、盲审队列和 manifest。
`lock/` 是提交前审阅草稿，保留追溯，不作正式入口。
盲审队列尚不能直接调用：必须先完成近重复筛查，并把 JD-only 清单冻结后再绑定 CV。

```bash
uv run python scripts/prepare_monitor_v2_freeze.py \
  --config configs/calibration/monitor_v2_djinni_freeze.json \
  --output "$HBM_OUTPUT_ROOT/outputs/monitor_v2_freeze/20261023-intake-v1/lock-final"
uv run python scripts/prepare_monitor_v2_freeze.py \
  --config configs/calibration/monitor_v2_djinni_freeze.json \
  --output "$HBM_OUTPUT_ROOT/outputs/monitor_v2_freeze/20261023-intake-v1/lock-final" --verify
```

入口校验来源指纹、排除集、固定顺序、文本指纹、分区和目标权重，拒绝覆盖现有冻结目录。
`--verify --require-stage arm-scoring` 或 `training` 在第一锁必定退出 2。
这不是对通用训练脚本的全局封锁；本轮不会提交训练任务。

第一锁已固定程序和输入。新增确认正例仍为 0。第二锁缺新标签、父 margin、最终行权重和
标签对照结论，随后还须完成运行时修复与独立验证，不能把第一锁等同于监控已经可用。
