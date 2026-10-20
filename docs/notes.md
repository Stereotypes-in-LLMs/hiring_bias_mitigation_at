 prepare data: prepare pairs,attribute_free prompt + rows(different attribtes included),attributed prompt
 def mentions_attribute 有局限，英语几乎不变格，截前缀几乎换不来召回，只会带来误报。-> fixed （换成了整词匹配只认词表里的词，同义改述还是漏掉，比如 “your faith”、"served in the army"。语义层面的暗示靠 judge 那一步来抓。）
 v1 锚点偏严，是因为所有 prompt 共用了 seed 42，改用 8 次投票多数当锚点，建了 SFT-v3 数据。v3 还补充了Djinni 新候选人，RJDB train，kaggle数据(整个候选池有 186,970 对，本轮只生成其中 9,034 对。) -> sft_v3_swap, hiring_bias_mitigation_at-v3additional/docs/V3_NEW_DATA.md
 新的 DPO data source:训练 SFT-v3 → 用 SFT-v3 回答属性变体 → 找出被属性影响的实际错误 → 教师核验 → 构建偏好对, + 旧 r2em
 new eval table： 带判定的汇总表是 reports/reference_metrics.csv，逐对比较的长表是 reports/comparisons.csv
 GPU usage