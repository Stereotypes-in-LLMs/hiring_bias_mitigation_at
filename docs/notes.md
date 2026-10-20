 prepare data: prepare pairs,attribute_free prompt + rows(different attribtes included),attributed prompt
 def mentions_attribute 有局限，英语几乎不变格，截前缀几乎换不来召回，只会带来误报。-> fixed （换成了整词匹配只认词表里的词，同义改述还是漏掉，比如 “your faith”、"served in the army"。语义层面的暗示靠 judge 那一步来抓。）
 v1 锚点偏严，是因为所有 prompt 共用了 seed 42，改用 8 次投票多数当锚点，建了 SFT-v3 数据