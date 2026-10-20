# 盲读案例的对照答案

| 案例 | 职位 | 抽样层 | A | B | C | D | GPT-4o 参考 | Claude 盲评（决定 / 匹配分 1–5） |
|---|---|---|---|---|---|---|---|---|
| 01 | Python Developer | SFT hires, GRPO pilot rejects | RFT（reject） | DPO r2em-100（hire） | GRPO pilot（reject） | SFT（hire） | hire | hire / 4 |
| 02 | Data Scientist | all four agree: reject | RFT（reject） | SFT（reject） | DPO r2em-100（reject） | GRPO pilot（reject） | reject | 未评 |
| 03 | DevOps engineer | SFT hires, GRPO pilot rejects | SFT（hire） | DPO r2em-100（hire） | RFT（reject） | GRPO pilot（reject） | hire | hire / 4 |
| 04 | Python Developer | SFT rejects, DPO r2em-100 hires | DPO r2em-100（hire） | RFT（reject） | GRPO pilot（reject） | SFT（reject） | reject | hire / 3 |
| 05 | Python Developer | SFT rejects, DPO r2em-100 hires | RFT（reject） | DPO r2em-100（hire） | SFT（reject） | GRPO pilot（reject） | hire | hire / 3 |
| 06 | Manual QA Engineer | SFT hires, GRPO pilot rejects | GRPO pilot（reject） | DPO r2em-100（hire） | RFT（hire） | SFT（hire） | hire | hire / 5 |
| 07 | Manual QA Engineer | SFT hires, GRPO pilot rejects | RFT（reject） | GRPO pilot（reject） | DPO r2em-100（hire） | SFT（hire） | hire | hire / 4 |
| 08 | Product manager | all four agree: hire | RFT（hire） | GRPO pilot（hire） | DPO r2em-100（hire） | SFT（hire） | hire | 未评 |
| 09 | Graphic Designer | SFT hires, GRPO pilot rejects | GRPO pilot（reject） | SFT（hire） | RFT（hire） | DPO r2em-100（hire） | hire | hire / 5 |
| 10 | Automation QA Engineer | all four agree: reject | DPO r2em-100（reject） | GRPO pilot（reject） | SFT（reject） | RFT（reject） | reject | 未评 |

pair_id 与抽样细节见 `key.json`；生成脚本 `scripts/blind_cases.py`，种子 7。