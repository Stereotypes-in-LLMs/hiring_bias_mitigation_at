"""How much learning signal each reference class still gave GRPO, step by step.

GRPO learns only from groups whose completions disagree. The pilot logged the share of
zero-variance groups for the whole step (TRL's `frac_reward_zero_std`) and, through the
decision reward, the sampled hire rate per reference class, but not the zero-variance share
per class, and no completions. This script recovers the per-class picture at every saved
checkpoint, without sampling.

Every completion opens with the forced prefix `{"decision": "` (99.98% of the SFT student's
temperature-1 samples, all greedy audit outputs), so the decision is the first token after it.
With p the probability of the decision token that matches the reference, at the temperature
the trainer sampled at (1.0), a group of k = 8 completions:

    is right on average          p
    has zero reward variance     p^k + q^k            (q: the wrong decision's probability)
    carries signal               1 - p^k - q^k

The format reward (+0.2, met by >= 99.9% of samples) is treated as constant. Mass on tokens
other than the two decision tokens is reported, not modelled.

Two weights per checkpoint:
    served   the merged bf16 weights vLLM sampled the rollouts from (mitigation/grpo.py pushes
             exactly these), so their numbers are what the trainer's groups saw
    trained  HF + PEFT, the policy the loss was computed on

Prompts: the run's training prompts (grpo-02, 1,500 per class) and its 256 held-out monitor
prompts. Each checkpoint is scored on every prompt, so a value at step t covers prompts already
used and not yet used; the per-step trainer log is the in-run counterpart.

    # GPU, one process per GPU (scripts/cscs/grpo_signal.sbatch)
    python scripts/grpo_signal_by_reference.py score --shard 0 --num-shards 4
    # CPU: tables, checks and figures
    python scripts/grpo_signal_by_reference.py report --log logs/xw-grpo-5066266.out
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hiring_bias_mitigation.utils.config import REPO_ROOT, resolve_output_path  # noqa: E402

RUN_DIR = "outputs/grpo/qwen3.5-9b_en_only_grpo_pilot"
DATA_DIR = "artifacts/semisynthetic-v2/grpo-02"
OUT_DIR = "outputs/grpo_signal/qwen3.5-9b_en_only_grpo_pilot"
REPORT_DIR = "reports/grpo_signal"
GROUP = 8                  # num_generations in the pilot config
PROMPTS_PER_STEP = 16      # 128 completions per optimizer step / 8
CLASSES = ("hire", "reject")
WHICH = ("served", "trained")


# ---- pure helpers (tested) ----------------------------------------------------------------

def checkpoint_steps(run_dir: Path) -> list[int]:
    """0 (SFT, the zero adapter) and every finished checkpoint-N, in order."""
    steps = [int(m.group(1)) for d in run_dir.iterdir()
             if (m := re.fullmatch(r"checkpoint-(\d+)", d.name))
             and (d / "adapter_model.safetensors").is_file()]
    return [0, *sorted(steps)]


def shard_steps(steps: list[int], shard: int, num_shards: int) -> list[int]:
    """Steps for one GPU, round-robin; step 0 needs one pass, the others two."""
    return [s for i, s in enumerate(steps) if i % num_shards == shard]


def group_stats(p_right: np.ndarray, p_wrong: np.ndarray, k: int = GROUP) -> dict:
    """Over a set of prompts: expected accuracy, the share of k-groups that come out all right,
    all wrong (together: zero variance, no gradient), and how many prompts carry signal."""
    p, q = np.asarray(p_right, dtype=float), np.asarray(p_wrong, dtype=float)
    if not len(p):
        return {"prompts": 0}
    right, wrong = p ** k, q ** k
    return {"prompts": len(p), "accuracy": float(p.mean()), "all_right": float(right.mean()),
            "all_wrong": float(wrong.mean()), "zero_variance": float((right + wrong).mean()),
            "signal_prompts": float((1 - right - wrong).sum()),
            "decision_mass": float((p + q).mean())}


def right_wrong(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities of the reference's decision token and of the other one."""
    hire, reject = np.exp(frame["lp_hire"].to_numpy()), np.exp(frame["lp_reject"].to_numpy())
    is_hire = (frame["reference_decision"] == "hire").to_numpy()
    return np.where(is_hire, hire, reject), np.where(is_hire, reject, hire)


def parse_trainer_log(text: str) -> pd.DataFrame:
    """The per-step metric dicts TRL prints (logging_steps 1), one row per optimizer step."""
    rows = []
    for m in re.finditer(r"\{'loss': [^{}]*'epoch': '[^']*'\}", text):
        rows.append({k: float(v) for k, v in ast.literal_eval(m.group(0)).items()})
    frame = pd.DataFrame(rows)
    frame.insert(0, "step", np.arange(1, len(frame) + 1))
    return frame


def per_step_series(log: pd.DataFrame) -> pd.DataFrame:
    """What the trainer itself logged, per step and per class where it logged one.

    Per-class accuracy comes from `sampled_hire_rate_ref_*` (mitigation/grpo.py): TRL averages
    the per-rank shares with equal weight (`_pending_metrics`), not per completion. Zero-variance
    groups were logged for the whole step only.
    """
    out = pd.DataFrame({"step": log["step"]})
    out["accuracy_hire_logged"] = log["sampled_hire_rate_ref_hire"]
    out["accuracy_reject_logged"] = 1 - log["sampled_hire_rate_ref_reject"]
    out["accuracy_all_logged"] = log["rewards/decision_reward/mean"]
    out["zero_variance_all_logged"] = log["frac_reward_zero_std"]
    out["signal_groups_all_logged"] = PROMPTS_PER_STEP * (1 - log["frac_reward_zero_std"])
    return out


def code_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                              text=True, check=True).stdout
    try:
        return {"commit": git("rev-parse", "HEAD").strip(),
                "uncommitted": [line[3:] for line in git("status", "--porcelain").splitlines()
                                if line.strip()]}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted": None}


# ---- score (GPU) --------------------------------------------------------------------------

def load_prompts(data_dir: Path) -> pd.DataFrame:
    keep = ["prompt", "reference_decision", "pair_id", "protected_group", "condition", "tipped"]
    parts = []
    for split, name in (("train", "grpo_train.parquet"), ("validation", "grpo_validation.parquet")):
        frame = pd.read_parquet(data_dir / name, columns=keep, use_threads=False)
        parts.append(frame.assign(split=split, row=np.arange(len(frame))))
    return pd.concat(parts, ignore_index=True)


def decision_logprobs(model, tokenizer, prompts: list[str], template_kwargs: dict | None,
                      batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """log p(hire token), log p(reject token) after the forced prefix, at temperature 1."""
    import torch

    from hiring_bias_mitigation.eval.backends import _chat_wrap
    from hiring_bias_mitigation.eval.margins import (
        PREFIX,
        WORDS,
        _last_position_logits,
        decision_token_ids,
    )

    hire_id, reject_id = decision_token_ids(tokenizer, WORDS["en"])
    texts = [_chat_wrap(tokenizer, p, template_kwargs) + PREFIX for p in prompts]
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    lp_hire, lp_reject = np.empty(len(texts)), np.empty(len(texts))
    side, tokenizer.padding_side = tokenizer.padding_side, "left"
    model.eval()
    try:
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            enc = tokenizer([texts[i] for i in rows], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            with torch.no_grad():
                logp = torch.log_softmax(_last_position_logits(model, enc).float(), dim=-1)
            lp_hire[rows] = logp[:, hire_id].cpu().numpy()
            lp_reject[rows] = logp[:, reject_id].cpu().numpy()
    finally:
        tokenizer.padding_side = side
    return lp_hire, lp_reject


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def cmd_score(args) -> None:
    import torch
    from peft import PeftModel
    from peft.utils import load_peft_weights, set_peft_model_state_dict

    from hiring_bias_mitigation.mitigation import training_common as C
    from hiring_bias_mitigation.mitigation.grpo import merged_into_base
    from hiring_bias_mitigation.utils.config import load_config

    run_dir = Path(resolve_output_path(args.run_dir))
    out = Path(resolve_output_path(args.out))
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run_dir / "run_config.json").read_text())
    train_cfg = load_config(REPO_ROOT / manifest["config_path"])
    template_kwargs = train_cfg.get("chat_template_kwargs")
    steps = shard_steps(checkpoint_steps(run_dir), args.shard, args.num_shards)
    todo = [(s, w) for s in steps for w in (("served",) if s == 0 else WHICH)
            if not (out / f"step{s:03d}--{w}.parquet").exists()]
    print(f"shard {args.shard}: steps {steps}; to score {todo}", flush=True)
    if not todo:
        return
    prompts = load_prompts(Path(resolve_output_path(args.data)))
    model, tokenizer = C.load_model_and_tokenizer(
        {"base_model": manifest["base_model"], "dtype": "bfloat16"}, for_training=False)
    model = model.to("cuda")
    policy = None
    for step in sorted({s for s, _ in todo}):
        started = time.monotonic()
        ckpt = run_dir / f"checkpoint-{step}"
        if step > 0:
            if policy is None:
                policy = PeftModel.from_pretrained(model, str(ckpt))
            else:
                set_peft_model_state_dict(policy, load_peft_weights(str(ckpt)))
        for which in [w for s, w in todo if s == step]:
            def score(m):
                return decision_logprobs(m, tokenizer, prompts["prompt"].tolist(),
                                         template_kwargs, args.batch_size)
            if step == 0:
                lp_h, lp_r = score(model)
            elif which == "trained":
                lp_h, lp_r = score(policy)
            else:
                with merged_into_base(policy), policy.disable_adapter():
                    lp_h, lp_r = score(policy)
            frame = prompts.drop(columns="prompt").assign(lp_hire=lp_h, lp_reject=lp_r,
                                                          step=step, which=which)
            frame.to_parquet(out / f"step{step:03d}--{which}.parquet", index=False)
            meta = {"step": step, "which": which, "checkpoint": str(ckpt) if step else None,
                    "adapter_sha256": _sha256(ckpt / "adapter_model.safetensors") if step else None,
                    "base_model": manifest["base_model"],
                    "sft_checkpoint": manifest["sft_checkpoint"],
                    "data": args.data, "prompts": len(prompts),
                    "chat_template_kwargs": template_kwargs, "temperature": 1.0,
                    "code": code_state(), "seconds": round(time.monotonic() - started, 1),
                    "torch": torch.__version__}
            (out / f"step{step:03d}--{which}.meta.json").write_text(json.dumps(meta, indent=2))
            print(f"wrote step {step} {which} ({meta['seconds']} s)", flush=True)


# ---- report (CPU) -------------------------------------------------------------------------

STEPS_TOTAL = 187     # optimizer steps of the pilot: 3,000 prompts // 16 per step
WINDOW = 25           # checkpoints every 25 steps


def training_order(n_rows: int, prompts_per_step: int = PROMPTS_PER_STEP,
                   seed: int = 42) -> np.ndarray:
    """The optimizer step (1-based) that used each training row; 0 for a row never used.

    TRL's RepeatSampler draws one `torch.randperm` of the rows from a generator seeded with the
    run's seed (42) and cuts it into chunks of `generation_batch_size // num_generations` = 16
    prompts; with steps_per_generation equal to the gradient accumulation (4), one chunk is one
    optimizer step, and an incomplete last chunk is dropped. `mapping_check` tests this order
    against the per-step rewards the trainer logged.
    """
    import torch

    perm = torch.randperm(n_rows, generator=torch.Generator().manual_seed(seed)).tolist()
    used = (n_rows // prompts_per_step) * prompts_per_step
    step = np.zeros(n_rows, dtype=int)
    for pos, row in enumerate(perm[:used]):
        step[row] = pos // prompts_per_step + 1
    return step


def all_agree_u(n_agree: int, n: int, k: int = GROUP) -> float:
    """Unbiased estimate of p^k from n stored samples of which n_agree have the outcome: the
    chance that k of them, drawn without replacement, all do (exact when n == k)."""
    from math import comb

    return comb(n_agree, k) / comb(n, k)


def zero_variance_u(n_right: int, n: int, k: int = GROUP) -> float:
    """Unbiased estimate of p^k + q^k: k samples all right or all wrong."""
    return all_agree_u(n_right, n, k) + all_agree_u(n - n_right, n, k)


def expected(frame: pd.DataFrame) -> dict:
    """`group_stats` over scored prompts (rows carrying p and q)."""
    return group_stats(frame["p"].to_numpy(), frame["q"].to_numpy())


def load_scores(score_dir: Path, n_train: int, seed: int = 42) -> dict:
    """(step, which) -> scored prompts with p, q and, for training rows, the step that used them."""
    used = training_order(n_train, seed=seed)
    scores = {}
    for path in sorted(score_dir.glob("step*--*.parquet")):
        frame = pd.read_parquet(path, use_threads=False)
        p, q = right_wrong(frame)
        rows = frame["row"].clip(upper=n_train - 1)
        frame = frame.assign(p=p, q=q,
                             used_at=np.where(frame["split"] == "train", used[rows], -1))
        scores[(int(frame["step"].iloc[0]), frame["which"].iloc[0])] = frame
    if (0, "served") in scores:      # SFT: one pass stands for both weights
        scores[(0, "trained")] = scores[(0, "served")]
    return scores


def checkpoints(scores: dict) -> list[int]:
    return sorted({s for s, w in scores if w == "served"})


def window_table(scores: dict) -> pd.DataFrame:
    """For each checkpoint t, the prompts the trainer used in steps t+1 .. t+25 (the window those
    weights, or ones close to them, sampled for), per class: what a group of 8 looked like."""
    rows = []
    for t in checkpoints(scores):
        lo, hi = t + 1, min(t + WINDOW, STEPS_TOTAL)
        if lo > STEPS_TOTAL:
            continue
        for which in WHICH:
            if (t, which) not in scores:
                continue
            frame = scores[(t, which)]
            window = frame[(frame["split"] == "train") & frame["used_at"].between(lo, hi)]
            for cls in (*CLASSES, "all"):
                part = window if cls == "all" else window[window["reference_decision"] == cls]
                stats = expected(part)
                n_steps = hi - lo + 1
                rows.append({"checkpoint": t, "steps": f"{lo}-{hi}", "x": (lo + hi) / 2,
                             "n_steps": n_steps, "which": which, "class": cls, **stats,
                             "signal_groups_per_step": stats.get("signal_prompts", 0) / n_steps,
                             "prompts_per_step": stats["prompts"] / n_steps})
    return pd.DataFrame(rows)


def per_step_expected(scores: dict, used: np.ndarray | None = None) -> pd.DataFrame:
    """Per optimizer step, the expected step-level values from the latest checkpoint at or
    before s-1 (the served weights), on the prompts that step used (or `used` instead)."""
    ckpts = checkpoints(scores)
    rows = []
    for s in range(1, STEPS_TOTAL + 1):
        c = max(t for t in ckpts if t <= s - 1)
        frame = scores[(c, "served")]
        train = frame[frame["split"] == "train"]
        if used is not None:
            train = train.assign(used_at=used[train["row"].to_numpy()])
        part = train[train["used_at"] == s]
        stats = expected(part)
        hire = part[part["reference_decision"] == "hire"]
        rows.append({"step": s, "checkpoint": c, "prompts": stats["prompts"],
                     "hire_prompts": len(hire), "accuracy_expected": stats.get("accuracy"),
                     "zero_variance_expected": stats.get("zero_variance")})
    return pd.DataFrame(rows)


def mapping_check(scores: dict, per_step: pd.DataFrame,
                  n_train: int) -> tuple[dict, pd.DataFrame]:
    """Is the reconstructed sampler order the one the run used? Per-step expected accuracy and
    zero-variance share against the trainer's logged values, and against a shuffled order."""
    def corr(exp: pd.DataFrame) -> dict:
        both = exp.merge(per_step, on="step")
        return {"accuracy_r": float(np.corrcoef(both["accuracy_expected"],
                                                both["accuracy_all_logged"])[0, 1]),
                "zero_variance_r": float(np.corrcoef(both["zero_variance_expected"],
                                                     both["zero_variance_all_logged"])[0, 1]),
                "accuracy_mean_expected": float(both["accuracy_expected"].mean()),
                "accuracy_mean_logged": float(both["accuracy_all_logged"].mean()),
                "zero_variance_mean_expected": float(both["zero_variance_expected"].mean()),
                "zero_variance_mean_logged": float(both["zero_variance_all_logged"].mean())}
    true = per_step_expected(scores)
    shuffled = training_order(n_train, seed=12345)
    return {"reconstructed_order": corr(true), "shuffled_order_seed_12345": corr(
        per_step_expected(scores, used=shuffled)),
            "prompts_per_step": sorted(true["prompts"].unique().tolist())}, true


def seen_table(scores: dict) -> pd.DataFrame:
    """At each checkpoint, per class: prompts already trained on, not yet, and held out."""
    rows = []
    for t in checkpoints(scores):
        for which in WHICH:
            if (t, which) not in scores:
                continue
            frame = scores[(t, which)]
            train = frame[frame["split"] == "train"]
            groups = {"seen": train[train["used_at"].between(1, t)],
                      "unseen": train[(train["used_at"] > t) | (train["used_at"] == 0)],
                      "held_out": frame[frame["split"] == "validation"]}
            for cls in CLASSES:
                row = {"checkpoint": t, "which": which, "class": cls}
                for name, part in groups.items():
                    stats = expected(part[part["reference_decision"] == cls])
                    row[f"{name}_prompts"] = stats["prompts"]
                    row[f"{name}_accuracy"] = stats.get("accuracy")
                    row[f"{name}_zero_variance"] = stats.get("zero_variance")
                    row[f"{name}_all_wrong"] = stats.get("all_wrong")
                rows.append(row)
    return pd.DataFrame(rows)


def step0_check(data_dir: Path, scores: dict) -> pd.DataFrame:
    """SFT's stored temperature-1 samples against the decision-token probabilities at step 0,
    per split and class. The samples are the GRPO data builder's `sft_hire_rate` (32 per hire
    row, 8 per reject row, drawn by vLLM from the merged SFT model); zero variance from them is
    the unbiased estimate `zero_variance_u` (exact for 8-sample rows).

    Only the validation split is a clean check: its rows were drawn at random within each class.
    Training rows were chosen on these same samples (tipped rows have a wrong one, and the draw
    balanced their summed pull), so their stored samples err more often than fresh draws.
    """
    tokens = scores[(0, "served")]
    rows = []
    for split, name in (("validation", "grpo_validation.parquet"), ("train", "grpo_train.parquet")):
        stored = pd.read_parquet(data_dir / name, use_threads=False)
        tok = tokens[tokens["split"] == split].set_index("row")
        for cls in CLASSES:
            part = stored[stored["reference_decision"] == cls]
            right_rate = part["sft_hire_rate"] if cls == "hire" else 1 - part["sft_hire_rate"]
            n_right = (right_rate * part["sft_samples"]).round().astype(int)
            n = part["sft_samples"]
            got = expected(tok.loc[part.index])
            rows.append({
                "split": split, "class": cls, "rows": len(part),
                "samples_per_row": "/".join(map(str, sorted(n.unique()))),
                "accuracy_samples": float(right_rate.mean()), "accuracy_tokens": got["accuracy"],
                "zero_variance_samples": float(np.mean([zero_variance_u(r, k)
                                                        for r, k in zip(n_right, n)])),
                "zero_variance_tokens": got["zero_variance"],
                "all_wrong_samples": float(np.mean([all_agree_u(k - r, k)
                                                    for r, k in zip(n_right, n)])),
                "all_wrong_tokens": got["all_wrong"]})
    return pd.DataFrame(rows)


def cmd_report(args) -> None:
    from hiring_bias_mitigation.viz.line_report import render_report

    data_dir = Path(resolve_output_path(args.data))
    n_train = len(pd.read_parquet(data_dir / "grpo_train.parquet", columns=["pair_id"],
                                  use_threads=False))
    scores = load_scores(Path(resolve_output_path(args.out)), n_train)
    report = REPO_ROOT / REPORT_DIR
    report.mkdir(parents=True, exist_ok=True)
    per_step = per_step_series(parse_trainer_log(Path(resolve_output_path(args.log)).read_text(
        errors="ignore")))
    windows = window_table(scores)
    seen = seen_table(scores)
    check0 = step0_check(data_dir, scores)
    mapping, per_step_exp = mapping_check(scores, per_step, n_train)
    for name, frame in (("per_step_logged", per_step), ("per_step_expected", per_step_exp),
                        ("windows_by_class", windows), ("seen_unseen_heldout", seen),
                        ("check_step0_samples", check0)):
        frame.to_csv(report / f"{name}.csv", index=False)
    run_dir = Path(resolve_output_path(args.run_dir))
    summary = {"group_size": GROUP, "prompts_per_step": PROMPTS_PER_STEP, "window": WINDOW,
               "log": args.log, "scores": args.out, "mapping_check": mapping, "code": code_state(),
               "missing_checkpoints": sorted(set(checkpoint_steps(run_dir))
                                             - set(checkpoints(scores)))}
    (report / "summary.json").write_text(json.dumps(summary, indent=2))
    rvm = REPO_ROOT / "reports" / "raw_vs_matched" / "table.csv"
    page = render_report(
        "GRPO 试点：按参考类别拆分的学习信号",
        ["作业 5066266：187 步，每步 16 个 prompt × 8 条回答，"
         "训练集 grpo-02（每类 1,500 个 prompt）。"
         "每个 checkpoint t 用它当时的采样权重，在训练器接下来 25 步"
         "（t+1 到 t+25）实际用到的 prompt 上，"
         "由温度 1 下决定 token 的概率算出每组 8 条回答的期望结果（作业 5070642，"
         "scripts/grpo_signal_by_reference.py）。"
         "每步用到哪些 prompt 由 TRL 的抽样顺序（种子 42）重建，并对照了训练日志。",
         "采样权重 = 推给 vLLM、真正用来抽 rollout 的 bf16 合并权重；训练策略 = HF + PEFT。"
         "零方差组 = 8 条回答奖励全相同（全对或全错），没有梯度；有信号的组 = 其余的组。"],
        chart_specs(per_step, windows),
        extra_tables(windows, seen, check0, mapping, pd.read_csv(rvm) if rvm.is_file() else None))
    (report / "report.html").write_text(page)
    pd.set_option("display.width", 220)
    show = windows[windows["which"] == "served"].pivot_table(
        index="steps", columns="class", values=["accuracy", "all_right", "all_wrong",
                                                 "signal_groups_per_step"], sort=False)
    print(show.round(3).to_string())
    print(check0.round(4).to_string(index=False))
    print(json.dumps(mapping, indent=1))
    print(f"wrote {report}/report.html")


def _series(windows: pd.DataFrame, which: str, cls: str, col: str) -> list:
    part = windows[(windows["which"] == which) & (windows["class"] == cls)].sort_values("x")
    return [[float(x), float(v)] for x, v in zip(part["x"], part[col])]


def chart_specs(per_step: pd.DataFrame, windows: pd.DataFrame) -> list:
    """Accuracy, zero-variance share and signal-carrying groups per step, by reference class."""
    x_ticks = [0, 25, 50, 75, 100, 125, 150, 175]

    def primary(cls: str, col: str, name: str, label: str) -> dict:
        return {"name": name, "label": label, "role": "a" if cls == "hire" else "b",
                "weight": "primary", "points": _series(windows, "served", cls, col)}

    def logged(col: str, name: str, role: str) -> dict:
        smooth = per_step[col].rolling(9, center=True, min_periods=1).mean()
        return {"name": name, "role": role, "weight": "secondary",
                "points": [[int(a), float(b)] for a, b in zip(per_step["step"], smooth)]}

    def rows(cols: list[tuple[str, str, str]], fmt) -> list:
        served = windows[windows["which"] == "served"]
        out = []
        for steps, part in served.groupby("steps", sort=False):
            row = [steps]
            for cls, col, _ in cols:
                hit = part[part["class"] == cls]
                row.append(fmt(hit[col].iloc[0]) if len(hit) else "–")
            out.append(row)
        return out

    def pct(v):
        return "–" if pd.isna(v) else f"{100 * v:.1f}%"

    def num(v):
        return "–" if pd.isna(v) else f"{v:.2f}"

    acc_cols = [("hire", "accuracy", "录"), ("reject", "accuracy", "拒"),
                ("all", "accuracy", "全部")]
    zero_cols = [("hire", "zero_variance", "录·零方差"), ("hire", "all_wrong", "录·其中全错"),
                 ("reject", "zero_variance", "拒·零方差"), ("reject", "all_wrong", "拒·其中全错")]
    sig_cols = [("hire", "signal_groups_per_step", "录"),
                ("reject", "signal_groups_per_step", "拒"),
                ("all", "signal_groups_per_step", "合计"),
                ("hire", "prompts_per_step", "录·每步组数"),
                ("reject", "prompts_per_step", "拒·每步组数")]
    note = ("实线带点：每个 25 步窗口（点在窗口中点）的期望值，来自窗口起点 checkpoint 的采样权重、"
            "该窗口实际用到的 prompt。")
    return [
        {"id": "accuracy", "title": "判决正确率，按参考类别",
         "subtitle": note + "细线：训练日志里每步的采样正确率（训练器把各 rank 的比例等权平均），"
                     "9 步居中滑动平均。",
         "x_label": "步", "x_ticks": x_ticks, "x_min": 0, "x_max": STEPS_TOTAL,
         "y_min": 0.75, "y_max": 1.0, "y_ticks": [0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
         "percent": True,
         "series": [primary("hire", "accuracy", "参考 = 录（期望）", "录"),
                    primary("reject", "accuracy", "参考 = 拒（期望）", "拒"),
                    logged("accuracy_hire_logged", "参考 = 录（训练日志）", "a"),
                    logged("accuracy_reject_logged", "参考 = 拒（训练日志）", "b")],
         "table": {"columns": ["窗口（步）", *[c[2] for c in acc_cols]],
                   "rows": rows(acc_cols, pct)}},
        {"id": "zero", "title": "零方差组占比，按参考类别",
         "subtitle": note + "灰色细线：训练日志的零方差占比（训练器只记录了全部组），9 步滑动平均。"
                     "表格把零方差拆成全对与全错。",
         "x_label": "步", "x_ticks": x_ticks, "x_min": 0, "x_max": STEPS_TOTAL,
         "y_min": 0.0, "y_max": 1.0, "y_ticks": [0, 0.25, 0.5, 0.75, 1.0], "percent": True,
         "series": [primary("hire", "zero_variance", "参考 = 录（期望）", "录"),
                    primary("reject", "zero_variance", "参考 = 拒（期望）", "拒"),
                    logged("zero_variance_all_logged", "全部组（训练日志）", "neutral")],
         "table": {"columns": ["窗口（步）", *[c[2] for c in zero_cols]],
                   "rows": rows(zero_cols, pct)}},
        {"id": "signal", "title": "每步仍有学习信号的组数，按参考类别",
         "subtitle": note + "每步共 16 组（纵轴只画到 8）。"
                     "灰色细线：训练日志里每步实际有信号的组数（两类合计），"
                     "9 步滑动平均；它应接近两条实线之和。",
         "x_label": "步", "x_ticks": x_ticks, "x_min": 0, "x_max": STEPS_TOTAL,
         "y_min": 0.0, "y_max": 8.0, "y_ticks": [0, 2, 4, 6, 8], "percent": False, "digits": 1,
         "series": [primary("hire", "signal_groups_per_step", "参考 = 录（期望）", "录"),
                    primary("reject", "signal_groups_per_step", "参考 = 拒（期望）", "拒"),
                    logged("signal_groups_all_logged", "两类合计（训练日志）", "neutral")],
         "table": {"columns": ["窗口（步）", *[c[2] for c in sig_cols]],
                   "rows": rows(sig_cols, num)}},
    ]


def extra_tables(windows: pd.DataFrame, seen: pd.DataFrame, check0: pd.DataFrame, mapping: dict,
                 rvm: pd.DataFrame | None) -> list:
    def pct(v):
        return "–" if v is None or pd.isna(v) else f"{100 * v:.1f}%"

    served = seen[seen["which"] == "served"].set_index(["checkpoint", "class"])
    trained = seen[seen["which"] == "trained"].set_index(["checkpoint", "class"])
    rows = []
    for (t, cls), r in served.iterrows():
        tr = trained.loc[(t, cls)] if (t, cls) in trained.index else None
        rows.append([t, "录" if cls == "hire" else "拒", pct(r["seen_zero_variance"]),
                     pct(r["unseen_zero_variance"]), pct(r["held_out_zero_variance"]),
                     pct(r["unseen_all_wrong"]),
                     pct(tr["unseen_zero_variance"]) if tr is not None else "–"])
    m, s = mapping["reconstructed_order"], mapping["shuffled_order_seed_12345"]
    out = [{"title": "已训练、未训练与留出 prompt 上的零方差占比（采样权重）",
            "note": "checkpoint t 时：已训练 = 第 1 到 t 步用过的训练 prompt；"
                    "未训练 = 之后才用（或从未用）的；"
                    "留出 = 256 个从未进训练的 prompt（每类 128）。"
                    "最后一列是同一批未训练 prompt 在训练策略（HF + PEFT）下的零方差，"
                    "用来对照采样权重的滞后。",
            "columns": ["checkpoint", "参考", "已训练", "未训练", "留出", "未训练·其中全错",
                        "未训练·训练策略"],
            "rows": rows},
           {"title": "校验：重建的抽样顺序",
            "note": "按重建顺序，每步期望值（该步之前最近 checkpoint 的采样权重、"
                    "该步的 16 个 prompt）与训练日志逐步值的相关系数；"
                    "对照组是一个随机打乱的顺序。",
            "columns": ["顺序", "正确率 r", "零方差 r", "正确率均值·期望/日志",
                        "零方差均值·期望/日志"],
            "rows": [[name, f"{d['accuracy_r']:.2f}", f"{d['zero_variance_r']:.2f}",
                      f"{pct(d['accuracy_mean_expected'])} / {pct(d['accuracy_mean_logged'])}",
                      f"{pct(d['zero_variance_mean_expected'])} / "
                      f"{pct(d['zero_variance_mean_logged'])}"]
                     for name, d in (("重建（种子 42）", m), ("随机打乱", s))]},
           {"title": "校验：第 0 步（SFT）的 token 概率对照已存的真实采样",
            "note": "已存采样：GRPO 数据构建时 vLLM 从合并的 SFT 模型在同一批行上抽的温度 1 样本"
                    "（hire 行 32 个，reject 行 8 个）；零方差用无偏估计（任取 8 个全一致的概率）。"
                    "留出行是在类内随机抽的，是干净的对照；训练行是按这些样本挑出来的"
                    "（tipped 行至少有一个错样本），所以它们的已存样本会比重新抽样错得多。",
            "columns": ["行", "参考", "行数", "每行采样数", "正确率·采样", "正确率·token",
                        "零方差·采样", "零方差·token", "全错·采样", "全错·token"],
            "rows": [["留出（干净）" if r["split"] == "validation" else "训练（按样本挑选）",
                      "录" if r["class"] == "hire" else "拒", r["rows"], r["samples_per_row"],
                      pct(r["accuracy_samples"]), pct(r["accuracy_tokens"]),
                      pct(r["zero_variance_samples"]), pct(r["zero_variance_tokens"]),
                      pct(r["all_wrong_samples"]), pct(r["all_wrong_tokens"])]
                     for _, r in check0.iterrows()]}]
    if rvm is not None:
        def pp(v):
            return "–" if pd.isna(v) else f"{v:+.2f}"

        def ci(lo, hi):
            return "" if pd.isna(lo) else f" [{lo:+.2f}, {hi:+.2f}]"
        out.append({
            "title": "原始比较与相同录用率比较：不稳定集合，各臂对 SFT（百分点）",
            "note": "口径：录用率 = 全部 31,050 个 prompt 中判录的比例；"
                    "集合 = 2,700 个（pair × 属性组 × 注入条件，不含无属性行），"
                    "属性变体决定不一致即不稳定。"
                    "原始·贪心 = 审计生成的决定（温度 0）；"
                    "原始·margin = 决定 token 的 log 概率差 > 0；"
                    "两者都是各模型在自己的录用率下与 SFT 相减。"
                    "相同录用率 = 把 SFT 的 margin 阈值移到该臂的录用人数再比（op-margin 标准）。"
                    "区间 = 按候选人 bootstrap 1,000 次，种子 42。"
                    "v1 行的相同录用率数字取自旧版报告、不 conform；r2-50 没有 margin 打分。"
                    "来源 reports/raw_vs_matched/table.csv（scripts/raw_vs_matched.py）。",
            "columns": ["臂", "标准", "录用率·贪心", "不稳定·贪心", "Δ 原始·贪心", "录用率·margin",
                        "不稳定·margin", "Δ 原始·margin", "SFT 移到", "SFT 在该点", "Δ 相同录用率"],
            "rows": [[r["arm"], r.get("standard", "–"), pct(r["greedy_hire_rate"]),
                      pct(r["greedy_unstable"]),
                      pp(r["greedy_delta_pp"]) + ci(r["greedy_ci_low"], r["greedy_ci_high"]),
                      pct(r.get("margin_hire_rate")), pct(r.get("margin_unstable")),
                      pp(r.get("margin_delta_pp"))
                      + ci(r.get("margin_ci_low"), r.get("margin_ci_high")),
                      pct(r.get("matched_sft_hire_rate")), pct(r.get("matched_sft_unstable")),
                      pp(r.get("matched_delta_pp"))
                      + ci(r.get("matched_ci_low"), r.get("matched_ci_high"))]
                     for _, r in rvm.iterrows()]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("score", "report"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", default=RUN_DIR)
        p.add_argument("--data", default=DATA_DIR)
        p.add_argument("--out", default=OUT_DIR)
        if name == "score":
            p.add_argument("--shard", type=int, default=0)
            p.add_argument("--num-shards", type=int, default=1)
            p.add_argument("--batch-size", type=int, default=8)
        else:
            p.add_argument("--log", default="logs/xw-grpo-5066266.out")
    args = parser.parse_args()
    cmd_score(args) if args.cmd == "score" else cmd_report(args)


if __name__ == "__main__":
    main()
