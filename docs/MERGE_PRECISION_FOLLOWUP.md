# Final-adapter serving comparison

Registered after the user authorized the follow-up runs on 2026-11-02. This is
an inference-only diagnostic on the development benchmark. It does not train,
select a new checkpoint, or establish held-out performance.

Question: do the differences between full-r32, window-r32 and window-r163 change
when the final adapter runs as PEFT instead of being folded into bf16 weights?
The current projection-based merge retention differs across these arms. That
quantity is a parameter-space measurement, not behavioral agreement.

## Conditions and reconstruction

Seeds 42, 43 and 44 use the declared final checkpoint-341 of each existing arm.
The saved final adapter must be byte-identical to checkpoint-341. Each chain
must contain exactly the common SFT-v3 adapter followed by the final adapter.

The script rebuilds the common SFT base on CPU using the existing merge code and
checks every loaded unique parameter against the saved SFT merge. It attaches
only the final adapter for PEFT scoring. It then merges that adapter on CPU,
checks every loaded unique parameter against the already-served checkpoint,
and scores the merged condition. No existing checkpoints are written.

Both conditions use the same HF implementation, bf16 base weights, SDPA,
tokenizer, texts, explicit positions, left padding, batch composition and
readout. PEFT's normal adapter autocasting remains enabled, and actual adapter
dtypes are recorded. This compares the two existing serving representations;
it does not isolate weight rounding from all consequences of PEFT execution.
Both fp32 residual readout and native bf16-logit margins are retained separately.
No vLLM threshold is imported. Comparisons at matched hire rates must use each
HF score table's own thresholds, with the same candidate-level grouping.

## Pilot and gates

The seed-42 pilot runs all three arms. Twelve complete CV/job pairs are selected
without outcomes, spaced over pair median prompt lengths. All cue styles,
families and attribute-free rows remain, 828 prompts per arm. The third pilot
condition loads the saved served checkpoint directly. Its fp32 and native
bf16 margins must reproduce the rebuilt merge within 1e-5 on every pilot row.
Any chain, weight, input, nonfinite-score or parity failure stops that arm.

Each full pass requires the same-arm passing pilot, identical shared input and
code/version fingerprints, and 31,050 distinct prompts. Every arm repeats the
bitwise served-weight check. The full pass has two scoring conditions only.
One Slurm node runs the three arms concurrently on three of four allocated GPUs.
The pilot time cap is 45 minutes. No automatic retry is scheduled. Set full-pass
caps after measuring the pilot throughput, allowing for its length distribution
and loader overhead. Used GPU-hours and allocated GPU-hours remain separate.

## Artifacts and resumption

Outputs are under `reports/research_followup/merge_precision/{pilot,full}/`.
Each arm has content hashes for chain, tokenizer/configuration and served weights,
the package versions, code hashes, exact prompt IDs and tokenization hash, job
metadata, resumable parquet chunks and a JSON summary. Raw prompt text is not
copied into these reports. Changed inputs refuse resume; partial chunks are
written atomically. An advisory lock rejects duplicate workers for one arm.

The primary diagnostic is the within-checkpoint change in margins, decisions and
matched-rate instability caused by the serving representation. Across-arm mean
differences, three seeds and working-point curves should then be compared on the
same HF basis. A changed ordering would motivate numerical controls before a
claim about edit localization; an unchanged ordering would reduce this concern.

## Pilot execution

Pilot job 5241337 stopped before scoring because the saved multimodal checkpoint
uses `model.language_model.*` while HF's text model exposes `model.*`. Its three
cost records total 0.0886 used GPU-hours. Those outputs and identities remain in
the original `pilot/` directory.

After review, the verifier gained an explicit prefix mapping restricted to the
Qwen3.5 conditional-generation checkpoint layout. It requires a one-to-one name
map and retains exact dtype and bitwise weight checks. Seven CPU tests passed.
Supervised retry 5241355 writes under `attempts/r2-qwen-text-layout/`; the active
source pointer and submission ledger name that directory.

The retry completed in 5m11s. All three arms passed both common-SFT and final
served-weight checks. Rebuilt versus directly loaded saved scores were exactly
equal on both readouts, 828 prompts per arm. The unmerged pass took 52.2–59.4s;
the merged pass took 43.6–43.7s. Torch peak allocated memory was 20.4–20.6 GiB.
This validates the loader and comparison. The length-selected pilot is not an
estimate of full-benchmark behavioral effects.

Each of the ten saved model configurations, common SFT plus nine final arms,
has SHA256 `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`.
The jobs record CUDA-visible GPU capacity as 102005473280 bytes, exactly 95 GiB;
the device-name string says NVIDIA GH200 120GB. Neither is relabelled as a
measurement of physical HBM capacity.

## Full passes

The measured pilot supports a two-hour cap per seed node. Full jobs are 5241383
for seed 42, 5241378 for seed 43, and 5241379 for seed 44. Each runs three arms
concurrently at batch size 8, retaining the pilot's environment and source.
Expected scoring plus overhead is about 65–85 minutes per arm; the three-node
allocation cap is 24 GPU-hours, distinct from expected used GPU-hours.

Seed 42 also runs the separately reviewed mechanism batch diagnostic on GPU 3,
with a 900-second timeout. Its FLA import path and flags stay inside its own
subshell. Its exit status and cost are separate from the merge comparison.
After all three merge workers of a node succeed, the wrapper invokes the CPU
aggregator with `--when-ready`. The last completed seed produces the aggregate
once all nine validated summaries exist; no new GPU computation is triggered.
