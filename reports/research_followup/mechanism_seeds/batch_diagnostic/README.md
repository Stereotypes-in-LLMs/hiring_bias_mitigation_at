# Baseline batch-context replay

The replay exactly reproduced both historical baseline passes for the selected prompts. Repeating the same context twice gave zero difference, while switching from the original full-margin batches to the original patch batches changed one prompt margin by 15.687629 logits. This establishes a reproducible batch-context sensitivity in this tested HF execution. It is not evidence that the discrepancy is harmless noise.

## Scope and selection

The checkpoint is DPO seed 43, `dpo4880_s43`, and the contrast is combat versus civilian, explicit cues. From the existing 64 SFT-selected pairs, the frozen CPU plan chose the four largest and four smallest observed patch/full baseline discrepancies. These eight pairs and 16 prompts are diagnostic extremes, not a random sample or an estimate of the frequency of this problem.

The model uses the original Qwen Base with the SFT-v3 and DPO adapters activated together in HF PEFT. The diagnostic did not merge weights, train, or modify the environment. It compared:

- The original full-margin batches of eight, reconstructed using stored part order and stable sorting by rendered text length.
- The original patch A/B batches, recovered directly from completed patch parquet blocks.
- Each selected prompt alone, as a batch of one.

Each context ran twice. The second repetition reversed the context order. The run performed 90 forwards: per repetition, 15 full-margin batches, 14 patch A/B batches and 16 singletons. All 16 target prompts and all 188 distinct prompts including batch neighbors matched freshly rendered and tokenized inputs exactly. These were baseline forwards only, without activation swaps or source-activation capture.

## Measurements

All differences below are maximum absolute differences over the 16 selected prompts and both repetitions, in logits.

| Replayed context | Between repetitions | Versus recorded full pass | Versus recorded patch pass |
|---|---:|---:|---:|
| Original full batches | 0 | 0 | 15.687629 |
| Original patch batches | 0 | 15.687629 | 0 |
| Singletons | 0 | 4.771772 | 10.915857 |

For original patch batches versus original full batches, one of eight diagnostic pairs changed its hire/reject disagreement at the zero threshold in both repetitions. For anonymous pair W01, the full-batch margins were 13.920051 and -15.481561, yielding different decisions; the patch-batch margins were -1.767579 and -16.236652, yielding two rejects. Hire means margin strictly greater than zero.

The maximum change in the pair effect `dm = margin_A - margin_B` was 14.932538 logits. Absolute-effect rank correlation within these eight diagnostic pairs was 0.880952 for patch versus full batches and 0.952381 for singleton versus full batches. No `dm` direction changed within this deliberately selected set. These ranks concern pair effect magnitudes, not attention heads or components. The zero-threshold result does not describe a calibrated hire-rate operating point.

## What the replay resolves

The preceding CPU audit had found the cross-pass discrepancy but could not isolate a reproducible execution-context dependence. This follow-up reproduces both historical values under one current runtime, using the same checkpoint, prompt tokens and readout definition. The discrepancy therefore does not require a changed checkpoint, stale prompt text, a different decision prefix, or random variation between these two repetitions.

The replay does not separate the effects of batch size, neighbor composition, padding length or the resulting kernel execution. It does not identify a specific faulty kernel. The singleton output is another context, not a ground-truth margin.

The diagnostic also does not rerun activation interventions. It cannot establish whether component-profile correlations, L15.H11/L23.H1 effect sizes, or tested-head ranks remain stable when batching changes. Those results remain measurements under their recorded patch batching, and their numerical robustness across contexts remains untested. No causal explanation of the high-hire-rate behavioral results follows from this experiment.

## Runtime and cost

The diagnostic ran on GPU 3 of shared job 5241383, an NVIDIA GH200 120GB, while separate merge work used the other GPUs. The recorded runtime was PyTorch 2.13.0+cu130 and Transformers 5.17.0. FLA and fla-core were 0.6.0, loaded from `/ritom/scratch/cscs/zhang/xw/pylib-fla/fla/`; `causal_conv1d` was unavailable. `FLA_DISABLE_BACKEND_DISPATCH=1`, TF32 matmul was disabled and float32 matmul precision was `highest`.

Historical artifact metadata recorded the same PyTorch/Transformers versions and FLA/causal-convolution availability, but did not record the exact FLA version, module path or dispatch runtime flag. That provenance gap remains. The current replay nevertheless reproduced both recorded baseline contexts exactly for the selected prompts.

The script's model-load-plus-replay interval was 51.03 seconds. The separate cost wrapper measured 103.2 seconds for the entire process, including CPU preparation, using 0.0287 used GPU-hours and 21,392.8 J of sampled GPU energy. PyTorch peak allocated memory was 19.19 GiB; the cost wrapper's sampled process peak was 20.84 GiB. These are different memory measures. No whole-node allocated GPU-hours are attributed to this diagnostic.

The copied cost record is `cost.json`, sourced from `/ritom/scratch/cscs/zhang/xw/cost/records/20261102T202629_patching_mechanism-batch-diagnostic_5241383_16927.json`.

## Artifact verification

The completed summary refers to plan SHA256 `eba6f06c9f195e1bbd37b857122b98e4848a4a261fc5e700c90396badc81b98f`, which matches the downloaded `plan.json`. All 24 files and both adapter weight hashes recorded in the plan still matched on retrieval. All seven repository files covered by those plan hashes also matched the local copies. No frozen source or CPU summary was changed to produce this interpretation.

The 96 prompt measurements and 64 pair rows were checked independently against the summary. Within-context repeat differences, historical replay errors, `dm`, absolute effects and zero-threshold flags all reconciled. `input_hash_verification.json` and `result_verification.json` record these checks. The hashes cover the files named in the plan; they are not a claim that every installed library file had historical hash provenance.

The report contains anonymous diagnostic labels and aggregate measurements, not CV text or candidate identifiers. `plan.json` preserves the selection rule, contexts, input hashes and representation; `prompt_margins.csv` and `pair_metrics.csv` preserve the measurements; `summary.json` records completion and runtime.
