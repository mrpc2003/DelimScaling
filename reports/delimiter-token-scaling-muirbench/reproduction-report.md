# Full MuirBench reproduction

## Headline claim

Paper Table 1 reports Qwen2.5-VL-3B MuirBench 37.31% to 42.42% (+5.11 pp; 970/2600 to 1103/2600). On the full public test set, this reproduction obtained 37.00% (962/2600) baseline and 41.19% (1071/2600) with delimiter scaling: +4.19 pp. Verdict: **partially reproduced** — the direction and most of the gain reproduce, but the scaled score is 1.23 pp below the paper.

| Run | Score | Delta | Time |
|---|---:|---:|---:|
| Paper | 37.31% | +5.11 pp | not reported |
| Local baseline | 37.00% | — | 1,009.36 s |
| Local scaled | 41.19% | +4.19 pp | 1,814.18 s |

The code setting was `lambda=8`, layers 0–3. It is the released-code reference setting, not a proven exact Table-1 MuirBench factor: Appendix A.3 says factors were tuned per model/benchmark on 10% of test, and that tuning protocol was not reproduced. Appendix A.5 supports layers 0–3.

## Independent result-integrity check

The scaled sample JSONL contains exactly 2,600 rows, 2,600 unique `doc_id`s spanning 0–2599, and no duplicates. Its exact-correct count is 1,071; the results JSON reports both original and effective sample counts as 2,600. Paired against the baseline: 277 improved, 168 regressed, 794 both-correct, and 1,361 both-wrong — net +109 correct answers.

## Controlled deviations and provenance

- Fixed command on both experiments: `bash reproduction/run_muirbench.sh`; upstream base: [`a2880738e3ed7284953f3611e7c06a3c0ede9334`](https://github.com/mrpc2003/DelimScaling/commit/a2880738e3ed7284953f3611e7c06a3c0ede9334).
- Baseline: experiment `f09f16ea-be33-45f0-afd8-7fa091d366ca`; run `8bdfef7a-f9a9-43bf-8b1b-d3923347c336`; [`orx/memory-safe-blockwise-vision-sdpa-baseline`](https://github.com/mrpc2003/DelimScaling/tree/orx/memory-safe-blockwise-vision-sdpa-baseline), commit [`fae6265`](https://github.com/mrpc2003/DelimScaling/commit/fae6265); 1,009.36 s.
- Scaled: experiment `c471c993-9904-4539-ab0d-ee424f64f52d`; run `49e393b4-3fc3-4d43-bcc5-901b1089d8fe`; [`orx/released-code-delimiter-scaling-lambda-8-layers`](https://github.com/mrpc2003/DelimScaling/tree/orx/released-code-delimiter-scaling-lambda-8-layers), commit [`cfde04c`](https://github.com/mrpc2003/DelimScaling/commit/cfde04c); 1,814.18 s.
- Compute: local 2× NVIDIA RTX PRO 6000 Blackwell (97,887 MiB each), Python 3.10.20, PyTorch 2.7.1+cu128.
- Actual seed tuple: `random=0,numpy=1234,torch=1234,fewshot=1234`. Pinned dependencies include Transformers 4.53.1 and Datasets 3.6.0.
- Model/dataset hashes are local-cache-resolved revisions: model `66285546d2b821cf421d4f5eb2576359d3770cd3`; MUIRBENCH `4c393cffc985c77d28de3b9045e2e5186920df80`.
- Runtime: two processes and SDPA for vision/language, with blockwise `cu_seqlens` SDPA to avoid a quadratic mask; Python-object result gathering uses CPU Gloo. The paper README uses four-process FlashAttention2 and sampled decoding (temperature 0.2). Local CUDA 11.2 nvcc cannot build Blackwell sm_120 FlashAttention.
- The memory-safe blockwise SDPA is mathematically equivalent to the intended block-diagonal varlen attention semantics, but not bitwise identical: small validation maximum difference was ~2.98e-7 in FP32 and 0.0078125 in BF16.
- `token: False` only removes an unnecessary credential requirement for the public ungated dataset. `pytablewriter==1.2.1` was pinned solely for final CLI table rendering.

Initial setup/diagnostic failures and dense-mask OOM run `c858b69d` were excluded from scientific comparison. Attention-reduction and no-overhead claims were not attempted; local measured scaled runtime was higher under this controlled runtime deviation.
