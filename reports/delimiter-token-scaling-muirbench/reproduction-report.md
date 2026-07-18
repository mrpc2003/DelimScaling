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

- Compute: local 2× NVIDIA RTX PRO 6000 Blackwell (97,887 MiB each), Python 3.10.20, PyTorch 2.7.1+cu128.
- Model/dataset: cached model revision `66285546d2b821cf421d4f5eb2576359d3770cd3`; MUIRBENCH revision `4c393cffc985c77d28de3b9045e2e5186920df80`.
- Runtime: two processes and SDPA for vision/language, with blockwise `cu_seqlens` SDPA to avoid a quadratic mask; Python-object result gathering uses CPU Gloo. The paper README uses four-process FlashAttention2 and sampled decoding (temperature 0.2). Local CUDA 11.2 nvcc cannot build Blackwell sm_120 FlashAttention.
- `token: False` only removes an unnecessary credential requirement for the public ungated dataset. `pytablewriter==1.2.1` was pinned solely for final CLI table rendering.

Attention-reduction and no-overhead claims were not attempted; local measured scaled runtime was higher under this controlled runtime deviation.
