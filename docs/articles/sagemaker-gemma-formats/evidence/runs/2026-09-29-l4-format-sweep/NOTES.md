# 2026-09-29 — Weight formats on one L4: int4, int4 embeddings, FP8, int8

Every text-only build of Google's Gemma 4 QAT weights measured on SageMaker `ml.g6.xlarge`
(1× NVIDIA L4) on 2026-09-29 (E2B Google QAT 2026-09-25, E4B Google QAT 2026-09-27), same
container (vLLM 0.30.0), standard settings (`max_model_len` 8192, memory 0.9, temperature 0) and
`compare.py`. \*31B runs on `ml.g6.2xlarge` with reduced settings (memory 0.97, 4 sequences,
1,024 context), the only way it starts on an L4.

| Size | Build | Linear layers | Embeddings + lm_head | Weights (GiB) | KV cache (tokens) | Decode (tok/s) | 16 parallel (tok/s) | Correct (of 40) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| E2B | Google QAT | int4 | bf16 | 8.01 | 867,999 | 105.1 | 1077.25 | 37 |
| E2B | text-only | int4 | bf16 | 6.33 | 650,783 | 103.0 | 927.0 | 36 |
| E2B | ple4 | int4 | int4 per-layer only | 3.19 | 1,180,383 | 104.1 | 1088.35 | 36 |
| E2B | emb4 | int4 | int4 | 2.86 | 1,209,977 | 141.7 | 1259.15 | 36 |
| E2B | FP8 | FP8 | bf16 | 7.07 | 589,433 | 73.0 | 810.95 | 35 |
| E2B | FP8 + emb4 | FP8 | int4 | 3.6 | 1,144,044 | 89.7 | 939.45 | 36 |
| E2B | int8 | int8 | bf16 | 7.07 | 582,397 | 75.5 | 825.05 | 37 |
| E2B | int8 + emb4 | int8 | int4 | 3.6 | 1,138,763 | 92.1 | 980.45 | 34 |
| E4B | Google QAT | int4 | bf16 | 11.04 | 200,972 | 60.7 | 599.5 | 36 |
| E4B | text-only | int4 | bf16 | 8.84 | 119,416 | 62.1 | 447.7 | 36 |
| E4B | emb4 | int4 | int4 | 4.52 | 376,156 | 79.9 | 848.95 | 36 |
| E4B | FP8 | FP8 | bf16 | 10.46 | 77,376 | 39.4 | 508.2 | 36 |
| E4B | int8 | int8 | bf16 | 10.46 | 74,463 | 40.1 | 517.55 | 38 |
| 12B | text-only | int4 | bf16 | 8.18 | 41,654 | 29.0 | 358.5 | 40 |
| 12B | emb4 | int4 | int4 | 7.36 | 52,541 | 35.0 | 411.8 | 40 |
| 12B | FP8 | FP8 | bf16 | 12.65 | 20,383 | 16.8 | 229.65 | 40 |
| 12B | int8 | int8 | bf16 | 12.65 | 19,929 | 17.0 | 230.9 | 40 |
| 26B A4B | text-only | int4 | bf16 | 14.8 | 19,183 | 65.8 | 441.25 | 40 |
| 26B A4B | emb4 | int4 | int4 | 14.2 | 30,456 | 91.5 | 596.6 | 40 |
| 31B* | text-only | int4 | bf16 | 18.7 | 2,555 | 12.6 | 39.4 | 40 |
| 31B* | emb4 | int4 | int4 | 17.55 | 3,241 | 14.1 | 50.7 | 40 |
- **emb4** packs `embed_tokens` and an untied `lm_head` (and, at E2B/E4B, the per-layer
  embeddings) to int4 on the QAT grid. **ple4** packs only the per-layer table.
- FP8 and int8 are W8A8 with per-channel weight scales and per-token dynamic activations. The L4
  runs both natively.
- Decode is the fixed-length rate that cancels client overhead; 16-parallel rates include each
  run's `aws` CLI overhead (see `../2026-09-29-e2b-e4b-repack/NOTES.md`).

## Findings

- **int4 embeddings and `lm_head` speed up decode at every size**: emb4 against text-only is
  1.38× (E2B), 1.29× (E4B), 1.21× (12B), 1.39× (26B A4B) and 1.12× (31B, reduced settings),
  with identical answers at every size. ple4 (per-layer table only) decodes at text-only speed (104.1 against 103.0) while saving
  3.1 GiB, so the speed comes from the int4 `lm_head`, read in full for every generated token.
  The gain is largest where the output layer is a large share of each token's reads: the small
  dense sizes and the 4B-active MoE; smallest at 31B dense.
- **8-bit linears are slower than 4-bit on the L4**: FP8 against text-only is 0.71× (E2B), 0.63×
  (E4B) and 0.58× (12B); int8 is within 3 % of FP8 at every size. Decode here is bound by memory
  bandwidth, and 8-bit weights are twice the bytes of int4, so native FP8/int8 tensor cores do not
  help. With int4 embeddings, E2B int8 reaches 92.1 and FP8 89.7, still below emb4's 141.7.
- **No format costs measurable accuracy**: every build scores 34–40 of 40, inside what 40
  questions can resolve. Answers are byte-identical between emb4 and text-only at every size.
- **FP8 and int8 builds of 26B and 31B do not fit one L4** (about 26 and 31 GB of 8-bit weights).

## SageMaker start-up delay (26B emb4)

`gemma-4-26b-emb4-l4` was created 00:09:22Z and reached InService 03:35:58Z. Its container
(`AllTraffic/i-0bb0c45fa9756df1b`) was serving from 00:18:29Z and answered every SageMaker
`/ping` from 00:18:39Z with 200 OK, every 5 s, for 3.29 h before InService: about $3.71 of
instance time at $1.1267/h with nothing to measure. The other 20 endpoints that reached InService
today took 11.3 minutes at most. The run folders hold the container log.

## Files

- `run.sh`, `run.log`, `watchdog.sh`, `watchdog.log`, `measure-*.json`, `.log`, `compare-*.json`
- `../2026-09-29-l4-<size>-<format>/` (including `l4-31b-emb4`) — deploy attempt, timeline, run environment, container log
- Builds: `../2026-09-29-e4b-8bit-emb4-build/`, `../2026-09-29-big-8bit-emb4-build/`,
  `../_emb4logs/`; publishing: `../2026-09-29-publish/`
