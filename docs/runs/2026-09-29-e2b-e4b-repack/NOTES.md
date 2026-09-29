# 2026-09-29 — Gemma 4 E2B and E4B W4A16 repacks against Google's QAT on one L4

[`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct)
and [`xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct`](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct)
are Google's `-qat-q4_0-unquantized` builds written out as compressed-tensors W4A16 (int4,
group 32, bf16 scales), the method of the 12B and 26B repacks. Each ran on `ml.g6.xlarge`
(1× L4), the instance Google's `-qat-w4a16-ct` was measured on: E2B on 2026-09-25
(`../2026-09-25-qat-vs-bf16/`), E4B on 2026-09-27 (`../2026-09-27-e4b-qat-vs-bf16/`). Same
container tag (vLLM 0.30.0), standard settings (`max_model_len` 8192, GPU memory utilization
0.9, temperature 0), measurement script and region (us-east-2). Configs at Hugging Face
revisions `da64116` (E2B) and `0a3e624` (E4B).

## Results (`compare-e2b.json`, `compare-e4b.json`; ratios by `compare.py combine`)

| Measure | E2B Google | E2B repack | E4B Google | E4B repack |
| --- | ---: | ---: | ---: | ---: |
| Checkpoint (bytes) | 8,315,979,014 | 7,510,672,646 | 11,513,497,412 | 10,171,320,132 |
| Weights in GPU memory (GiB) | 8.01 | 7.26 | 11.04 | 9.79 |
| KV cache (tokens) | 867,999 | 929,454 | 200,972 | 234,879 |
| Decode, one request (tokens/s) | 105.1 | 107.2 | 60.7 | 60.4 |
| 1 request, 256 tokens (tokens/s) | 85.35 | 82.35 | 46.45 | 51.05 |
| 4 parallel (tokens/s) | 328.7 | 322.0 | 179.35 | 198.7 |
| 16 parallel (tokens/s) | 1077.25 | 1059.25 | 599.5 | 666.9 |
| Questions correct (of 40) | 37 | 37 | 36 | 36 |

- Decode rate is (512 − 16) / (median wall at 512 tokens − median wall at 16 tokens), 5 calls
  per length. 512-token ranges: E2B Google 5.392–5.526 s, repack 5.434–5.718 s; E4B Google
  9.582–9.659 s, repack 9.083–9.335 s.
- 16-parallel batches: E2B repack 1059.6 and 1058.9, E4B repack 664.0 and 669.8 tokens/s.

## The parallel figures include the client

The parallel rates are output tokens over the wall time of a batch of `aws
sagemaker-runtime invoke-endpoint` calls, so each includes the per-call cost of the CLI
process, request signing and the network. The decode rate cancels that cost; the parallel
rates do not. It differed between runs:

| | E4B Google (09-27) | E4B repack (09-29) |
| --- | ---: | ---: |
| Per-call fixed cost (s) | 1.174 | 0.782 |
| 256 tokens at the decode rate (s) | 4.217 | 4.238 |
| Batch wall, 1 / 4 / 16 parallel (s) | 5.512 / 5.714 / 6.833 | 5.016 / 5.153 / 6.142 |

Generating 256 tokens takes the same time on both builds; the 0.50–0.69 s gap in batch wall
follows the 0.39 s gap in per-call cost and grows with concurrency. **The E4B repack's
1.10–1.11× at 1–16 parallel is the client, and the two builds decode at the same rate.** The
E2B rows show the same pattern at a smaller size. Compare builds by the decode rate; compare
parallel rates only within one run.

## Where the memory goes

Both Google builds store `lm_head.weight` although their configs set `tie_word_embeddings:
true`, so the output layer should reuse `embed_tokens`. The repacks drop it
(`tensors-E2B-*.json`, `tensors-E4B-*.json`, read from the safetensors headers):

| | E2B | E4B |
| --- | ---: | ---: |
| `lm_head.weight` in Google's build | 262,144 × 1,536 bf16 | 262,144 × 2,560 bf16 |
| Bytes | 805,306,368 (0.75 GiB) | 1,342,177,280 (1.25 GiB) |
| Sampled rows byte-identical to `embed_tokens` | 7 of 7 | 7 of 7 |
| Google file − repack file | 805,306,368 | 1,342,177,280 |
| GPU weights, Google − repack (GiB) | 0.75 | 1.25 |

The file difference is exactly that tensor, and vLLM loads Google's copy, so the saving
reaches the GPU and becomes 7 % (E2B) and 17 % (E4B) more KV cache. Rows sampled: 0, 1, 1000,
50000, 131072, 200000 and 262143.

## Quality

The same 40 seeded questions (15 two-digit multiplications, 15 three-number sums, 10
capitals), temperature 0, scored by regex. E2B: 39 of 40 answers identical; the other is
add-14 (150 + 937 − 139 = 948), wrong in both (Google `1010`, repack `1018`). E4B: 37 of 40
identical; the other three are sums both builds get wrong with different wrong answers
(add-3, add-9, add-14). Scores are equal.

## The checkpoint's config.json

The first uploads' quantization ignore list had 115 entries and left out 140 linear layers of
the audio tower that the repacks keep in bf16. vLLM builds a linear layer that is not ignored
as int4 and looks for `weight_packed`, so the load would fail, as it did for the 12B
(`../2026-09-28-12b-repack/NOTES.md`). Checked offline before any deploy, against
`index-repack-*.json`; the list was replaced with Google's 250 entries (revisions `da64116`,
`0a3e624`), which name every bf16 linear layer and none of the 276 (E2B) or 343 (E4B)
quantized ones. Nothing else in the config differs from Google's except an `observer`
metadata field vLLM does not read. Configs kept here as `config-*.json`.

## Start-up and teardown

E2B: created 17:53:19Z, weights loaded 18:00:06Z (52.0 s load), InService 18:02:49Z,
measured 18:02:52–18:05:42Z, deleted 18:05:46Z. E4B: created 18:05:51Z, weights loaded
18:13:12Z (84.4 s), InService 18:15:54Z, measured 18:15:57–18:19:29Z, deleted 18:19:35Z.
`watchdog.log` shows nothing left in any of the three US regions. One deployment of each.

## Files

- `run.sh`, `run.log`, `watchdog.sh`, `watchdog.log`
- `measure-*.json`, `measure-*.log`, `compare-e2b.json`, `compare-e4b.json`
- `config-*.json`, `index-repack-*.json`, `tensors-*.json`
- `../2026-09-29-{e2b,e4b}-repack/` — deploy attempt, timeline, run environment, status at
  InService, container log, delete time
