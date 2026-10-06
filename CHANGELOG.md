# Changelog

## 0.6.0 (2026-10-06)

- vLLM is pinned to upstream main at
  [`a3e0243b`](https://github.com/vllm-project/vllm/commit/a3e0243b1c3ea313b902a43b0f6559fb4148ff81)
  (0.30.1rc1.dev707), up from `1b3b88ec` (the #57250 merge). It has every DiffusionGemma change
  merged after #57250:

  | PR | Change |
  |---|---|
  | [#58216](https://github.com/vllm-project/vllm/pull/58216) | Constrained reads: a read can score only its label tokens |
  | [#58226](https://github.com/vllm-project/vllm/pull/58226) | One-pass kernel for the sampler statistics |
  | [#51994](https://github.com/vllm-project/vllm/pull/51994) | The attention mask no longer freezes under CUDA graph replay |
  | [#48521](https://github.com/vllm-project/vllm/pull/48521) | The LM head gets the checkpoint's quantization config |

  Both image patches (512 label ids, bidirectional image attention) are still needed and still
  apply. The `clm` and `jevk5` images stay on `1b3b88ec`.
- New setting `OPENJEV_CONSTRAINED=1` sends #58216's `diffusion_constrained`. It is off by
  default: the logprobs come back normalized over the labels, which changes the entropy that
  triggers re-reads. With it on, reads were about 10% faster one at a time and 178 of 180 answers
  were the same.
- Accuracy against 0.5.0 on 2,501 labeled text questions (choice and yes/no) from the v6
  fast-dev set: 76.7% against 77.0%. 0.5.0 alone was right on 35 and 0.6.0 alone on 28, within
  run-to-run noise. Images and scores were not re-measured.
- `scripts/bench.py` and a README throughput table by state length, up to the 64K window
  ([#5](https://github.com/razorback16/openjev/issues/5)).
