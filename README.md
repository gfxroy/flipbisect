# flipbisect

**git bisect for LLM nondeterminism.** You send the same request at temperature 0 and get different answers. flipbisect replays it under controlled conditions, finds the first token where the answers split, and automatically toggles server settings to find the smallest change that causes the split.

→ Sample report: https://gfxroy.github.io/flipbisect/

```
$ flipbisect bisect --model smollm2-360m-q4_k_m.gguf \
    --good cache_prompt=false flash_attn=on parallel=1 \
    --bad  cache_prompt=true  flash_attn=off parallel=8 threads=8 ubatch=32
· launching llamacpp parallel=1, cache_prompt=False, flash_attn=on      -> 0/32 unstable runs
· launching llamacpp threads=8, ubatch=32, parallel=8                   -> 17/32 unstable runs
· launching llamacpp threads=8, ubatch=32, parallel=1, ...              -> 18/32 unstable runs
· launching llamacpp threads=8, parallel=1, cache_prompt=False, ...     -> 20/32 unstable runs
minimal: {'threads': 8}
```

## Why

Temperature 0 is supposed to be greedy and repeatable. In practice, serving engines change the floating-point order of operations depending on batch size, cache state, kernel choice and thread count. When two candidate tokens are nearly tied, that tiny drift flips the argmax, and the rest of the answer goes somewhere else. The [vLLM tracker #27433](https://github.com/vllm-project/vllm/issues/27433) is one of the most discussed threads on this. Thinking Machines' [batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) fixes the kernels. What was missing is a tool that tells you **which setting on your server** is doing it, on your model and hardware.

## What it does

1. **Black-box probe** (any OpenAI-compatible server). It replays identical temperature-0 requests: solo, after a cache-evicting request, and inside bursts of 2, 4 and 8 concurrent requests with padding neighbours of different lengths and shuffled order.
2. **Tie vs real.** It compares each run to the first one token by token. At the first split it reads the top-k logprobs and reports the *margin* (top-1 minus top-2 in the reference) and the *swing* (how far the two candidates moved). A split with margin under `--tie-eps` (default 0.1 nats) where the other run picked the runner-up is labelled `near-tie`. Everything else is `divergence`.
3. **White-box sweep and bisect.** It launches the server itself with different settings, either one at a time (`sweep`) or with delta debugging over the settings that differ between a good and a bad config (`bisect`), and returns a 1-minimal reproducing change. There are two predicates:
   - `stability`: the same config gives different outputs under load.
   - `parity`: the config's output differs from the baseline config's output.
4. **Reports** as JSON plus a self-contained monochrome HTML page.

Backends: **llama.cpp** (implemented and tested, CPU). **vLLM** and **SGLang** ship as flag maps on the same interface (`max_num_seqs`, prefix caching, chunked prefill, eager vs CUDA graphs, attention backend, TP, batch-invariant / deterministic mode). They are **untested** because this was built on a CPU-only machine.

## Quickstart

```bash
pip install -e .           # stdlib only, no runtime dependencies
# black-box, against anything OpenAI-compatible
flipbisect probe --url http://localhost:8080 --repeats 3
flipbisect probe --url https://api.example.com --model my-model --api-key-env MY_KEY --chat
# white-box, llama.cpp
flipbisect sweep  --model model.gguf --server-bin /path/to/llama-server --set cache_prompt=false
flipbisect bisect --model model.gguf --server-bin /path/to/llama-server \
    --good threads=1 --bad threads=8 flash_attn=on --predicate parity
```

Knobs for llama.cpp: `threads, batch, ubatch, parallel, cache_prompt, flash_attn, cache_type_k, cache_ram`.

## Real findings

Setup: SmolLM2-360M-Instruct Q4_K_M, llama.cpp `b12015` CPU build, an 8-vCPU Linux box, 4 prompts, 48–64 tokens, top-5 logprobs. Raw JSON is in [`runs/`](runs/).

| Experiment | Result |
|---|---|
| Probe on defaults (`np=4, t=4, fa=off, cache_prompt=on`), 60 runs | 37 identical, 18 near-tie flips, 5 real divergences. Even a plain **solo repeat** split from the first request (3/12). |
| Sweep from defaults | Turning **`cache_prompt` off** took unstable runs from 18/32 to **0/32**, and so did `flash_attn=on` and `cache_type_k=q8_0`. Changing `threads, batch, ubatch, parallel` alone did not fix it (`parallel=1` still had 3/32). |
| Sweep with `cache_prompt=false` | Mostly stable (0–2/32 per config), **except `threads=8`** (every core on this box): 10/32, then 20/48 in a re-run. `cache_ram=0` changed nothing. |
| Stability bisect, 5 settings differ | ddmin needed 4 server launches to isolate **`threads=8`** alone. |
| Parity bisect (solo outputs vs baseline) | **`flash_attn=on`** alone changes the solo greedy output on 3 of 4 prompts. `batch` and `ubatch` do not (pytest-verified). |
| Split margins | Real splits happened at margins of 0.09–0.24 nats, with swings up to 0.57 nats. This is far more than "1e-6 float noise" suggests, plausibly because Q4 dequantisation magnifies the reordering. |

Takeaways for llama.cpp on CPU: prompt-cache reuse is the main source of temp-0 instability, because a warm prefix is computed in a different batch shape than a cold one. Separately, using every core for threads made outputs unstable on this machine. The `threads=8` effect also depends on load: one later run with only 2 prompts showed no flips. I have not root-caused it.

## Tests

`pytest -q` runs 9 unit tests (diff classification, both logprob formats, ddmin, command builders, HTML escaping) and 3 live tests that launch a real llama-server:
- with prompt cache off and 1 thread, outputs stay identical under a 4-way burst;
- a parity bisect blames `flash_attn` alone;
- a stability bisect returns a minimal reproducer, and **skips** if the bad config happens to stay stable on the machine, since that effect depends on the hardware.

CI downloads the same llama.cpp build and model.

## Limits

- **"0 flips" isn't proof.** Several stable-looking configs showed rare flips (about 1–2%). Bisect treats one flip as bad and needs `--repeats` big enough to see it. ddmin assumes the effect is monotone.
- The tie threshold is a judgement call. Margin and swing are always reported, so you can apply your own.
- Black-box mode can't flush a provider's cache. "Cold" means sending an unrelated long prompt first.
- Only greedy decoding is checked, and only the first split is analysed, not later ones. It does not attribute the split to a layer or kernel.
- vLLM and SGLang backends are written but unverified (no GPU). Logprob formats from hosted APIs vary. Both the OpenAI legacy and chat formats are parsed.
- One model, one machine, one llama.cpp build. Treat the findings as a demo, not a benchmark.

## Related work

- [thinking-machines-lab/batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) and the "Defeating Nondeterminism in LLM Inference" post: batch-invariant kernels (the fix, not the diagnosis).
- vLLM batch-invariant mode and [vllm#27433](https://github.com/vllm-project/vllm/issues/27433); SGLang `--enable-deterministic-inference`.
- [DiFR / provider auditing](https://adamkarvonen.github.io/machine_learning/2025/11/28/difr.html): uses near-determinism to audit providers.
- `dncampo/llm-prompt-nondeterminism`: measures logit drift. No tool I found bisects server settings.

MIT © Aaditya Roy
