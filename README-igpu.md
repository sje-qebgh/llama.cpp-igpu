# llama.cpp for integrated GPUs (Vulkan)

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) tuned for **integrated GPUs**:
Intel Meteor Lake (Arc Xe-LPG, no matrix cores) for prefill and Lunar Lake (Arc 140V, Xe2) for token generation,
with a focus on **low-bit quants (Q2-Q4, IQ\*)** that people actually run on laptops: MoE models at 2-4 bits per weight.

Everything here is in the Vulkan backend. Other backends are untouched.

## What is different from upstream

| Area | Change | Status |
|---|---|---|
| Prefill, IQ2_S | integer dot (q8_1) MMQ shaders for mul_mat and mul_mat_id | measured |
| Prefill, IQ4_NL | integer dot MMQ shaders (upstream only had them on coopmat GPUs) | measured |
| Prefill, IQ2_XXS / IQ2_XS / IQ3_XXS | integer dot MMQ shaders | tests pass, not benchmarked yet |
| Prefill, IQ1_S / IQ1_M | integer dot MMQ shaders (`8 * (grid + delta)` is an exact int8); UD-IQ2_XXS quants also contain many IQ1_S tensors | tests pass, not benchmarked yet |
| Qwen3.5/3.6 (Gated DeltaNet) | `gated_delta_net` variant that shares each token's k/q between 16 subgroups via shared memory (Intel only) | measured |
| Token generation, IQ2_S / IQ3_S / IQ4_NL | integer dot mat-vec (MMVQ) shaders, incl. MoE (`mul_mat_vec_id`) | measured on Lunar Lake |
| Token generation, Xe2 + Windows driver | MMVQ for every type that has it (upstream: Q2_K/Q3_K/Q6_K only), tuned rows per workgroup (Q6_K: 4) | measured on Lunar Lake |
| Token generation, IQ2_S / IQ3_S | grid tables kept bit-packed in shared memory in MMVQ (IQ2_S 8 KiB -> 2 KiB, IQ3_S 2 KiB -> 1 KiB): the table is copied once per workgroup, and token generation launches many small workgroups | measured on Lunar Lake |
| Token generation, Qwen3.5/3.6 and other delta-net models | when a ubatch reads its single recurrent state from the cell it writes back to (one sequence), use a view of that cell instead of gathering a copy with `GET_ROWS` (graph change, all backends) | measured on Lunar Lake |
| Token generation, MMVQ | optional lane split: several rows per subgroup, a few lanes per row (`GGML_VK_MMVQ_LANES`); on by default only for IQ3_S MoE on Xe2 (8 lanes per row) | measured on Lunar Lake |

Upstream already has integer dot MMQ for Q2_K..Q6_K, Q4_0/Q4_1/Q5_x/Q8_0, IQ3_S, IQ4_XS and MXFP4.

## Measured results

### Prefill, Meteor Lake

Intel Core Ultra (Meteor Lake), Arc iGPU, DDR5-5600 x2, Windows, `-ngl 99`.

| Model | Test | Upstream | This fork |
|---|---|---|---|
| Qwen3.6-35B-A3B UD-IQ3_S | pp4096, `-ub 4096` | 79.0 t/s | 144.8 t/s |
| Gemma-4 26B-A4B UD-IQ4_NL | pp512, `-ub 512` | 69.0 t/s | 90.9 t/s |
| Gemma-4 26B-A4B UD-IQ4_NL | pp2048, `-ub 2048` | 80.4 t/s | 101.4 t/s |

### Token generation, Lunar Lake

Intel Core Ultra (Lunar Lake), Arc 140V, LPDDR5X, Windows, `llama-bench -p 0 -n 128 -r 3`.

| Model | Test | Upstream | This fork |
|---|---|---|---|
| Qwen3.6-35B-A3B UD-IQ3_S | tg128 | 23.8 t/s (23.4-24.0) | 28.5 t/s (28.0-28.7) |
| Qwen3.6-35B-A3B UD-IQ3_S | pp512 | 573.9-592.8 t/s | 613.3-618.4 t/s |

tg128: upstream 5a4afd8 vs this fork eaf44b1, 5 alternating runs each, mean and range: +20%.
pp512 was measured with an earlier build of this fork (before the MMVQ grid packing and Q6_K changes).

Run-to-run variation on this laptop is about 5-10% (power and thermal state) and differs between sessions, so each step below was compared
against the step before it in the same session, and the per-op timings (`GGML_VK_PERF_LOGGER=1`) were checked to
move where expected.

| Step | Model | Before | After | Where it comes from |
|---|---|---|---|---|
| packed IQ grid tables | Qwen3.6-35B-A3B UD-IQ3_S | 23.0 t/s | 23.9 t/s | IQ2_S MoE gate/up 93.5 -> 74.9 us per call |
| packed IQ grid tables (+ rows per workgroup) | Qwen3.6-35B-A3B UD-IQ2_XXS | 26.1 t/s | 28.4 t/s | IQ2_S MoE down 126 -> 69 us per call |
| Q6_K: 4 rows per workgroup | Qwen3.6-35B-A3B UD-IQ3_S | 23.9 t/s | 24.8 t/s | dense Q6_K is about half of the token time in this quant |
| IQ3_S MoE: 8 lanes per row | Gemma-4 26B-A4B UD-IQ4_NL | 20.2 t/s | 21.0 t/s | IQ3_S gate_up (k=2816) 300 -> 245 us per call |
| recurrent state read in place | Qwen3.6-35B-A3B UD-IQ3_S | 28.85 t/s | 30.24 t/s | 60 of 62 `GET_ROWS` per token gone (1.8 ms); `GATED_DELTA_NET` +0.3 ms since the state no longer comes from cache; 5 alternating runs each |

The absolute numbers in this table are lower than the alternating runs above because they were taken in
other sessions; compare within a row only.

Other models on the current build, for reference: GLM-4.7-Flash UD-Q3_K_XL 25.3-25.7 t/s,
LFM2 MoE 24B-A2B Q4_K_S 48.3 t/s.

Token generation on Meteor Lake is not measured yet.

### What the profiles say (Lunar Lake, tg)

- Large dense mat-vecs (lm_head, big FFNs) reach about 100 GB/s of the ~136 GB/s theoretical bandwidth.
  Dense models are close to the limit; only fewer bytes per weight make them faster.
- GPU time is 96-97% of wall time, so host-side submission is not the bottleneck.
- Small MoE mat-vecs with IQ quants are limited by the cost of decoding each weight, not by bandwidth
  (about 100-115 G weights/s for IQ2_S/IQ3_S experts against about 150-230 for K-quants).
- Lane split helps IQ3_S but makes IQ4_NL slower (k=704 down: 202 -> 245 us at 4 lanes, 349 us at 2),
  probably because neighbouring lanes then read different rows.
- `MUL_MAT_ADD` timings in the perf logger sometimes jump 2-3x without any change in wall time; treat that line with care.

### Knobs for experiments (Vulkan, MMVQ)

- `GGML_VK_FORCE_MMVQ=1` / `GGML_VK_DISABLE_MMVQ=1`: always / never use the integer dot mat-vec path
- `GGML_VK_XE2_MMVQ_LEGACY=1`: upstream MMVQ selection and launch shapes on Xe2
- `GGML_VK_MMVQ_ROWS`, `GGML_VK_MMVQ_ID_ROWS`: rows per workgroup, e.g. `4` or `q3_K=4,iq3_s=16`
- `GGML_VK_MMVQ_WG`, `GGML_VK_MMVQ_ID_WG`: `subgroup` or `large`, same syntax
- `GGML_VK_MMVQ_LANES`, `GGML_VK_MMVQ_ID_LANES`: lanes per row in the single-subgroup MMVQ shader, e.g. `iq3_s=4`
  (default: the whole subgroup works on one row, except IQ3_S MoE on Xe2 which uses 8; `iq3_s=16` turns that off).
  Needs subgroup shuffle; lanes x values per lane must be a multiple of 32, other values are ignored.
- `GGML_VK_MMVQ_SG`: subgroup size for the MMVQ pipelines (Intel), e.g. `16`
- `GGML_VK_MMVQ_DEBUG=1`: log once per shape whether a mat-vec took the MMVQ path and why
- `GGML_VK_MMVQ_IQ_SHMEM=1`: IQ2_S / IQ3_S MMVQ copy the (bit-packed) grid table into shared memory per workgroup instead of reading it from a storage buffer

PowerShell keeps environment variables for the whole session: remove them (`Remove-Item Env:NAME`) before
measuring a baseline.

## Recommended settings on iGPUs

- MoE models: use a large ubatch, e.g. `-b 4096 -ub 4096` (Qwen3.6-35B-A3B) or `-b 2048 -ub 2048` (Gemma-4 26B-A4B).
  More tokens per batch means more tokens per expert and better use of each matmul tile.
- Gemma-4 on Meteor Lake: `-fa off` was faster than flash attention (no KV cache quantization then).

## Correctness

Every new shader is checked with `test-backend-ops` (`MUL_MAT`, `MUL_MAT_ID`, `GATED_DELTA_NET`) against the CPU backend,
and each check was confirmed to fail when the shader is deliberately broken.
The packed grid tables were also checked offline against all entries of the original tables, since a one-step error in
a single level (42 instead of 43) stays inside the test tolerance.
Run them on your own GPU before trusting numbers:

```
test-backend-ops -o MUL_MAT    -b Vulkan0 -p "type_a=iq(1_s|1_m|2_xxs|2_xs|2_s|3_xxs|3_s|4_nl)"
test-backend-ops -o MUL_MAT_ID -b Vulkan0 -p "type_a=iq(1_s|1_m|2_xxs|2_xs|2_s|3_xxs|3_s|4_nl)"
test-backend-ops -o GATED_DELTA_NET -b Vulkan0
```

To check a lane split setting, run `MUL_MAT_ID` with the same `GGML_VK_MMVQ_ID_LANES` value.

On GPUs with matrix cores (e.g. Lunar Lake / Xe2) upstream uses a different matmul path, so the new MMQ shaders are
not used there. Set `GGML_VK_DISABLE_COOPMAT=1` to exercise them anyway.

## Roadmap

- Qwen3.5/3.6: the new recurrent state is still written to the op output and then copied into the cache (`CPY`),
  about 0.6-0.8 ms per token on Lunar Lake. Writing it straight into the cache needs a change to the op.
- IQ2_S / IQ3_S expert mat-vecs: fewer instructions per weight in the decode (the remaining limit after the table work).
- MMVQ for IQ2_XXS / IQ2_XS / IQ3_XXS (Qwen3.6 UD-IQ2_XXS / IQ3_XXS experts still use the float path).
- IQ3_S and K-quant MoE matmuls are about 2.7x slower than IQ4_NL on Meteor Lake; suspected register spills at SIMD16.
- Check MMVQ on Meteor Lake (`GGML_VK_FORCE_MMVQ=1`).
- Lunar Lake (Xe2): integer XMX (coopmat int8) path for prefill, which upstream enables only on AMD.

## Keeping up with upstream

```
git fetch upstream
git rebase upstream/master
```
