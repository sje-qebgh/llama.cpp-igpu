# llama.cpp for integrated GPUs (Vulkan)

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) tuned for **integrated GPUs without
matrix cores** (Intel Meteor Lake / Arc Xe-LPG; Lunar Lake / Xe2 for token generation), with a focus on **low-bit quants (Q2-Q4, IQ\*)**
that people actually run on laptops: MoE models at 2-4 bits per weight.

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
| Token generation, Xe2 + Windows driver | MMVQ for every type that has it (upstream: Q2_K/Q3_K/Q6_K only), tuned rows per workgroup (incl. 4 rows for dense Q6_K) | measured on Lunar Lake |
| Token generation, IQ2_S / IQ3_S | grid tables bit-packed in shared memory for MMVQ (8 KiB -> 2 KiB, 2 KiB -> 1 KiB), cheaper per-workgroup copy | measured on Lunar Lake |

Upstream already has integer dot MMQ for Q2_K..Q6_K, Q4_0/Q4_1/Q5_x/Q8_0, IQ3_S, IQ4_XS and MXFP4.

## Measured results

Intel Core Ultra (Meteor Lake), Arc iGPU, DDR5-5600 x2, Windows, `-ngl 99`.

| Model | Test | Upstream | This fork |
|---|---|---|---|
| Qwen3.6-35B-A3B UD-IQ3_S | pp4096, `-ub 4096` | 79.0 t/s | 144.8 t/s |
| Gemma-4 26B-A4B UD-IQ4_NL | pp512, `-ub 512` | 69.0 t/s | 90.9 t/s |
| Gemma-4 26B-A4B UD-IQ4_NL | pp2048, `-ub 2048` | 80.4 t/s | 101.4 t/s |

Intel Core Ultra (Lunar Lake), Arc 140V, LPDDR5X, Windows, `llama-bench` defaults.

| Model | Test | Upstream | This fork |
|---|---|---|---|
| Qwen3.6-35B-A3B UD-IQ3_S | tg128 | 23.8 t/s (23.4-24.0) | 28.5 t/s (28.0-28.7) |
| Qwen3.6-35B-A3B UD-IQ3_S | pp512 | 573.9-592.8 t/s | 613.3-618.4 t/s |

tg128: upstream 5a4afd8 vs this fork eaf44b1, 5 alternating runs each (`-p 0 -n 128 -r 3`), mean and range: +20%.
pp512 was measured with an earlier build of this fork (before the MMVQ grid packing and Q6_K changes).

Token generation on Meteor Lake is not measured yet.

### Knobs for experiments (Vulkan, MMVQ)

- `GGML_VK_FORCE_MMVQ=1` / `GGML_VK_DISABLE_MMVQ=1`: always / never use the integer dot mat-vec path
- `GGML_VK_XE2_MMVQ_LEGACY=1`: upstream MMVQ selection and launch shapes on Xe2
- `GGML_VK_MMVQ_ROWS`, `GGML_VK_MMVQ_ID_ROWS`: rows per workgroup, e.g. `4` or `q3_K=4,iq3_s=16`
- `GGML_VK_MMVQ_WG`, `GGML_VK_MMVQ_ID_WG`: `subgroup` or `large`, same syntax
- `GGML_VK_MMVQ_SG`: subgroup size for the MMVQ pipelines (Intel), e.g. `16`
- `GGML_VK_MMVQ_DEBUG=1`: log once per shape whether a mat-vec took the MMVQ path and why

## Recommended settings on iGPUs

- MoE models: use a large ubatch, e.g. `-b 4096 -ub 4096` (Qwen3.6-35B-A3B) or `-b 2048 -ub 2048` (Gemma-4 26B-A4B).
  More tokens per batch means more tokens per expert and better use of each matmul tile.
- Gemma-4 on Meteor Lake: `-fa off` was faster than flash attention (no KV cache quantization then).

## Correctness

Every new shader is checked with `test-backend-ops` (`MUL_MAT`, `MUL_MAT_ID`, `GATED_DELTA_NET`) against the CPU backend,
and each check was confirmed to fail when the shader is deliberately broken.
Run them on your own GPU before trusting numbers:

```
test-backend-ops -o MUL_MAT    -b Vulkan0 -p "type_a=iq(1_s|1_m|2_xxs|2_xs|2_s|3_xxs|4_nl)"
test-backend-ops -o MUL_MAT_ID -b Vulkan0 -p "type_a=iq(1_s|1_m|2_xxs|2_xs|2_s|3_xxs|4_nl)"
test-backend-ops -o GATED_DELTA_NET -b Vulkan0
```

On GPUs with matrix cores (e.g. Lunar Lake / Xe2) upstream uses a different matmul path, so the new MMQ shaders are
not used there. Set `GGML_VK_DISABLE_COOPMAT=1` to exercise them anyway.

## Roadmap

- IQ3_S and K-quant MoE matmuls are about 2.7x slower than IQ4_NL on Meteor Lake; suspected register spills at SIMD16.
- Token generation is still far from memory bandwidth limits. Next: MMVQ for IQ2_XXS / IQ2_XS / IQ3_XXS,
  check MMVQ on Meteor Lake (`GGML_VK_FORCE_MMVQ=1`), then dispatch overhead.
- Lunar Lake (Xe2): integer XMX (coopmat int8) path, which upstream enables only on AMD.

## Keeping up with upstream

```
git fetch upstream
git rebase upstream/master
```
