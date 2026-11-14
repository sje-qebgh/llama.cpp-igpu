# llama.cpp for integrated GPUs (Vulkan)

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) tuned for **integrated GPUs without
matrix cores** (first target: Intel Meteor Lake / Arc Xe-LPG), with a focus on **low-bit quants (Q2-Q4, IQ\*)**
that people actually run on laptops: MoE models at 2-4 bits per weight.

Everything here is in the Vulkan backend. Other backends are untouched.

## What is different from upstream

| Area | Change | Status |
|---|---|---|
| Prefill, IQ2_S | integer dot (q8_1) MMQ shaders for mul_mat and mul_mat_id | measured |
| Prefill, IQ4_NL | integer dot MMQ shaders (upstream only had them on coopmat GPUs) | measured |
| Prefill, IQ2_XXS / IQ2_XS / IQ3_XXS | integer dot MMQ shaders | tests pass, not benchmarked yet |
| Qwen3.5/3.6 (Gated DeltaNet) | `gated_delta_net` variant that shares each token's k/q between 16 subgroups via shared memory (Intel only) | measured |

Upstream already has integer dot MMQ for Q2_K..Q6_K, Q4_0/Q4_1/Q5_x/Q8_0, IQ3_S, IQ4_XS and MXFP4.

## Measured results

Intel Core Ultra (Meteor Lake), Arc iGPU, DDR5-5600 x2, Windows, `-ngl 99`.

| Model | Test | Upstream | This fork |
|---|---|---|---|
| Qwen3.6-35B-A3B UD-IQ3_S | pp4096, `-ub 4096` | 79.0 t/s | 144.8 t/s |
| Gemma-4 26B-A4B UD-IQ4_NL | pp512, `-ub 512` | 69.0 t/s | 90.9 t/s |
| Gemma-4 26B-A4B UD-IQ4_NL | pp2048, `-ub 2048` | 80.4 t/s | 101.4 t/s |

Token generation speed is unchanged so far.

## Recommended settings on iGPUs

- MoE models: use a large ubatch, e.g. `-b 4096 -ub 4096` (Qwen3.6-35B-A3B) or `-b 2048 -ub 2048` (Gemma-4 26B-A4B).
  More tokens per batch means more tokens per expert and better use of each matmul tile.
- Gemma-4 on Meteor Lake: `-fa off` was faster than flash attention (no KV cache quantization then).

## Correctness

Every new shader is checked with `test-backend-ops` (`MUL_MAT`, `MUL_MAT_ID`, `GATED_DELTA_NET`) against the CPU backend,
and each check was confirmed to fail when the shader is deliberately broken.
Run them on your own GPU before trusting numbers:

```
test-backend-ops -o MUL_MAT    -b Vulkan0 -p "type_a=iq(2_xxs|2_xs|2_s|3_xxs|4_nl)"
test-backend-ops -o MUL_MAT_ID -b Vulkan0 -p "type_a=iq(2_xxs|2_xs|2_s|3_xxs|4_nl)"
test-backend-ops -o GATED_DELTA_NET -b Vulkan0
```

On GPUs with matrix cores (e.g. Lunar Lake / Xe2) upstream uses a different matmul path, so the new MMQ shaders are
not used there. Set `GGML_VK_DISABLE_COOPMAT=1` to exercise them anyway.

## Roadmap

- IQ3_S and K-quant MoE matmuls are about 2.7x slower than IQ4_NL on Meteor Lake; suspected register spills at SIMD16.
- Token generation is far from memory bandwidth limits (about 1/3 of it). Profile, then add integer dot mat-vec shaders
  for IQ2_*/IQ3_*/IQ4_NL or reduce dispatch overhead.
- Lunar Lake (Xe2): integer XMX (coopmat int8) path, which upstream enables only on AMD.

## Keeping up with upstream

```
git fetch upstream
git rebase upstream/master
```
