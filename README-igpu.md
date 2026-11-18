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
| Prefill, IQ2_XXS / IQ2_XS / IQ3_XXS | integer dot MMQ shaders | measured (Qwen3.6 UD-IQ2_XXS) |
| Prefill, MoE (`mul_mat_id` integer dot MMQ) | a tile covers up to BN rows of one expert; columns past the rows the expert has are no longer loaded or multiplied. With 128-256 experts most of a 64-column tile is empty at pp512 | measured on Meteor Lake |
| Prefill, IQ1_S / IQ1_M | integer dot MMQ shaders (`8 * (grid + delta)` is an exact int8); UD-IQ2_XXS quants also contain many IQ1_S tensors | tests pass on Meteor Lake |
| Qwen3.5/3.6 (Gated DeltaNet) | `gated_delta_net` variant that shares each token's k/q between 16 subgroups via shared memory (Intel only) | measured |
| Qwen3.5/3.6 prefill, Xe2 | that variant loads k/q/g/beta of 8 tokens per shared memory block, so the workgroup waits once per 8 tokens instead of every token; calls with more than 1024 tokens keep 1 token per block (slower with 8). `GGML_VK_GDN_TB=N` sets the block size | measured on Lunar Lake |
| Token generation, IQ2_S / IQ3_S / IQ4_NL | integer dot mat-vec (MMVQ) shaders, incl. MoE (`mul_mat_vec_id`) | measured on Lunar Lake |
| Token generation, IQ2_XXS / IQ2_XS / IQ3_XXS | integer dot mat-vec (MMVQ) shaders, incl. MoE; grid from the storage buffer like IQ2_S / IQ3_S | measured on Lunar Lake |
| Token generation, Xe2 + Windows driver | MMVQ for every type that has it (upstream: Q2_K/Q3_K/Q6_K only), tuned rows per workgroup (Q6_K: 4) | measured on Lunar Lake |
| Token generation, IQ2_S / IQ3_S | MMVQ reads the grid tables from a small storage buffer instead of copying them into shared memory in every workgroup (token generation launches many small workgroups); `GGML_VK_MMVQ_IQ_SHMEM=1` selects the shared memory copy, kept bit-packed (IQ2_S 8 KiB -> 2 KiB, IQ3_S 2 KiB -> 1 KiB) | measured on Lunar Lake |
| Token generation, Qwen3.5/3.6 and other delta-net models | when a ubatch reads its single recurrent state from the cell it writes back to (one sequence), use a view of that cell instead of gathering a copy with `GET_ROWS` (graph change, all backends) | measured on Lunar Lake |
| Token generation, Xe-LPG iGPU + Windows driver | MMVQ per type: Q2_K, Q3_K, Q4_K, Q5_K, IQ2_S, IQ4_XS, and IQ3_S for k < 2048; fp16 for the rest (upstream: fp16 for every type) | measured on Meteor Lake |
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

Core Ultra 5 135H, driver 32.0.101.8991, upstream 9fba51d vs this fork b9f1952, `-fa 0 -r 2`, 2 alternating runs (mean):

| Model | pp512 (`-ub 512`) upstream / fork | pp2048 (`-ub 2048`) upstream / fork |
|---|---|---|
| Qwen3.6-35B-A3B UD-IQ2_XXS | 71.4 / 152.5 (+114%) | 95.0 / 197.3 (+108%) |
| Qwen3.6-35B-A3B UD-IQ3_S | 74.6 / 123.2 (+65%) | 102.5 / 183.9 (+79%) |
| Gemma-4 26B-A4B UD-IQ4_NL | 83.1 / 114.7 (+38%) | 101.6 / 128.3 (+26%) |
| gpt-oss-20b UD-Q4_K_XL (MXFP4 experts) | 298.9 / 298.2 | 234.8 / 237.9 |
| Qwen3-Coder-30B-A3B UD-Q2_K_XL (K-quant experts) | 35.1 / 35.1 | 49.8 / 49.4 |

All `test-backend-ops` checks pass on this GPU (new IQ MMQ types, `GATED_DELTA_NET`, MMVQ forced on).

MoE prefill, skipping the unused columns of a tile (later session; the machine was slower than above, so compare
within the table only). `-fa 0 -r 1`, 2 alternating runs (mean):

| Model | pp512 before / after | pp2048 before / after |
|---|---|---|
| Qwen3-Coder-30B-A3B UD-Q2_K_XL (128 experts) | 31.6 / 67.3 (+113%) | 44.3 / 55.0 (+24%) |
| Qwen3.6-35B-A3B UD-IQ3_S (256 experts) | 80.0 / 145.5 (+82%) | 122.7 / 153.2 (+25%) |

Skipping only the dot products gave about half of this (59.6 and 121.8 t/s at pp512). Forcing the smaller tile
(32 columns instead of 64) instead was faster for Q2_K (pp512 52.5) but slower for every other model tested
(Qwen3.6 IQ3_S / IQ2_XXS -20 to -34%, gpt-oss-20b -11%), and slower than column skipping with the 64-column tile.

### Prefill, Lunar Lake

Qwen3.6-35B-A3B UD-IQ3_S, Gated DeltaNet tokens per block (`GGML_VK_GDN_TB`), t/s, 2 alternating runs of `-r 1`,
flash attention on (default). llama.cpp splits prompts into ubatches of 512 tokens by default (`-ub`).

| Tokens per block | pp512 | pp2048 `-ub 512` | pp2048 `-ub 1024` | pp2048 `-ub 2048` |
|---|---|---|---|---|
| 1 (before) | 587.8 | 597.2 | 692.9 | 670.6 |
| 4 | 625.6 | 618.3 | 690.7 | 580.1 |
| 8 (new default up to 1024 tokens per call) | 639.9 (+9%) | 630.8 (+6%) | 710.6 | 565.3 |

GATED_DELTA_NET per call at 512 tokens: 4.67 ms (1 token per block) -> 3.74 ms (8). With 2048 tokens per call it
gets slower (28.0 -> 45.8 ms), so calls above 1024 tokens keep 1 token per block. tg128 does not change (32.1 / 32.1).
`-ub 1024` is the fastest ubatch size here for long prompts (about +13% over the default 512).

On Meteor Lake the same blocks do not help (t/s, same test, `GGML_VK_GDN_TB`; pp512 at TB 1 had one slow run):

| Tokens per block | pp512 | pp2048 `-ub 512` | pp2048 `-ub 1024` | pp2048 `-ub 2048` |
|---|---|---|---|---|
| 1 (default there) | 137.5 | 142.3 | 154.7 | 159.6 |
| 2 | 153.0 | 139.1 | 151.0 | 147.6 |
| 4 | 149.2 | 142.1 | 150.3 | 152.0 |
| 8 | 132.9 | 138.2 | 140.3 | 146.9 |

The MoE prefill tile (`mul_mm.comp`, coopmat) was checked too: the default (large, 128 columns) is the fastest for
Qwen3.6 IQ3_S, Gemma-4 IQ4_NL and gpt-oss-20b; forcing 32 or 64 columns is 20-70% slower.

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
| IQ grid tables in a storage buffer (vs bit-packed shared memory) | Qwen3.6-35B-A3B UD-IQ3_S | 30.24 t/s | 31.39 t/s | 6 alternating runs each |
| IQ grid tables in a storage buffer | Qwen3.6-35B-A3B UD-IQ2_XXS | 32.97 t/s | 33.80 t/s | |
| IQ grid tables in a storage buffer | Gemma-4 26B-A4B UD-IQ4_NL | 22.07 t/s | 22.79 t/s | |
| MMVQ for IQ2_XXS / IQ2_XS / IQ3_XXS | Qwen3.6-35B-A3B UD-IQ2_XXS | 33.95 t/s (33.7-34.1) | 35.67 t/s (35.2-36.1) | 3 alternating runs each, faster in every round |
| MMVQ for IQ2_XXS / IQ2_XS / IQ3_XXS | Qwen3.6-35B-A3B UD-IQ3_XXS | 31.24 t/s (31.2-31.3) | 31.61 t/s (31.5-31.8) | same; small gain, within the run-to-run spread of other sessions |
| IQ2_S / IQ2_XXS MoE: 4 rows per workgroup (was 16) | Qwen3.6-35B-A3B UD-IQ3_XXS | 31.62 t/s (31.5-31.7) | 32.71 t/s (32.6-32.9) | IQ2_S gate/up 60 -> 50 us per call; 3 alternating runs each, set with `GGML_VK_MMVQ_ID_ROWS` |
| IQ2_S / IQ2_XXS MoE: 4 rows per workgroup (was 16) | Qwen3.6-35B-A3B UD-IQ3_S | 32.03 t/s (32.0-32.1) | 32.86 t/s (32.8-32.9) | same; measured with all IQ2/IQ3 types at 4 rows |
| IQ2_S / IQ2_XXS MoE: 4 rows per workgroup (was 16) | Qwen3.6-35B-A3B UD-IQ2_XXS | 36.21 t/s (35.9-36.6) | 37.05 t/s (36.9-37.2) | IQ2_XXS gate/up 50 -> 40 us per call; measured with all IQ2/IQ3 types at 4 rows |

The absolute numbers in this table are lower than the alternating runs above because they were taken in
other sessions; compare within a row only.

Other models on the current build, for reference: GLM-4.7-Flash UD-Q3_K_XL 25.3-25.7 t/s,
LFM2 MoE 24B-A2B Q4_K_S 48.3 t/s.

### Token generation, Meteor Lake

Same machine and builds, tg128, 2 alternating rounds of `-r 3`. The new per-type MMVQ default is not in this build;
"MMVQ forced" (`GGML_VK_FORCE_MMVQ=1`) shows the effect for the models whose MoE experts are all MMVQ-friendly types.

| Model | upstream | fork (fp16 mat-vec) | fork, MMVQ forced |
|---|---|---|---|
| Qwen3-Coder-30B-A3B UD-Q2_K_XL | 18.9 | 18.8 | 30.1 (+60%) |
| Qwen3.6-35B-A3B UD-IQ2_XXS | 15.8 | 16.2 | 23.1 (+46%) |
| Qwen3.6-35B-A3B UD-IQ3_S | 15.7 | 15.7 | 20.4 (+30%) |
| Gemma-4 26B-A4B UD-IQ4_NL | 14.7 | 14.6 | 13.1 (-10%) |
| gpt-oss-20b UD-Q4_K_XL | 20.6 | 20.6 | 16.6 (-19%) |

Per-op timings show which types win: Q8_0, MXFP4 and IQ4_NL mat-vecs are slower with MMVQ on this GPU, so the
per-type default above keeps them on fp16. GPU time per token is at or above wall time, so submission is not the limit.

The per-type default, measured later on the same machine (build 67e7220, other session, so compare within the table only):
it matches the faster of the two other settings for every model.

| Model | fork default (per type) | `GGML_VK_XE1_MMVQ_LEGACY=1` (fp16) | `GGML_VK_FORCE_MMVQ=1` |
|---|---|---|---|
| Qwen3-Coder-30B-A3B UD-Q2_K_XL | 26.3 (25.2-27.4) | 17.0 | 26.9 (26.6-27.3) |
| Qwen3.6-35B-A3B UD-IQ2_XXS | 21.0 (20.4-21.6) | 15.3 | 21.3 (21.0-21.7) |
| Qwen3.6-35B-A3B UD-IQ3_S | 19.0 (18.8-19.3) | 14.4 | 19.2 (18.6-19.7) |
| Gemma-4 26B-A4B UD-IQ4_NL | 14.1 | 14.3 | 12.7 |
| gpt-oss-20b UD-Q4_K_XL | 20.6 | 20.5 | 16.1 |

### What the profiles say (Lunar Lake, tg)

- Large dense mat-vecs (lm_head, big FFNs) reach about 100 GB/s of the ~136 GB/s theoretical bandwidth.
  Dense models are close to the limit; only fewer bytes per weight make them faster.
- GPU time is 96-97% of wall time, so host-side submission is not the bottleneck.
- How the IQ grid table reaches the shader matters: a GLSL constant array indexed per lookup was 2x slower than the
  shared memory copy on the Windows driver, while a storage buffer beat the shared memory copy by about 4%.
- Small MoE mat-vecs with IQ quants are limited by the cost of decoding each weight, not by bandwidth
  (about 100-115 G weights/s for IQ2_S/IQ3_S experts against about 150-230 for K-quants).
- IQ2_S MoE gate/up (m=512, k=2048, 60 us per call): removing parts of the decode in a test build gave 55 us without the sign apply,
  53 without the grid load, 49 when the grid address does not depend on loaded data, and 44 with no decode at all.
  So the chain of dependent loads costs more than the instruction count, and even without decode this shape
  stays at about 60 GB/s. Fewer rows per workgroup helped more than any of these (50 us at 4 rows); lane split did not help.
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
- `GGML_VK_XE1_MMVQ_LEGACY=1`: upstream MMVQ selection on Xe-LPG iGPUs with the Windows driver (fp16 for every type)

### Knobs for experiments (Vulkan, MMQ / prefill, GPUs without matrix cores)

- `GGML_VK_WT_<NAME>="BLOCK,BM,BN,BK,WM,WN,WMITER,TM,TN,TK,WARP"`: matmul warptile, `NAME` one of `L M S`, `L_MMQ M_MMQ`,
  `L_INT M_INT`, `L_INT_K M_INT_K`, and for MoE `ID_L ID_M MMQID_L MMQID_M ID_INT_L ID_INT_M ID_INT_K_L ID_INT_K_M`.
  BM/BN must stay the same; invalid shapes are ignored with a warning. `GGML_VK_WT_PRINT=1` prints the tiles in use.
- `GGML_VK_SUBGROUP_OVERRIDE="matmul=16,!iq3_s_q8_1=8"`: subgroup size by pipeline name substring; `!` also replaces a
  size the backend already pins. Check results with `test-backend-ops` under the same setting.
- `GGML_VK_GDN_TB=N`: Gated DeltaNet (Intel variant) tokens per shared memory block for calls of up to 1024 tokens (Xe2 default 8, others 1)
- `GGML_VK_MMVQ_IQ_SHMEM=1`: IQ2/IQ3 MMVQ copy the (bit-packed) grid table into shared memory per workgroup instead of reading it from a storage buffer

PowerShell keeps environment variables for the whole session: remove them (`Remove-Item Env:NAME`) before
measuring a baseline.

## Recommended settings on iGPUs

- MoE models: use a larger ubatch than the default 512 for long prompts. More tokens per batch means more tokens
  per expert and better use of each matmul tile. Measured with Qwen3.6-35B-A3B UD-IQ3_S, pp2048 (5095e8b):
  Lunar Lake `-ub 512` 631, `-ub 1024` 711, `-ub 2048` 671 t/s; Meteor Lake 142, 155, 160 t/s.
  So `-b 2048 -ub 1024` on Lunar Lake, `-b 2048 -ub 2048` on Meteor Lake (Gemma-4 26B-A4B: `-ub 2048` measured earlier).
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
- IQ2_S / IQ3_S expert mat-vecs: issue the next grid index load before the current grid lookup, to break the chain
  of dependent loads (the largest part of the decode cost in the probe above). Upper bound about 1.5% of tg on Lunar Lake
  (IQ2_S gate/up 50 us per call at 4 rows, about 44 us with no decode at all).
- Meteor Lake: measure the new IQ2_XXS / IQ2_XS / IQ3_XXS MMVQ (off by default there, `GGML_VK_FORCE_MMVQ=1`).
- MoE prefill on Meteor Lake: the slow Qwen3-Coder UD-Q2_K_XL prefill (33-37 t/s at pp512) came from mostly empty
  MoE tiles, not from the K-quant decode: dense Q4_K mat-mul ran at 2000-2600 GFLOPS while the MoE ones ran at 100-290.
  SIMD8 and other MoE tiles did not help. Column skipping (above) fixes most of it; the float / coopmat shader
  (`mul_mm.comp`) already did this upstream, only the integer dot one (`mul_mmq.comp`) did not. Still open: a check of
  Gemma-4 / gpt-oss with it.
- Lunar Lake (Xe2): integer XMX (coopmat int8) path for prefill, which upstream enables only on AMD.

- Gated DeltaNet prefill with more than 1024 tokens per call: 8 tokens per block is slower there (Lunar Lake),
  the cause is not known yet.

### Not planned

- Repacking IQ2/IQ3 blocks at load time into an MMVQ-friendly layout (e.g. the 10-bit IQ2_S grid index in one word
  instead of split over `qs` and `qh`). The Vulkan backend has no repack buffer type, so it would touch tensor upload and
  read-back, size computation, the mmap path and every other shader that reads these blocks, and make rebases harder.
  It only removes the index assembly (about 8 us of 60 per IQ2_S gate/up call in the probe), inside the same
  1.5% upper bound as above.

## Keeping up with upstream

```
git fetch upstream
git rebase upstream/master
```
