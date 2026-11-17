IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## This repository is a fork: llama.cpp-igpu

AGENTS.md and CONTRIBUTING.md are upstream's rules for contributions to ggml-org/llama.cpp. Their rules on commits, commit messages, pushes and PRs do not apply to work in this fork. Follow them only if something is prepared for upstream. The code rules in AGENTS.md still apply here: ASCII only, short comments, match the surrounding code.

- Read README-igpu.md and `git log` before starting. README-igpu.md lists every change of the fork, the measured results and the experiment knobs.
- Branch `igpu` is the default. Never push to `master`, it tracks upstream.
- The assistant may write commit messages and commit or push to `igpu` once the user agreed to the change. Style: like the existing `git log` (subject `vulkan: ...`, body with what was measured).
- Before history is rewritten (e.g. rebase onto upstream), push a backup branch `igpu-backup-YYYYMMDD` and push `igpu` with `--force-with-lease`.
- No PRs to upstream. Upstream does not accept AI-written PR descriptions or commit messages.
- Code only for measurements (probe switches, debug prints) is not committed.
- When a change lands, update the table and the results in README-igpu.md.

### Testing

- Run `test-backend-ops` (MUL_MAT, MUL_MAT_ID, and the ops touched) on Vulkan with lavapipe. Set `GGML_VK_VISIBLE_DEVICES=0`, otherwise CPU devices are skipped.
- lavapipe does not report integer dot product as accelerated, so the MMQ/MMVQ paths are off. For tests only, remove the `integerDotProduct4x8BitPackedSignedAccelerated` checks in ggml-vulkan.cpp. Never commit this.
- Check every new shader twice: the tests pass, and a deliberate error in the shader makes them fail.
- A packed or rewritten lookup table is also compared offline against all entries of the original, because a small error can stay inside the test tolerance.
- Old glslc (e.g. Ubuntu 24.04) does not support GL_EXT_integer_dot_product. Use a recent glslang/shaderc.

### Performance

- Speed is measured by the user on Intel iGPUs with the Windows driver: Meteor Lake (Xe-LPG, no matrix cores, DP4a) and Lunar Lake (Xe2, XMX). Give the user a PowerShell script or exact commands.
- Run-to-run noise is 5-10%. Compare settings in alternating runs, several rounds each.
- PowerShell keeps environment variables: clear all `GGML_VK_*` before a baseline run.
- Do not compare quant types by the t/s of different models (experts, sizes and shapes differ). Compare within one model: per-op times by shape, dense vs MoE.
- `GGML_VK_PERF_LOGGER=1` gives per-op times but slows down the run; compare t/s only without it. Its `MUL_MAT_ADD` lines sometimes jump 2-3x and do not match wall time.
