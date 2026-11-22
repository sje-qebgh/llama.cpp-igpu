"""Release charts for llama.cpp-igpu. Numbers are copied from README-igpu.md; each panel compares
numbers from one table (same machine, same session)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.transforms import ScaledTranslation
from matplotlib.patches import FancyBboxPatch, Patch

for f in font_manager.findSystemFonts():
    if "/inter/" in f.lower():
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "Inter", "font.size": 10,
    "axes.edgecolor": "#d9d8d3", "axes.linewidth": 1,
    "xtick.color": "#52514e", "ytick.color": "#0b0b0b",
    "text.color": "#0b0b0b",
})
SURFACE = "#fcfcfb"
GRID = "#e8e7e2"
INK2 = "#52514e"
PAD = 40
SUB_Y = 1.11
BASE = "#9b9a94"    # upstream / before (de-emphasized)
FORK = "#2a78d6"    # this fork
REF = "#1baf7a"     # f16 reference


def rounded_barh(ax, y, w, h, color):
    # square at the baseline, 4px-ish rounded data end
    r = min(h * 0.35, w * 0.5)
    ax.add_patch(FancyBboxPatch((0, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=1, linewidth=0, facecolor=color))
    ax.add_patch(plt.Rectangle((0, y - h / 2), min(w, r * 2), h, linewidth=0, facecolor=color))


def panel(ax, title, subtitle, models, series, unit="t/s", gain_from=0, gain_to=-1):
    """series: list of (label, color, values). Gain label compares series[gain_to] with series[gain_from]."""
    n = len(series)
    bar_h = 0.8 / n * 0.82
    slot = 0.8 / n
    xmax = max(max(v for v in s[2] if v is not None) for s in series)
    ax.set_xlim(0, xmax * 1.28)
    ax.set_ylim(len(models) - 0.5, -0.5)
    for i, m in enumerate(models):
        for j, (lab, col, vals) in enumerate(series):
            v = vals[i]
            if v is None:
                continue
            y = i - 0.4 + slot * (j + 0.5)
            rounded_barh(ax, y, v, bar_h, col)
            txt = f"{v:.1f}"
            if j == (gain_to % n) and series[gain_from][2][i]:
                g = v / series[gain_from][2][i] - 1
                if abs(g) >= 0.03:
                    txt += f"  ({g:+.0%})"
            ax.text(v + xmax * 0.012, y, txt, va="center", ha="left", fontsize=9,
                    color="#0b0b0b" if j == (gain_to % n) else INK2,
                    fontweight="semibold" if j == (gain_to % n) else "normal")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=0, labelsize=8.5)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(unit, color=INK2, fontsize=9)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold", pad=PAD)
    fig = ax.figure
    ax.text(0, 1, subtitle, transform=ax.transAxes + ScaledTranslation(0, 22 / 72, fig.dpi_scale_trans),
            fontsize=8.5, color=INK2, va="bottom")
    ax.legend(bbox_transform=ax.transAxes + ScaledTranslation(0, 2 / 72, fig.dpi_scale_trans), handles=[Patch(color=c, label=l) for l, c, _ in series], loc="lower left",
              bbox_to_anchor=(-0.01, 1.0), frameon=False, fontsize=8.5, ncol=len(series),
              handlelength=1.2, borderaxespad=0.2, columnspacing=1.6)


def save(fig, name, note):
    fig.text(0.01, 0.01, note, fontsize=7.5, color=INK2, ha="left", va="bottom")
    fig.savefig(name, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------- Meteor Lake ----------------
# bench-compare.ps1, 2026-11-22 (second run): Core Ultra 5 135H, driver 32.0.101.9028, upstream 8d60873 vs fork 9db50c9,
# 3 alternating rounds at depth 0 (2 at depth 16384), medians.
MTL_MODELS = ["Qwen3.6-35B-A3B UD-IQ2_XXS", "Qwen3.6-35B-A3B UD-IQ3_S", "Qwen3-Coder-30B-A3B UD-Q2_K_XL",
              "Gemma-4 26B-A4B UD-IQ4_NL", "gpt-oss-20b UD-Q4_K_XL"]
fig, axs = plt.subplots(3, 1, figsize=(8.2, 11.2), facecolor=SURFACE,
                        gridspec_kw={"height_ratios": [5, 5, 2.2], "hspace": 0.75})
panel(axs[0], "Prefill",
      "pp512, llama-bench defaults · 3 alternating rounds, median",
      MTL_MODELS,
      [("upstream 8d60873", BASE, [68.2, 77.8, 33.9, 86.9, 303.7]),
       ("this fork 9db50c9", FORK, [229.3, 209.6, 71.6, 187.9, 331.9])])
panel(axs[1], "Token generation",
      "tg128, llama-bench defaults · 3 alternating rounds, median",
      MTL_MODELS,
      [("upstream 8d60873", BASE, [15.0, 14.9, 18.7, 14.5, 20.6]),
       ("this fork 9db50c9", FORK, [23.7, 19.3, 31.4, 16.0, 21.7])])
panel(axs[2], "Long context: 16k tokens in the KV cache",
      "Qwen3.6-35B-A3B UD-IQ3_S, -d 16384 -fa 1 -ctk q8_0 -ctv q8_0 · 2 alternating rounds",
      ["pp512", "tg128"],
      [("upstream 8d60873", BASE, [48.3, 11.8]),
       ("this fork 9db50c9", FORK, [109.5, 16.8])])
fig.suptitle("Meteor Lake (Core Ultra 5 135H, Arc Xe-LPG iGPU) · llama.cpp-igpu", x=0.01, ha="left",
             fontsize=14, fontweight="bold", y=0.985)
fig.subplots_adjust(left=0.34, right=0.97, top=0.885, bottom=0.075)
save(fig, "igpu-meteor-lake.png",
     "Windows, Vulkan, -ngl 99, driver 32.0.101.9028, Dell Optimizer Ultra Performance, on AC. All numbers from one session (2026-11-22).\n"
     "Run-to-run noise is 5-10%. Details: README-igpu.md")

# ---------------- Lunar Lake ----------------
# bench-compare.ps1, 2026-11-22: Core Ultra 7 268V, driver 32.0.101.9028,
# upstream 8d60873 vs fork 63cce34, 3 alternating rounds at depth 0 (2 at depth 16384), medians.
LNL_MODELS = ["Qwen3.6-35B-A3B UD-IQ2_XXS", "Qwen3.6-35B-A3B UD-IQ3_S", "Qwen3-Coder-30B-A3B UD-Q2_K_XL",
              "Gemma-4 26B-A4B UD-IQ4_NL", "gpt-oss-20b UD-Q4_K_XL"]
fig, axs = plt.subplots(3, 1, figsize=(8.2, 11.2), facecolor=SURFACE,
                        gridspec_kw={"height_ratios": [5, 5, 2.2], "hspace": 0.75})
panel(axs[0], "Prefill",
      "pp512, llama-bench defaults · 3 alternating rounds, median",
      LNL_MODELS,
      [("upstream 8d60873", BASE, [650.3, 621.1, 854.9, 822.2, 982.3]),
       ("this fork 63cce34", FORK, [653.0, 630.3, 855.8, 824.7, 975.9])])
panel(axs[1], "Token generation",
      "tg128, llama-bench defaults · 3 alternating rounds, median",
      LNL_MODELS,
      [("upstream 8d60873", BASE, [25.7, 23.4, 41.8, 21.7, 31.4]),
       ("this fork 63cce34", FORK, [36.5, 32.5, 46.7, 24.4, 32.9])])
panel(axs[2], "Long context: 16k tokens in the KV cache",
      "Qwen3.6-35B-A3B UD-IQ3_S, -d 16384 -fa 1 -ctk q8_0 -ctv q8_0 · 2 alternating rounds",
      ["pp512", "tg128"],
      [("upstream 8d60873", BASE, [164.2, 18.3]),
       ("this fork 63cce34", FORK, [446.1, 28.4])])
fig.suptitle("Lunar Lake (Core Ultra 7 268V, Arc 140V) · llama.cpp-igpu", x=0.01, ha="left",
             fontsize=14, fontweight="bold", y=0.985)
fig.subplots_adjust(left=0.34, right=0.97, top=0.885, bottom=0.075)
save(fig, "igpu-lunar-lake.png",
     "Windows, Vulkan, -ngl 99, driver 32.0.101.9028, Windows power mode best performance, on AC. One session (2026-11-22).\n"
     "Run-to-run noise is 5-10%. Details: README-igpu.md")
