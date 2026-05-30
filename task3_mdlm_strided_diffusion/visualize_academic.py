"""Academic-quality visualisation script.

Produces two publication-ready figures:
  1. pca_manifold_alignment.png  — PCA 2-D manifold + K-means clustering
  2. system_stress_benchmark.png — throughput & memory stress benchmark
"""

from __future__ import annotations

import json
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ── 中文字体配置 ──────────────────────────────────────────────────────────────
import matplotlib.font_manager as fm
import platform

def _setup_chinese_font():
    """配置中文字体，按优先级尝试常见中文字体。"""
    candidates = [
        "SimHei", "Microsoft YaHei", "STHeiti", "WenQuanYi Micro Hei",
        "Noto Sans CJK SC", "Source Han Sans CN", "PingFang SC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    # 回退：尝试系统字体路径
    fallback_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    import os
    for p in fallback_paths:
        if os.path.exists(p):
            fe = fm.FontEntry(fname=p, name="CustomChinese")
            fm.fontManager.ttflist.insert(0, fe)
            return "CustomChinese"
    return "DejaVu Sans"  # 最终回退（可能乱码，但不崩溃）

_CN_FONT = _setup_chinese_font()

plt.rcParams.update({
    "font.family":        _CN_FONT,
    "axes.unicode_minus": False,   # 修复负号显示为方块的问题
    "font.size":          13,
    "axes.titlesize":     16,
    "axes.labelsize":     14,
    "axes.linewidth":     1.4,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.major.width":  1.2,
    "ytick.major.width":  1.2,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    11,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#cccccc",
    "lines.linewidth":    2.5,
    "lines.markersize":   8,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

# Palette
REAL_COLOR    = "#2166AC"
GEN_COLOR     = "#D6604D"
CLUSTER_COLOR = "#1A1A1A"

# ── Figure 1: PCA manifold ────────────────────────────────────────────────────

def fig_pca_manifold(out_path: str = "pca_manifold_alignment.png") -> None:
    real = np.load("real_latents.npy").astype(np.float32)
    gen  = np.load("gen_latents.npy").astype(np.float32)

    combined = np.vstack([real, gen])
    pca      = PCA(n_components=2, random_state=42)
    proj     = pca.fit_transform(combined)
    real_2d  = proj[:500]
    gen_2d   = proj[500:]
    var      = pca.explained_variance_ratio_ * 100

    km      = KMeans(n_clusters=3, random_state=42, n_init=20)
    labels  = km.fit_predict(proj)
    centers = km.cluster_centers_

    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.patch.set_facecolor("white")

    cluster_bg = ["#D0E4F7", "#FCE4D6", "#D5ECD4"]
    for k in range(3):
        mask = labels == k
        if mask.sum() == 0:
            continue
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   color=cluster_bg[k], s=90, alpha=0.28, linewidths=0, zorder=1)

    ax.scatter(real_2d[:, 0], real_2d[:, 1],
               c=REAL_COLOR, marker="o", s=45, alpha=0.72,
               linewidths=0.3, edgecolors="white",
               label="真实 DNA 序列", zorder=3)

    ax.scatter(gen_2d[:, 0], gen_2d[:, 1],
               c=GEN_COLOR, marker="X", s=55, alpha=0.72,
               linewidths=0.3, edgecolors="white",
               label="零样本生成 (50步跳步扩散)", zorder=3)

    for i, (cx, cy) in enumerate(centers):
        ax.scatter(cx, cy, marker="*", s=340, c=CLUSTER_COLOR,
                   edgecolors="white", linewidths=1.0, zorder=5)
        ax.annotate(f"$C_{i+1}$",
                    xy=(cx, cy), xytext=(cx + 0.25, cy + 0.25),
                    fontsize=11, fontweight="bold", color=CLUSTER_COLOR,
                    arrowprops=dict(arrowstyle="-", color=CLUSTER_COLOR,
                                   lw=0.8, alpha=0.6),
                    zorder=6)

    def _confidence_ellipse(pts, ax_, n_std=2.0, color="#000000", lw=1.5, ls="--"):
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
        cov  = np.cov(pts[:, 0], pts[:, 1])
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(vals)
        ell = Ellipse(xy=pts.mean(0), width=width, height=height, angle=angle,
                      edgecolor=color, facecolor="none", linewidth=lw,
                      linestyle=ls, alpha=0.85, zorder=4)
        ax_.add_patch(ell)

    _confidence_ellipse(real_2d, ax, color=REAL_COLOR, lw=2.0, ls="--")
    _confidence_ellipse(gen_2d,  ax, color=GEN_COLOR,  lw=2.0, ls=(0, (4, 2)))

    ax.set_xlabel(f"主成分 1  ({var[0]:.1f}% 方差)", labelpad=8)
    ax.set_ylabel(f"主成分 2  ({var[1]:.1f}% 方差)", labelpad=8)
    ax.set_title("潜在空间流形对比\nPCA 投影与 K-均值聚类  (k = 3)",
                 pad=14, fontweight="bold")

    star_patch = mpl.lines.Line2D(
        [], [], marker="*", color="w", markerfacecolor=CLUSTER_COLOR,
        markersize=12, label="聚类质心 ($C_k$)")
    handles, _ = ax.get_legend_handles_labels()
    # 图例置于右下角，避免与左上注释框及数据点重叠
    ax.legend(handles=handles + [star_patch],
              loc="lower right", frameon=True, ncol=1,
              borderpad=0.8, labelspacing=0.5)

    ax.text(0.02, 0.97,
            f"每组 n = 500  ·  PCA 维度 = 2\n"
            f"K-均值 k = 3  ·  虚线 = 2σ 置信椭圆",
            transform=ax.transAxes, fontsize=9.5,
            verticalalignment="top", color="#555555",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.85))

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  已保存 → {out_path}")


# ── Figure 2: Stress benchmark ────────────────────────────────────────────────

def fig_stress_benchmark(out_path: str = "system_stress_benchmark.png") -> None:
    with open("stress_results.json") as f:
        data = json.load(f)

    results = data["results"]

    def collect(label: str):
        rows = [r for r in results if r["model"] == label]
        bs, tps, mem, status = [], [], [], []
        for r in rows:
            bs.append(r["batch_size"])
            tps.append(r.get("tokens_per_sec"))
            mem_val = r.get("memory_mb")
            mem.append(mem_val / 1024 if mem_val is not None else None)
            status.append(r.get("status", "ok"))
        return bs, tps, mem, status

    ar_bs,   ar_tps,   ar_mem,   ar_st   = collect("AR")
    diff_bs, diff_tps, diff_mem, diff_st = collect("Diffusion")

    def split_ok(bs_list, vals, statuses):
        ok_bs, ok_v, bad_bs = [], [], []
        for b, v, s in zip(bs_list, vals, statuses):
            if s == "ok" and v is not None:
                ok_bs.append(b); ok_v.append(v)
            else:
                bad_bs.append(b)
        return ok_bs, ok_v, bad_bs

    ar_ok_bs,   ar_ok_tps,   ar_bad   = split_ok(ar_bs,   ar_tps,   ar_st)
    diff_ok_bs, diff_ok_tps, diff_bad = split_ok(diff_bs, diff_tps, diff_st)

    ar_ok_bs_m,   ar_ok_mem,   _ = split_ok(ar_bs,   ar_mem,   ar_st)
    diff_ok_bs_m, diff_ok_mem, _ = split_ok(diff_bs, diff_mem, diff_st)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 8.5),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
        sharex=True)
    fig.patch.set_facecolor("white")

    AR_C   = "#E84855"
    DIFF_C = "#3A86FF"
    LW     = 2.8
    MS     = 8

    # ── 上图：吞吐量 ──────────────────────────────────────────────────────────
    ax_top.plot(ar_ok_bs, ar_ok_tps,
                color=AR_C, lw=LW, marker="o", ms=MS,
                label="AR (自回归基线)", zorder=3)
    ax_top.plot(diff_ok_bs, diff_ok_tps,
                color=DIFF_C, lw=LW, marker="s", ms=MS,
                label="跳步扩散模型 (50步)", zorder=3)

    common = sorted(set(ar_ok_bs) & set(diff_ok_bs))
    if len(common) >= 2:
        ar_interp   = np.interp(common, ar_ok_bs,   ar_ok_tps)
        diff_interp = np.interp(common, diff_ok_bs, diff_ok_tps)
        ax_top.fill_between(common, ar_interp, diff_interp,
                            alpha=0.10, color=DIFF_C, zorder=1)

    for b in ar_bad:
        ax_top.axvline(b, color=AR_C, lw=1.2, ls=":", alpha=0.7)
        ax_top.text(b * 1.03, ax_top.get_ylim()[1] * 0.98,
                    "OOM", color=AR_C, fontsize=9, va="top", ha="left",
                    rotation=90)

    ax_top.set_ylabel("吞吐量 (Tokens / sec)", labelpad=8)
    ax_top.set_title("系统级并发压力测试 — AR 与跳步扩散模型对比",
                     pad=12, fontweight="bold", fontsize=16)
    ax_top.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}"))
    # 图例自适应：优先左上，避免与曲线重叠
    ax_top.legend(loc="best", frameon=True, borderpad=0.7, labelspacing=0.4)
    ax_top.grid(axis="y", lw=0.6, alpha=0.4, color="#aaaaaa")
    ax_top.set_xscale("log", base=2)

    # ── 下图：内存 ────────────────────────────────────────────────────────────
    ax_bot.plot(ar_ok_bs_m, ar_ok_mem,
                color=AR_C, lw=LW, marker="o", ms=MS, zorder=3)
    ax_bot.plot(diff_ok_bs_m, diff_ok_mem,
                color=DIFF_C, lw=LW, marker="s", ms=MS, zorder=3)

    if diff_ok_mem:
        med_diff = float(np.median(diff_ok_mem))
        ax_bot.axhline(med_diff, color=DIFF_C, lw=1.0, ls="--", alpha=0.55)
        ax_bot.text(diff_ok_bs_m[-1] * 1.05, med_diff,
                    f"≈{med_diff*1024:.0f} MB",
                    color=DIFF_C, fontsize=9, va="center")

    for b in ar_bad:
        ax_bot.axvline(b, color=AR_C, lw=1.2, ls=":", alpha=0.7)
        ax_bot.annotate("AR 内存溢出 →",
                        xy=(b, ax_bot.get_ylim()[1]),
                        xytext=(b / 2.5, ax_bot.get_ylim()[1] * 0.85),
                        color=AR_C, fontsize=9,
                        arrowprops=dict(arrowstyle="->", color=AR_C, lw=1.2),
                        ha="center")

    ax_bot.set_xlabel("并发批次 (Batch Size)", labelpad=8)
    ax_bot.set_ylabel("物理内存峰值 (GB, RSS)", labelpad=8)
    ax_bot.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.3f}"))
    ax_bot.grid(axis="y", lw=0.6, alpha=0.4, color="#aaaaaa")

    all_bs = sorted(set(ar_bs + diff_bs))
    ax_bot.set_xticks(all_bs)
    ax_bot.set_xticklabels([str(b) for b in all_bs])

    fig.text(0.99, 0.01,
             "仅 CPU · 序列长度=256 · d=64 · 2层 Transformer",
             ha="right", va="bottom", fontsize=8.5, color="#888888")

    plt.savefig(out_path)
    plt.close()
    print(f"  已保存 → {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("学术可视化流水线")
    print("=" * 60)

    print("\n[1/2] 潜在空间流形对比 …")
    fig_pca_manifold("pca_manifold_alignment.png")

    print("\n[2/2] 系统级并发压力测试 …")
    fig_stress_benchmark("system_stress_benchmark.png")

    print("\n完成。两张图均已以 300 dpi 保存。")
    print("=" * 60)
