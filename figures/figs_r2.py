"""B r2 그림 세 장: 데이터 효율·고정 대 질의별·헤드 대 정제기. 수치는 data_r2.json 에서 읽는다."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = json.loads((Path(__file__).parent / "data_r2.json").read_text())
OUT = Path(__file__).resolve().parent  # 그림은 이 폴더에 저장한다
INK, MUTED, GRID = "#222222", "#666666", "#e3e3e3"
BLUE, ORANGE = "#0072B2", "#D55E00"
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "legend.fontsize": 7.5,
                     "legend.frameon": False, "pdf.fonttype": 42,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_eff():
    K = [1, 2, 3, 5, 10]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for ax, (bb, v) in zip(axes, D["eff"].items()):
        ax.axhline(v["k0"], color=MUTED, lw=0.9, ls="--")
        ax.axhline(v["ref"], color=MUTED, lw=0.9, ls=":")
        ax.text(10.3, v["k0"], "unrefined", fontsize=7, color=MUTED, va="bottom", ha="right")
        ax.text(1.0, v["ref"], "reference ($k{=}23$)", fontsize=7, color=MUTED, va="bottom", ha="left")
        ax.errorbar(K, v["rand"], yerr=v["sd"], fmt="s--", ms=3.5, lw=1.4, color=ORANGE, capsize=2,
                    label="random (20 draws, ±1 s.d.)")
        ax.plot(K, v["near"], "o-", ms=3.5, lw=1.6, color=BLUE, label="nearest")
        ax.set_title(bb, loc="left")
        ax.set_xlabel("annotated sequences $k$")
        ax.set_xticks(K)
        style(ax)
    axes[0].set_ylabel("PA-MPJPE (mm)")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(h[::-1], l[::-1], loc="lower center", ncol=2)
    fig.savefig(OUT / "fig_data_efficiency.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_fixed():
    bbs = list(D["fixed"].keys())
    arms = [("fixed: medoid", "medoid", None, "#c9ced4"),
            ("fixed: spread", "spread", None, "#9aa3ad"),
            ("fixed: random (10 draws)", "rand", "rand_sd", "#5f6b78"),
            ("nearest, per query", "query", "query_sd", ORANGE)]
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    w = 0.19
    for i, (name, key, sdk, col) in enumerate(arms):
        xs = [j + (i - 1.5) * w for j in range(len(bbs))]
        v = [D["fixed"][b][key] for b in bbs]
        e = [D["fixed"][b][sdk] for b in bbs] if sdk else None
        ax.bar(xs, v, width=w, color=col, edgecolor="white", lw=2, label=name,
               yerr=e, error_kw=dict(ecolor=INK, lw=0.8, capsize=2))
    ax.set_xticks(range(len(bbs)))
    ax.set_xticklabels(bbs)
    ax.set_ylabel("PA-MPJPE reduction (%)")
    ax.axhline(0, color=MUTED, lw=0.6)
    style(ax)
    ax.grid(False, axis="x")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(OUT / "fig_fixed_vs_query.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_head():
    k = [1, 2, 3, 4]
    h = D["head"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.errorbar(k, h["head"], yerr=h["head_sd"], fmt="s-", ms=4.5, lw=1.8, color=ORANGE,
                capsize=2, label="SMPL head adapted")
    ax.errorbar(k, h["ref"], yerr=h["ref_sd"], fmt="o--", ms=4.5, lw=1.8, color=BLUE,
                capsize=2, label="joint-space refiner")
    ax.axhline(10.30, color=BLUE, lw=0.9, ls=":")
    ax.text(3.6, 10.6, "refiner, every available sequence", color=MUTED, fontsize=7, ha="right")
    ax.set_xticks(k)
    ax.set_xlabel("annotated sequences $k$ (subject held out)")
    ax.set_ylabel("PA-MPJPE reduction (%)")
    ax.set_ylim(0, 20)
    style(ax)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_head_vs_refiner.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_eff()
    fig_fixed()
    fig_head()
    print("saved")
