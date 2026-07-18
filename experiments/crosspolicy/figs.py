"""Generate the 4 figures for docs/paper/plugandplay from the experiment results.
Outputs docs/paper/plugandplay/figs/{fig_gap,fig_robust,fig_region,fig_grfnoise}.pdf.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGDIR = os.path.join(REPO, "docs/paper/plugandplay/figs")
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200,
                     "legend.frameon": False, "savefig.bbox": "tight"})
C = {"proprioception": "#d1495b", "residual": "#2e86de", "both": "#7a7a7a"}
CTRL = ["A_base", "B_soft", "C_stiff", "D_softud", "E_vstiff", "F_underd", "G_overd"]
CLABEL = ["base", "soft", "stiff", "soft-ud", "v-stiff", "under-d", "over-d"]


def load_sweep():
    return json.load(open(os.path.join(REPO, "data/wbc/cross/sweep_result.json")))


# ---- fig_gap: per-controller leave-one-out regAcc ------------------------- #
def fig_gap():
    s = load_sweep()
    order = {"proprio": "proprioception", "resid": "residual", "both": "both"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.3))
    x = np.arange(len(CTRL))
    for feat, name in order.items():
        by = {f["held"]: f for f in s[feat]["folds"]}
        reg = [by[c]["regAcc"] for c in CTRL]
        pr = [by[c]["prec"] for c in CTRL]
        a1.plot(x, reg, "-o", ms=3, color=C[name], label=name)
        a2.plot(x, pr, "-o", ms=3, color=C[name], label=name)
    for a, ttl in [(a1, "region accuracy"), (a2, "detection precision")]:
        a.set_xticks(x); a.set_xticklabels(CLABEL, rotation=35, ha="right")
        a.set_ylim(0.3, 1.0); a.set_title(ttl); a.set_xlabel("held-out (unseen) controller")
        a.axhline(0.9, ls=":", c="0.7", lw=0.8)
    a1.set_ylabel("accuracy"); a1.legend(loc="lower left")
    fig.savefig(os.path.join(FIGDIR, "fig_gap.pdf")); plt.close(fig)


# ---- fig_robust: worst / mean regAcc bars --------------------------------- #
def fig_robust():
    s = load_sweep()
    feats = [("proprio", "proprioception"), ("resid", "residual"), ("both", "both")]
    x = np.arange(len(feats)); w = 0.35
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    worst = [s[f]["worst_regAcc"] for f, _ in feats]
    mean = [s[f]["mean_regAcc"] for f, _ in feats]
    ax.bar(x - w/2, worst, w, label="worst-case", color="#c0392b")
    ax.bar(x + w/2, mean, w, label="mean", color="#2e86de")
    for xi, (wv, mv) in enumerate(zip(worst, mean)):
        ax.text(xi - w/2, wv + .01, f"{wv:.2f}", ha="center", fontsize=6.5)
        ax.text(xi + w/2, mv + .01, f"{mv:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in feats])
    ax.set_ylabel("region accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("cross-controller robustness (7 held-out)"); ax.legend()
    fig.savefig(os.path.join(FIGDIR, "fig_robust.pdf")); plt.close(fig)


# ---- fig_region: per-region recall (proprio vs resid) --------------------- #
def fig_region():
    regions = ["left\narm", "right\narm", "left\nleg", "right\nleg", "trunk"]
    proprio = [0.807, 0.794, 0.559, 0.563, 0.427]
    resid = [0.982, 0.976, 0.894, 0.884, 0.855]
    x = np.arange(len(regions)); w = 0.38
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.bar(x - w/2, proprio, w, label="proprioception", color=C["proprioception"])
    ax.bar(x + w/2, resid, w, label="residual (ours)", color=C["residual"])
    ax.axvspan(1.5, 3.5, color="0.92", zorder=0)   # highlight legs
    ax.text(2.5, 0.05, "legs", ha="center", fontsize=7, color="0.4")
    ax.set_xticks(x); ax.set_xticklabels(regions)
    ax.set_ylabel("recall"); ax.set_ylim(0, 1.05); ax.legend(loc="upper right")
    ax.set_title("per-region recall (cross-controller)")
    fig.savefig(os.path.join(FIGDIR, "fig_region.pdf")); plt.close(fig)


# ---- fig_grfnoise: recall vs GRF estimation error ------------------------- #
def fig_grfnoise():
    alpha = [0.0, 0.10, 0.25, 0.50, 1.00]
    legs = [0.918, 0.840, 0.790, 0.732, 0.658]
    arms = [0.982, 0.976, 0.974, 0.973, 0.967]
    trunk = [0.639, 0.475, 0.434, 0.394, 0.321]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.axvspan(0, 0.10, color="#eaf3ea", zorder=0)
    ax.text(0.05, 0.30, "foot-sensor\nregime (<10%)", ha="center", fontsize=6, color="#4a7a4a")
    ax.plot(alpha, arms, "-o", ms=3, color="#27ae60", label="arms")
    ax.plot(alpha, legs, "-o", ms=3, color=C["residual"], label="legs")
    ax.plot(alpha, trunk, "-o", ms=3, color="#e67e22", label="trunk")
    ax.set_xlabel(r"ground-reaction estimation error $\alpha$")
    ax.set_ylabel("recall"); ax.set_ylim(0.25, 1.02); ax.legend(loc="center right")
    ax.set_title("legs are robust to GRF error")
    fig.savefig(os.path.join(FIGDIR, "fig_grfnoise.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_gap(); fig_robust(); fig_region(); fig_grfnoise()
    print("wrote 4 figures ->", FIGDIR)
    for f in sorted(os.listdir(FIGDIR)):
        print("  ", f)
