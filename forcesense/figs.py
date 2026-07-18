"""Generate the figures for the plug-and-play cross-policy paper from the
result JSONs in data/wbc/cross/. Outputs PDF figures to docs/paper/plugandplay/figs/.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CROSS = os.path.join(REPO, "data/wbc/cross")
OUT = os.path.join(REPO, "docs/paper/plugandplay/figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.spmnes.top": False,
} if False else {"font.size": 9})
for k in ("axes.spines.top", "axes.spines.right"):
    plt.rcParams[k] = False

# muted, colorblind-safe palette
C = {"proprio": "#B0592D", "resid": "#2E6E8E", "both": "#5B8C5A"}
LAB = {"proprio": "proprioception\n(baseline)", "resid": "residual\n(ours)",
       "both": "both"}


def load(name):
    with open(os.path.join(CROSS, name)) as f:
        return json.load(f)


def row(rows, needle):
    for r in rows:
        if needle in r["set"]:
            return r
    raise KeyError(needle)


# ---- Fig 1: single-controller training (train on base A only) -------------
# for each feature mode, eval on A (seen, full), B (unseen soft), C (unseen stiff)
feats = ["proprio", "resid", "both"]
data = {f: load(f"result_trainA_{f}.json") for f in feats}
# in-domain uses the HELD-OUT columns of base (never trained on); unseen uses
# the full soft / stiff datasets.
domains = [("IN-DOMAIN val", "base\n(in-domain)"),
           ("cross_B_soft", "soft kp0.6\n(unseen)"),
           ("cross_C_stiff", "stiff kp1.5\n(unseen)")]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
metrics = [("region_acc", "Region localization accuracy"),
           ("det_prec", "Detection precision")]
x = np.arange(len(domains)); w = 0.26
for ax, (mkey, mlab) in zip(axes, metrics):
    for i, f in enumerate(feats):
        vals = [row(data[f], d[0])[mkey] for d in domains]
        ax.bar(x + (i - 1) * w, vals, w, color=C[f], label=LAB[f],
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([d[1] for d in domains])
    ax.set_ylim(0, 1.0); ax.set_ylabel(mlab)
    ax.axvspan(0.5, 2.5, color="0.9", alpha=0.4, zorder=0)
    ax.grid(axis="y", color="0.85", lw=0.6)
axes[0].legend(frameon=False, fontsize=7.5, loc="lower left", ncol=1)
fig.suptitle("Trained on ONE controller (base), evaluated on unseen controllers",
             fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "fig_gap.pdf"))
print("wrote fig_gap.pdf")

# ---- Fig 2: controller randomization (leave-one-out -> unseen stiff C) -----
# single (train A) vs randomized (train A+B), per feature mode, regAcc on unseen C
single = {f: row(load(f"result_trainA_{f}.json"), "cross_C_stiff")["region_acc"]
          for f in feats}
rand = {f: row(load(f"result_AB_{f}.json"), "cross_C_stiff")["region_acc"]
        for f in feats}
fig2, ax = plt.subplots(figsize=(3.5, 2.7))
x = np.arange(len(feats)); w = 0.36
ax.bar(x - w / 2, [single[f] for f in feats], w, color="#BBBBBB",
       edgecolor="white", label="1 controller")
ax.bar(x + w / 2, [rand[f] for f in feats], w,
       color=[C[f] for f in feats], edgecolor="white",
       label="2 controllers (rand.)")
for i, f in enumerate(feats):
    ax.text(i - w / 2, single[f] + .02, f"{single[f]:.2f}", ha="center", fontsize=7)
    ax.text(i + w / 2, rand[f] + .02, f"{rand[f]:.2f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([LAB[f].split("\n")[0] for f in feats])
ax.set_ylim(0, 1.0); ax.set_ylabel("Region acc. on unseen stiff controller")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.grid(axis="y", color="0.85", lw=0.6)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT, "fig_randomization.pdf"))
print("wrote fig_randomization.pdf")

# ---- dump a consolidated table to stdout for the paper ----------------------
print("\n==== CONSOLIDATED (regAcc / detF1 / detPrec / actAcc) ====")
for f in feats:
    print(f"\n[train base A only] feat={f}")
    for key, lab in domains:
        r = row(data[f], key)
        print(f"  {lab.splitlines()[0]:14s} regAcc {r['region_acc']:.3f}  "
              f"F1 {r['det_f1']:.3f}  prec {r['det_prec']:.3f}  act {r['active_acc']:.3f}")
for f in feats:
    r = row(load(f"result_AB_{f}.json"), "cross_C_stiff")
    print(f"[train A+B -> unseen C] feat={f:8s} regAcc {r['region_acc']:.3f}  "
          f"F1 {r['det_f1']:.3f}  prec {r['det_prec']:.3f}  act {r['active_acc']:.3f}")
