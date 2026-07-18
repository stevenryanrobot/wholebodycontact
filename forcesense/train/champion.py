"""Round-2 finale: pick the deployment champion and export its ONNX.

Reads data/wbc/sweep_v4/deploy_leaderboard.jsonl (Track A + Track B records;
last record per label wins), ranks by debounced honest F1, writes
champion.json, and exports the champion to force_sensor_v4_deploy_champion.onnx
(+ .meta.json) via forcesense/export.py. The web demo JS switch is handled
by the orchestrator — this only produces the artifact + verdict.
"""
import os
import sys
import json
import argparse
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--board", type=str, default="data/wbc/sweep_v4/deploy_leaderboard.jsonl")
    p.add_argument("--out_dir", type=str, default="data/wbc/sweep_v4")
    p.add_argument("--baseline", type=str, default="mlp_w10_regions")
    args = p.parse_args()

    recs = {}
    with open(args.board) as f:
        for line in f:
            r = json.loads(line)
            recs[r["label"]] = r          # last record per label wins
    ranked = sorted(recs.values(), key=lambda r: r["debounce_best"]["f1"], reverse=True)
    # near-ties (within 0.02 debF1 of the top) are broken by region accuracy —
    # a detector that highlights the wrong body part is not a better demo.
    top_f1 = ranked[0]["debounce_best"]["f1"]
    near = [r for r in ranked if top_f1 - r["debounce_best"]["f1"] <= 0.02]
    champ = max(near, key=lambda r: r["region_acc_confirmed"])
    base = recs.get(args.baseline)

    print("[champion] deployment ranking (debounced honest F1):")
    for r in ranked:
        d = r["debounce_best"]
        print(f"  {r['label']:<26} debF1={d['f1']:.3f} P={d['prec']:.3f} R={d['rec']:.3f} "
              f"regAcc={r['region_acc_confirmed']:.3f} "
              f"mot={r['debounce_motion']['f1']:.3f} stat={r['debounce_static']['f1']:.3f}")
    beats = base is None or champ["debounce_best"]["f1"] > base["debounce_best"]["f1"]
    verdict = (f"{champ['label']} is the deployment champion "
               f"(debF1 {champ['debounce_best']['f1']:.3f}"
               + (f" vs baseline {args.baseline} {base['debounce_best']['f1']:.3f}" if base else "")
               + ")")
    print(f"[champion] {verdict}")

    onnx_out = os.path.join(args.out_dir, "force_sensor_v4_deploy_champion.onnx")
    subprocess.run([sys.executable, "-u", os.path.join(REPO, "forcesense/export.py"),
                    "--ckpt", champ["ckpt"], "--out", onnx_out], check=True, cwd=REPO)

    with open(os.path.join(args.out_dir, "champion.json"), "w") as f:
        json.dump({"champion": champ, "baseline": base, "beats_baseline": beats,
                   "onnx": onnx_out,
                   "recommended_det": champ["debounce_best"]}, f, indent=2)
    print(f"[champion] exported -> {onnx_out}; verdict + recommended "
          f"threshold/debounce -> champion.json", flush=True)


if __name__ == "__main__":
    main()
