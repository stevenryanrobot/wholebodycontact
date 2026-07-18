"""Deployment-semantics metrics for the whole-body contact sensor (round 2).

What the demo user perceives is not per-frame accuracy at th=0.5 on raw labels:
  1. HONEST LABELS: positive = true |F| > 5 N (proprioceptive dead-zone);
     frames with contact below the dead-zone (ramp heads/tails) and frames
     within 1 s (50 frames @50 Hz) AFTER a force release are EXCLUDED from
     negatives — post-release ring-down is proprioceptively indistinguishable
     from contact, so neither prediction is penalized there.
  2. THRESHOLD is a free deployment parameter -> sweep it.
  3. TEMPORAL DEBOUNCING: k-frame persistence + hysteresis (ON after k
     consecutive frames p > th_hi, OFF after k consecutive p < th_lo) applied
     to the per-env probability SEQUENCE, like the demo would.

Shared by forcesense/eval/eval_deploy.py (post-hoc re-eval) and forcesense/train/core.py
(deploy_select model selection during training).
"""
import torch

DEADZONE_N = 5.0
RINGDOWN_FRAMES = 50


def honest_labels(grid, deadzone_n=DEADZONE_N, ringdown=RINGDOWN_FRAMES):
    """Returns (pos [T,E] bool, valid [T,E] bool).
    pos   : true |F| > deadzone_n.
    valid : frame counts toward metrics (False = excluded: sub-dead-zone
            contact, or ring-down within `ringdown` frames after the last
            positive frame). Ring-down state does not cross file segments."""
    mag, phase = grid["mag"], grid["phase"]
    T, E = mag.shape
    dev = mag.device
    pos = mag > (deadzone_n / grid["force_max"])
    subthr_contact = (~pos) & (phase != 0)          # contact below dead-zone
    tidx = torch.arange(T, device=dev).unsqueeze(1).expand(T, E)
    since = torch.full((T, E), 10**9, device=dev, dtype=torch.long)
    for s0, s1 in zip(grid["seg_bounds"][:-1], grid["seg_bounds"][1:]):
        seg_t = tidx[s0:s1] - s0
        marked = torch.where(pos[s0:s1], seg_t, torch.full_like(seg_t, -(10**9)))
        last = torch.cummax(marked, dim=0).values
        since[s0:s1] = seg_t - last                 # 0 at positives
    ringdown_excl = (~pos) & (since >= 1) & (since <= ringdown)
    valid = ~(subthr_contact | ringdown_excl)
    return pos, valid


def prf(pred, pos, valid, mask=None):
    """Precision/recall/F1 of pred [.,.] bool vs pos over valid (& mask) frames."""
    m = valid if mask is None else (valid & mask)
    p, t = pred[m], pos[m]
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"f1": round(f1, 4), "prec": round(prec, 4), "rec": round(rec, 4),
            "tp": tp, "fp": fp, "fn": fn}


def frame_threshold_sweep(det_p, pos, valid, ths=None):
    """Vectorized per-frame F1 over thresholds. det_p [T,E] (NaN = no window)."""
    if ths is None:
        ths = [round(0.30 + 0.05 * i, 2) for i in range(14)]     # 0.30..0.95
    ok = valid & ~torch.isnan(det_p)
    best = None
    for th in ths:
        r = prf(det_p > th, pos, ok)
        if best is None or r["f1"] > best["f1"]:
            best = {"th": th, **r}
    return best


def debounce_scan(det_p, th_hi, th_lo, k):
    """Hysteresis + k-frame persistence over the time axis.
    det_p [T,E] probs (NaN treated as 'below both thresholds').
    ON after k consecutive p > th_hi; OFF after k consecutive p < th_lo."""
    T, E = det_p.shape
    x = torch.nan_to_num(det_p, nan=0.0).cpu()
    state = torch.zeros(E, dtype=torch.bool)
    cnt_on = torch.zeros(E, dtype=torch.long)
    cnt_off = torch.zeros(E, dtype=torch.long)
    out = torch.zeros(T, E, dtype=torch.bool)
    for t in range(T):
        hi = x[t] > th_hi
        lo = x[t] < th_lo
        cnt_on = torch.where(hi, cnt_on + 1, torch.zeros_like(cnt_on))
        cnt_off = torch.where(lo, cnt_off + 1, torch.zeros_like(cnt_off))
        state = torch.where(~state & (cnt_on >= k), torch.ones_like(state), state)
        state = torch.where(state & (cnt_off >= k), torch.zeros_like(state), state)
        out[t] = state
    return out.to(det_p.device)


def debounce_sweep(det_p, pos, valid, base_th, ks=(3, 5), th_offsets=(-0.1, 0.0, 0.1),
                   hyst=0.15, mask=None):
    """Try k x th_hi combos around the best per-frame threshold; return best-F1
    config. th_lo = th_hi - hyst."""
    ok = valid & ~torch.isnan(det_p)
    best = None
    for k in ks:
        for off in th_offsets:
            th_hi = min(max(base_th + off, 0.30), 0.95)
            th_lo = max(th_hi - hyst, 0.05)
            state = debounce_scan(det_p, th_hi, th_lo, k)
            r = prf(state, pos, ok, mask)
            if best is None or r["f1"] > best["f1"]:
                best = {"k": k, "th_hi": round(th_hi, 2), "th_lo": round(th_lo, 2),
                        **r}
    return best


@torch.no_grad()
def predict_grids(model, grid, featurize, W, val_envs, bs=8192):
    """Run the model over ALL frames of the first `val_envs` envs.
    Returns det_p [T,V] (NaN where the W-window is invalid), loc_arg [T,V]."""
    T, E = grid["T"], grid["E"]
    dev = grid["X"].device
    V = val_envs
    det_p = torch.full((T, V), float("nan"), device=dev)
    loc_arg = torch.zeros(T, V, dtype=torch.long, device=dev)
    ts = [torch.arange(s0 + W - 1, s1, device=dev)
          for s0, s1 in zip(grid["seg_bounds"][:-1], grid["seg_bounds"][1:])]
    t_valid = torch.cat(ts)
    e = torch.arange(V, device=dev)
    tt, ee = torch.meshgrid(t_valid, e, indexing="ij")
    t_flat, e_flat = tt.reshape(-1), ee.reshape(-1)
    model.eval()
    for s in range(0, t_flat.shape[0], bs):
        ti, ei = t_flat[s:s + bs], e_flat[s:s + bs]
        det, loc, _, _ = model(featurize(ti, ei))
        det_p[ti, ei] = torch.sigmoid(det.squeeze(-1))
        loc_arg[ti, ei] = loc.argmax(-1)
    return det_p, loc_arg


def default_eval_slices(grid, W, frames_per_seg=8000):
    """One contiguous window-valid slice per file segment (for fast selection)."""
    slices = []
    for s0, s1 in zip(grid["seg_bounds"][:-1], grid["seg_bounds"][1:]):
        t0 = s0 + W - 1
        slices.append((t0, min(t0 + frames_per_seg, s1)))
    return slices


@torch.no_grad()
def deploy_score(model, grid, featurize, W, val_envs, pos, valid,
                 slices=None, ths=(0.5, 0.7, 0.85), k=3, hyst=0.15, bs=8192):
    """Light-weight deployment selection score for training-time eval:
    best debounced honest F1 (k-persistence + hysteresis) over a small
    threshold grid, computed on contiguous per-segment val slices."""
    if slices is None:
        slices = default_eval_slices(grid, W)
    dev = grid["X"].device
    e = torch.arange(val_envs, device=dev)
    model.eval()
    seg_probs = []
    for (t0, t1) in slices:
        t = torch.arange(t0, t1, device=dev)
        tt, ee = torch.meshgrid(t, e, indexing="ij")
        t_flat, e_flat = tt.reshape(-1), ee.reshape(-1)
        det_p = torch.empty(t1 - t0, val_envs, device=dev)
        for s in range(0, t_flat.shape[0], bs):
            ti, ei = t_flat[s:s + bs], e_flat[s:s + bs]
            det, _, _, _ = model(featurize(ti, ei))
            det_p[ti - t0, ei] = torch.sigmoid(det.squeeze(-1))
        seg_probs.append((det_p, pos[t0:t1, :val_envs], valid[t0:t1, :val_envs]))
    best = None
    for th in ths:
        tp = fp = fn = 0
        for det_p, p, v in seg_probs:
            state = debounce_scan(det_p, th, max(th - hyst, 0.05), k)
            m = v
            tp += int((state & p & m).sum()); fp += int((state & ~p & m).sum())
            fn += int((~state & p & m).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if best is None or f1 > best["f1"]:
            best = {"th_hi": th, "k": k, "f1": round(f1, 4),
                    "prec": round(prec, 4), "rec": round(rec, 4)}
    return best
