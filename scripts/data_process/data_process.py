import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pick(df: pd.DataFrame, name: str, occurrence: int = 0) -> pd.Series:
    """
    Handle duplicate column names.
    occurrence=0 -> first one (e.g., 't_ly')
    occurrence=1 -> second one (e.g., 't_ly.1')
    """
    if occurrence == 0:
        if name in df.columns:
            return df[name]
        raise KeyError(f"Missing column: {name}")
    alt = f"{name}.{occurrence}"
    if alt in df.columns:
        return df[alt]
    # fallback: if pandas used different suffix style, try scanning
    matches = [c for c in df.columns if c == name or c.startswith(name + ".")]
    if len(matches) > occurrence:
        return df[matches[occurrence]]
    raise KeyError(f"Missing column occurrence: {name} ({occurrence})")

def plot_xyz(title, t_x, t_y, t_z, a_x, a_y, a_z):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    x = range(len(t_x))

    axes[0].plot(x, t_x, label="target x")
    axes[0].plot(x, a_x, label="actual x")
    axes[0].set_ylabel("x")
    axes[0].legend()

    axes[1].plot(x, t_y, label="target y")
    axes[1].plot(x, a_y, label="actual y")
    axes[1].set_ylabel("y")
    axes[1].legend()

    axes[2].plot(x, t_z, label="target z")
    axes[2].plot(x, a_z, label="actual z")
    axes[2].set_ylabel("z")
    axes[2].set_xlabel("time index")
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig

def plot_rxyz(title, t_rx, t_ry, t_rz, a_rx, a_ry, a_rz):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    x = range(len(t_rx))

    axes[0].plot(x, t_rx, label="target rx")
    axes[0].plot(x, a_rx, label="actual rx")
    axes[0].set_ylabel("rx")
    axes[0].legend()

    axes[1].plot(x, t_ry, label="target ry")
    axes[1].plot(x, a_ry, label="actual ry")
    axes[1].set_ylabel("ry")
    axes[1].legend()

    axes[2].plot(x, t_rz, label="target rz")
    axes[2].plot(x, a_rz, label="actual rz")
    axes[2].set_ylabel("rz")
    axes[2].set_xlabel("time index")
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig

def main(csv_path: str):
    # Read CSV; allow spaces after commas
    df = pd.read_csv(csv_path, skipinitialspace=True)

    # ---- Positions: use first occurrence of ly/ry (occurrence=0) ----
    t_lx = pick(df, "t_lx", 0); t_ly_pos = pick(df, "t_ly", 0); t_lz = pick(df, "t_lz", 0)
    a_lx = pick(df, "a_lx", 0); a_ly_pos = pick(df, "a_ly", 0); a_lz = pick(df, "a_lz", 0)

    t_rx = pick(df, "t_rx", 0); t_ry_pos = pick(df, "t_ry", 0); t_rz = pick(df, "t_rz", 0)
    a_rx = pick(df, "a_rx", 0); a_ry_pos = pick(df, "a_ry", 0); a_rz = pick(df, "a_rz", 0)

    # ---- Rotations: use SECOND occurrence of t_ly / t_ry etc (occurrence=1) ----
    # According to your header: ... t_lr,t_lp,t_ly, ... and ... a_lr,a_lp,a_ly ...
    t_lrx = pick(df, "t_lr", 0)
    t_lry = pick(df, "t_lp", 0)
    t_lrz = pick(df, "t_ly", 1)   # second 't_ly' is the rotation-y in your header
    a_lrx = pick(df, "a_lr", 0)
    a_lry = pick(df, "a_lp", 0)
    a_lrz = pick(df, "a_ly", 1)   # second 'a_ly'

    t_rrx = pick(df, "t_rr", 0)
    t_rry = pick(df, "t_rp", 0)
    t_rrz = pick(df, "t_ry", 1)   # second 't_ry'
    a_rrx = pick(df, "a_rr", 0)
    a_rry = pick(df, "a_rp", 0)
    a_rrz = pick(df, "a_ry", 1)   # second 'a_ry'

    # ----------------- Jump detection for target left x (t_lx) -----------------
    # Compute sample-to-sample differences and flag large jumps as outliers.
    # Threshold is set adaptively: mean(abs(diff)) + 3*std(abs(diff)).
    try:
        t_lx_arr = np.asarray(t_lx, dtype=float)
        diffs = np.diff(t_lx_arr)
        abs_diffs = np.abs(diffs)
        if abs_diffs.size > 0:
            threshold = float(np.mean(abs_diffs) + 3.0 * np.std(abs_diffs))
        else:
            threshold = 0.0

        jump_indices = [int(i + 1) for i, v in enumerate(abs_diffs) if v > threshold]
    except Exception as e:
        # On any error, fallback to empty list but report the issue
        print(f"[jump_detection] error computing jumps for t_lx: {e}")
        jump_indices = []

    print(f"[jump_detection] detected {len(jump_indices)} jumps in t_lx; indices: {jump_indices}")
    # ---------------------------------------------------------------------------
    # If jumps found, compute errors for the 20 samples before each jump and
    # aggregate across all jumps. Error = target - actual.
    if len(jump_indices) == 0:
        print("[jump_analysis] no jumps found; skipping error-summary calculations")
    else:
        def _collect_errors(t_series, a_series, indices, window=20):
            t_arr = np.asarray(t_series, dtype=float)
            a_arr = np.asarray(a_series, dtype=float)
            segs = []
            for idx in indices:
                start = max(0, int(idx) - window)
                end = int(idx)
                seg_t = t_arr[start:end]
                seg_a = a_arr[start:end]
                if seg_t.size == 0:
                    continue
                segs.append(seg_t - seg_a)
            if len(segs) == 0:
                return np.array([], dtype=float)
            return np.concatenate(segs)

        # Collect left errors
        lx_errs = _collect_errors(t_lx, a_lx, jump_indices, window=20)
        ly_errs = _collect_errors(t_ly_pos, a_ly_pos, jump_indices, window=20)
        lz_errs = _collect_errors(t_lz, a_lz, jump_indices, window=20)
        lrx_errs = _collect_errors(t_lrx, a_lrx, jump_indices, window=20)
        lry_errs = _collect_errors(t_lry, a_lry, jump_indices, window=20)
        lrz_errs = _collect_errors(t_lrz, a_lrz, jump_indices, window=20)

        # Collect right errors
        rx_errs = _collect_errors(t_rx, a_rx, jump_indices, window=20)
        ry_errs = _collect_errors(t_ry_pos, a_ry_pos, jump_indices, window=20)
        rz_errs = _collect_errors(t_rz, a_rz, jump_indices, window=20)
        rrx_errs = _collect_errors(t_rrx, a_rrx, jump_indices, window=20)
        rry_errs = _collect_errors(t_rry, a_rry, jump_indices, window=20)
        rrz_errs = _collect_errors(t_rrz, a_rrz, jump_indices, window=20)

        # Compute means (use nan if empty)
        def _mean_or_nan(arr):
            return float(np.nan) if arr.size == 0 else float(np.mean(arr))

        lmx = _mean_or_nan(lx_errs)
        lmy = _mean_or_nan(ly_errs)
        lmz = _mean_or_nan(lz_errs)
        lmx_rx = _mean_or_nan(lrx_errs)
        lmy_ry = _mean_or_nan(lry_errs)
        lmz_rz = _mean_or_nan(lrz_errs)

        rmx = _mean_or_nan(rx_errs)
        rmy = _mean_or_nan(ry_errs)
        rmz = _mean_or_nan(rz_errs)
        rmx_rx = _mean_or_nan(rrx_errs)
        rmy_ry = _mean_or_nan(rry_errs)
        rmz_rz = _mean_or_nan(rrz_errs)

        # Print the 12 means: left x,y,z,rx,ry,rz then right x,y,z,rx,ry,rz
        print("[jump_analysis] Means over pre-jump windows (target-actual):")
        print(f"  LEFT  -> x={lmx:.6f}, y={lmy:.6f}, z={lmz:.6f}, rx={lmx_rx:.6f}, ry={lmy_ry:.6f}, rz={lmz_rz:.6f}")
        print(f"  RIGHT -> x={rmx:.6f}, y={rmy:.6f}, z={rmz:.6f}, rx={rmx_rx:.6f}, ry={rmy_ry:.6f}, rz={rmz_rz:.6f}")

        # Composite means: average of the 6 position-component means and 6 rotation-component means
        pos_means = np.array([v for v in [lmx, lmy, lmz, rmx, rmy, rmz] if not np.isnan(v)], dtype=float)
        pose_means = np.array([v for v in [lmx_rx, lmy_ry, lmz_rz, rmx_rx, rmy_ry, rmz_rz] if not np.isnan(v)], dtype=float)
        combined_pos_mean = float(np.nan) if pos_means.size == 0 else float(np.mean(pos_means))
        combined_pose_mean = float(np.nan) if pose_means.size == 0 else float(np.mean(pose_means))

        print(f"[jump_analysis] Combined position mean (avg of x,y,z left+right) = {combined_pos_mean:.6f}")
        print(f"[jump_analysis] Combined pose mean (avg of rx,ry,rz left+right) = {combined_pose_mean:.6f}")

    # ---- Rotations: use SECOND occurrence of t_ly / t_ry etc (occurrence=1) ----
    # According to your header: ... t_lr,t_lp,t_ly, ... and ... a_lr,a_lp,a_ly ...
    t_lrx = pick(df, "t_lr", 0)
    t_lry = pick(df, "t_lp", 0)
    t_lrz = pick(df, "t_ly", 1)   # second 't_ly' is the rotation-y in your header
    a_lrx = pick(df, "a_lr", 0)
    a_lry = pick(df, "a_lp", 0)
    a_lrz = pick(df, "a_ly", 1)   # second 'a_ly'

    t_rrx = pick(df, "t_rr", 0)
    t_rry = pick(df, "t_rp", 0)
    t_rrz = pick(df, "t_ry", 1)   # second 't_ry'
    a_rrx = pick(df, "a_rr", 0)
    a_rry = pick(df, "a_rp", 0)
    a_rrz = pick(df, "a_ry", 1)   # second 'a_ry'

    # 1) left pos
    plot_xyz("Left EE position (target vs actual)",
             t_lx, t_ly_pos, t_lz, a_lx, a_ly_pos, a_lz)

    # 2) right pos
    plot_xyz("Right EE position (target vs actual)",
             t_rx, t_ry_pos, t_rz, a_rx, a_ry_pos, a_rz)

    # 3) left rot (rx ry rz)
    plot_rxyz("Left EE rotation (target vs actual)",
              t_lrx, t_lry, t_lrz, a_lrx, a_lry, a_lrz)

    # 4) right rot (rx ry rz)
    plot_rxyz("Right EE rotation (target vs actual)",
              t_rrx, t_rry, t_rrz, a_rrx, a_rry, a_rrz)

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_ee_tracks.py path/to/data.csv")
        sys.exit(1)
    main(sys.argv[1])
