#!/usr/bin/env python3
"""
Summarise the repeat-run CNR measurements produced by submit_cnr_repeats.sh.

Reports mean +/- std over seeds for each top design, plus the per-sector CNR
spread, so CNR can be quoted with an error bar instead of as a bare point
estimate. Also prints pairwise design gaps against the measurement noise, which
is the number that decides whether one design actually beats another.

Usage:
  python3 analyze_cnr_repeats.py
  python3 analyze_cnr_repeats.py --results_dir results
"""
import argparse
import glob
import os
import re

import numpy as np

# Fallback only. Named designs go stale the moment a campaign advances, so the
# default path derives the top designs from the results CSV instead; this list
# is used only if that CSV cannot be read. It names configs from the retired
# 180-iteration campaign.
CONFIGS = [
    "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230",
    "mobo_0177_ap0.3512_nap97_nd1_604_nd2_584",
    "mobo_0173_ap0.3500_nap117_nd1_446_nd2_236",
]


def short_name(config: str) -> str:
    m = re.match(r"(mobo_\d+)", config)
    return m.group(1) if m else config[:12]


def collect(results_dir: str, config: str):
    """Return (seeds, sector_mean_cnrs, sector_matrix) for one config.

    The headline number is the MEAN OVER SECTORS, not overall_cnr. Those are
    different quantities: overall_cnr pools every hot pixel, so the largest rods
    dominate it, while cnr_sector_mean weights each rod size equally. RY asked
    for the sector mean in Jul 2026 and it is what the campaign optimizes and
    what every row of the results CSV holds.

    This function previously returned overall_cnr, which made repeat runs of
    mobo_0296 read as 4.91 against a campaign value of 4.77 and look like a
    systematic offset. It was the pooled metric being compared with the
    sector-mean one. The two differ by ~0.15 for these designs, which is larger
    than the effects being measured.
    """
    pattern = os.path.join(results_dir, config, "cnr_repeat_seed*", "cnr_results.npz")
    seeds, sector_mean, sectors = [], [], []
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"cnr_repeat_seed(\d+)", path)
        if not m:
            continue
        d = np.load(path, allow_pickle=True)
        sec = np.asarray(d["sector_cnrs"], dtype=float)
        seeds.append(int(m.group(1)))
        sector_mean.append(float(np.mean(sec)))
        sectors.append(sec)
    if not sector_mean:
        return [], np.array([]), np.empty((0, 0))
    order = np.argsort(seeds)
    return (
        [seeds[i] for i in order],
        np.array([sector_mean[i] for i in order]),
        np.vstack([sectors[i] for i in order]),
    )


def correlate_with_metrics(results_dir, stats_by_config, metrics_csv):
    """Test what design metric predicts CNR reproducibility.

    The three top designs ranked identically by MPXI, ASCI and n_det_ring2 as
    they did by CNR std, suggesting worse-conditioned (more multiplexed) systems
    amplify Poisson noise more. With only three designs that ordering is worth
    a third of a coin flip, so this runs the comparison over whatever set of
    designs has repeat runs.
    """
    if not os.path.exists(metrics_csv):
        print(f"\n(metrics CSV not found at {metrics_csv}; skipping correlation)")
        return
    import pandas as pd

    df = pd.read_csv(metrics_csv)
    rows = []
    for config, std in stats_by_config.items():
        m = df[df["config"] == config]
        if m.empty:
            continue
        r = m.iloc[0]
        # Current objective columns. These were mpxi_mean / asci_pct /
        # sensitivity_mean / fwhm_mean, all of which are retired: sensitivity
        # was dropped entirely, and the other three were redefined. Reading the
        # old names returns NaN and the metric is silently skipped rather than
        # reported as missing.
        rows.append({
            "config": short_name(config), "cnr_std": std,
            "mpxi": r.get("mpxi_windowed_active_mean"),
            "asci": r.get("asci_pct_fwhm0p45"),
            "ppds_ring1": r.get("ppds_ring1"),
            "fwhm": r.get("fwhm_weighted_mean"),
            "aperture": r.get("aperture_diam_mm"),
            "n_det_ring2": r.get("n_det_ring2"),
        })
    if len(rows) < 3:
        print("\n(need >= 3 designs with repeat runs to correlate; skipping)")
        return

    d = pd.DataFrame(rows)
    print()
    print("=" * 72)
    print("WHAT PREDICTS CNR REPRODUCIBILITY?")
    print("=" * 72)
    print()
    print(d.sort_values("cnr_std").to_string(index=False))
    print()
    print(f"Spearman rank correlation of each metric against CNR std (n={len(d)}):")
    for col in ["mpxi", "asci", "ppds_ring1", "fwhm", "aperture", "n_det_ring2"]:
        if col not in d or d[col].isna().any():
            continue
        rho = d["cnr_std"].corr(d[col], method="spearman")
        # A rank correlation over a near-constant column is noise dressed as a
        # result. The top designs' weighted FWHM spans 0.4811 to 0.4838 -- a
        # range of 0.003 mm -- and produced rho = -0.900, which means nothing.
        spread = d[col].max() - d[col].min()
        rel = spread / abs(d[col].mean()) if d[col].mean() else 0.0
        note = "   (metric near-constant here; rho is meaningless)" if rel < 0.02 else ""
        print(f"  {col:<14} rho = {rho:+.3f}   range {spread:.4g}{note}")
    if len(d) < 6:
        print("\nNOTE: with fewer than ~6 designs these correlations are not")
        print("      evidence. Re-run with a wider spread of designs.")


def main():
    ap = argparse.ArgumentParser(description="Summarise repeat-run CNR measurements")
    ap.add_argument("--results_dir", default="results",
                    help="Directory holding the per-config work dirs (default: results)")
    ap.add_argument("--config_list", default=None,
                    help="File with one config name per line (default: the built-in top designs)")
    ap.add_argument("--metrics_csv", default="results/results_summary_mobo.csv",
                    help="Campaign CSV, used to correlate CNR std against design metrics "
                         "and, without --config_list, to pick the top designs")
    ap.add_argument("--top_n", type=int, default=5,
                    help="How many top-CNR designs to analyse when --config_list "
                         "is not given")
    args = ap.parse_args()

    global CONFIGS
    if args.config_list:
        with open(args.config_list) as f:
            CONFIGS = [ln.strip() for ln in f if ln.strip()]
    else:
        # Derive the top designs from the archive rather than using the
        # built-in list, which names configs from the retired 180-iteration
        # campaign. A stale default here does not error -- it silently reports
        # on designs nobody is considering any more.
        try:
            import pandas as pd
            df = pd.read_csv(args.metrics_csv).dropna(subset=["cnr_sector_mean"])
            CONFIGS = df.nlargest(args.top_n, "cnr_sector_mean")["config"].astype(str).tolist()
            print(f"Top {len(CONFIGS)} designs by CNR from {args.metrics_csv}\n")
        except Exception as e:
            print(f"Could not read {args.metrics_csv} ({e}); falling back to the "
                  f"built-in list, which may be out of date:\n  {CONFIGS}\n")

    stats = {}
    std_by_config = {}

    print("=" * 72)
    print("PER-DESIGN CNR ACROSS SEEDS")
    print("=" * 72)
    for config in CONFIGS:
        seeds, overall, sector_mat = collect(args.results_dir, config)
        name = short_name(config)
        if overall.size == 0:
            print(f"\n{name}: no repeat runs found "
                  f"(expected {args.results_dir}/{config}/cnr_repeat_seed*/cnr_results.npz)")
            continue

        mean, std = overall.mean(), overall.std(ddof=1) if overall.size > 1 else 0.0
        stats[name] = (mean, std, overall.size)
        std_by_config[config] = std

        print(f"\n{name}  (n={overall.size})")
        print(f"  seeds:   {seeds}")
        print(f"  CNR:     {np.array2string(overall, precision=4, floatmode='fixed')}")
        print(f"  mean:    {mean:.4f}")
        print(f"  std:     {std:.4f}")
        print(f"  range:   {overall.min():.4f} - {overall.max():.4f}  "
              f"(spread {overall.max() - overall.min():.4f})")

        if sector_mat.size:
            print("  per-sector mean +/- std:")
            for s in range(sector_mat.shape[1]):
                col = sector_mat[:, s]
                sd = col.std(ddof=1) if col.size > 1 else 0.0
                print(f"    sector {s}: {col.mean():.3f} +/- {sd:.3f}")

    if len(stats) < 2:
        print("\nNeed at least two designs with repeat runs for a pairwise comparison.")
        return

    # Pooled std across all designs = the measurement noise floor.
    pooled = np.sqrt(np.mean([s ** 2 for (_, s, n) in stats.values() if n > 1]))

    print()
    print("=" * 72)
    print("PAIRWISE GAPS vs MEASUREMENT NOISE")
    print("=" * 72)
    print(f"\nPooled per-run std (noise floor): {pooled:.4f}")
    print("A gap is only meaningful if it clears ~2x the std of the difference,")
    print(f"i.e. roughly {2 * pooled * np.sqrt(2):.4f} CNR for a single-run comparison.\n")

    names = list(stats.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            (ma, sa, na), (mb, sb, nb) = stats[a], stats[b]
            gap = ma - mb
            # Standard error of the difference of two means
            se = np.sqrt((sa ** 2 / na if na else 0) + (sb ** 2 / nb if nb else 0))
            verdict = "DISTINGUISHABLE" if se > 0 and abs(gap) > 2 * se else "indistinguishable"
            se_str = f"{abs(gap) / se:.2f}" if se > 0 else "n/a"
            print(f"  {a} vs {b}:")
            print(f"    gap = {gap:+.4f}   SE(diff) = {se:.4f}   |gap|/SE = {se_str}")
            print(f"    -> {verdict}")

    correlate_with_metrics(args.results_dir, std_by_config, args.metrics_csv)

    print()
    print("=" * 72)
    print("QUOTABLE SUMMARY")
    print("=" * 72)
    for name, (mean, std, n) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name}: CNR = {mean:.2f} +/- {std:.2f}  (n={n} seeds)")


if __name__ == "__main__":
    main()
