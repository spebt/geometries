#!/usr/bin/env python3
"""
Generate explanatory figures for Direction 1:
ML-Driven SC-SPECT T8 Configuration Optimization

Figures:
1. End-to-end pipeline overview
2. Bayesian Optimization loop
3. Multi-Objective BO / Pareto front concept
4. 7-week timeline to IEEE MIC
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "plots", "direction1")
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# FIGURE 1: End-to-End Pipeline Overview
# ─────────────────────────────────────────────────────────────
def fig1_pipeline_overview():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-2, 4.5)
    ax.axis("off")
    fig.suptitle(
        "Direction 1: ML-Driven SC-SPECT T8 Optimization — Pipeline Overview",
        fontsize=14, fontweight="bold", y=0.97,
    )

    # Box style
    box_kw = dict(boxstyle="round,pad=0.4", linewidth=1.5)
    text_kw = dict(ha="center", va="center", fontsize=9, fontweight="bold")

    # Define pipeline stages: (x, y, width, height, label, color, sublabel)
    stages = [
        (0.5,  2.5, 2.2, 1.6, "Design Vector\n(a, b, phase)",
         "#E8F5E9", "LHS Sampling\n150 configs"),
        (3.5,  2.5, 2.2, 1.6, "PPDF\nRay-Tracing",
         "#E3F2FD", "arg_ppdf_t8.py\n16 HDF5/config"),
        (6.5,  2.5, 2.2, 1.6, "Beam\nAnalysis",
         "#FFF3E0", "ASCI, FWHM\nSensitivity maps"),
        (9.5,  2.5, 2.2, 1.6, "Scalar\nMetrics",
         "#F3E5F5", "JI = S×ASCI/FWHM²\nper-config scalar"),
        (12.5, 2.5, 2.2, 1.6, "Surrogate\nModel",
         "#FFEBEE", "GP (Matérn 5/2)\nR² > 0.85"),
    ]

    for x, y, w, h, label, color, sub in stages:
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            **{**box_kw, "facecolor": color, "edgecolor": "#333"},
        )
        ax.add_patch(box)
        ax.text(x, y + 0.15, label, **text_kw)
        ax.text(x, y - 0.55, sub, ha="center", va="center",
                fontsize=7, color="#555", style="italic")

    # Arrows between stages
    arrow_kw = dict(
        arrowstyle="-|>", color="#333", lw=2,
        connectionstyle="arc3,rad=0",
        mutation_scale=15,
    )
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + stages[i][2] / 2
        x2 = stages[i+1][0] - stages[i+1][2] / 2
        y = stages[i][1]
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                     arrowprops=arrow_kw)

    # BO feedback loop (from surrogate back to design vector)
    ax.annotate(
        "", xy=(0.5, 1.2), xytext=(12.5, 1.2),
        arrowprops=dict(
            arrowstyle="-|>", color="#D32F2F", lw=2.5,
            connectionstyle="arc3,rad=-0.15",
            mutation_scale=18, linestyle="--",
        ),
    )
    ax.text(6.5, 0.4, "Bayesian Optimization Loop",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#D32F2F")
    ax.text(6.5, -0.1, "Acquisition function proposes next (a, b, phase) to evaluate",
            ha="center", va="center", fontsize=8, color="#D32F2F", style="italic")

    # Timing annotations
    ax.text(3.5, 4.2, "~75 min/config", ha="center", fontsize=8,
            color="#1565C0", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#E3F2FD", ec="#1565C0"))
    ax.text(12.5, 4.2, "~0.1 ms/config", ha="center", fontsize=8,
            color="#C62828", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#FFEBEE", ec="#C62828"))

    # SLURM annotation
    ax.text(3.5, 1.3, "SLURM HPC\n25 CPUs × 150 jobs", ha="center",
            fontsize=7, color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1565C0", ls="--"))

    fig.savefig(os.path.join(OUT_DIR, "fig1_pipeline_overview.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved fig1_pipeline_overview.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 2: Bayesian Optimization Loop
# ─────────────────────────────────────────────────────────────
def fig2_bo_loop():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.2]})
    fig.suptitle(
        "Bayesian Optimization Loop for T8 Configuration Search",
        fontsize=14, fontweight="bold", y=0.97,
    )

    # Left panel: BO cycle diagram
    ax = axes[0]
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.axis("off")

    # 4 nodes in a cycle
    angles = [90, 0, -90, 180]  # top, right, bottom, left
    labels = [
        "1. Fit GP\nSurrogate",
        "2. Maximize\nAcquisition (EI)",
        "3. Evaluate\n(PPDF or Surrogate)",
        "4. Update\nDataset D",
    ]
    colors = ["#E8F5E9", "#E3F2FD", "#FFF3E0", "#F3E5F5"]
    details = [
        "GP(Matérn 5/2)\non D={(x,y)}",
        "x_new = argmax EI(x)\nL-BFGS-B + restarts",
        "75 min (ray-trace)\nor 0.1 ms (surrogate)",
        "D = D ∪ {(x_new, y_new)}\nrefit GP",
    ]

    r = 1.8
    node_positions = []
    for i, (ang, label, color, detail) in enumerate(zip(angles, labels, colors, details)):
        rad = np.deg2rad(ang)
        x, y = r * np.cos(rad), r * np.sin(rad)
        node_positions.append((x, y))

        box = FancyBboxPatch(
            (x - 0.9, y - 0.55), 1.8, 1.1,
            boxstyle="round,pad=0.15", facecolor=color,
            edgecolor="#333", linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.15, label, ha="center", va="center",
                fontsize=9, fontweight="bold")
        ax.text(x, y - 0.3, detail, ha="center", va="center",
                fontsize=6.5, color="#555", style="italic")

    # Arrows between nodes (clockwise: 0→1→2→3→0)
    order = [0, 1, 2, 3]
    for i in range(4):
        src = node_positions[order[i]]
        dst = node_positions[order[(i + 1) % 4]]
        dx, dy = dst[0] - src[0], dst[1] - src[1]
        norm = np.sqrt(dx**2 + dy**2)
        # Shorten arrows
        start = (src[0] + dx/norm * 1.0, src[1] + dy/norm * 0.7)
        end = (dst[0] - dx/norm * 1.0, dst[1] - dy/norm * 0.7)
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(arrowstyle="-|>", color="#333", lw=2, mutation_scale=15),
        )

    ax.text(0, 0, "BO\nLoop", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#D32F2F")

    # Right panel: conceptual GP + EI illustration
    ax2 = axes[1]
    np.random.seed(42)
    x_plot = np.linspace(0, 1, 200)

    # Fake "true" function
    true_f = 0.5 * np.sin(6 * x_plot) + 0.3 * np.cos(3 * x_plot) + 0.8

    # Observed points
    x_obs = np.array([0.05, 0.15, 0.35, 0.55, 0.7, 0.85, 0.95])
    y_obs = 0.5 * np.sin(6 * x_obs) + 0.3 * np.cos(3 * x_obs) + 0.8 + np.random.normal(0, 0.05, len(x_obs))

    # Fake GP mean and uncertainty
    from scipy.interpolate import interp1d
    gp_mean_interp = interp1d(x_obs, y_obs, kind="cubic", fill_value="extrapolate")
    gp_mean = gp_mean_interp(x_plot)
    # Higher uncertainty where fewer observations
    dist_to_nearest = np.min(np.abs(x_plot[:, None] - x_obs[None, :]), axis=1)
    gp_std = 0.08 + 0.4 * dist_to_nearest

    ax2.fill_between(x_plot, gp_mean - 2*gp_std, gp_mean + 2*gp_std,
                      alpha=0.2, color="#1976D2", label="GP 95% CI")
    ax2.fill_between(x_plot, gp_mean - gp_std, gp_mean + gp_std,
                      alpha=0.3, color="#1976D2")
    ax2.plot(x_plot, gp_mean, color="#1976D2", lw=2, label="GP mean")
    ax2.plot(x_plot, true_f, color="#333", lw=1, ls="--", alpha=0.5, label="True f(x)")
    ax2.scatter(x_obs, y_obs, c="#D32F2F", s=60, zorder=5,
                edgecolors="white", linewidths=1.5, label="Observations")

    # EI curve (fake)
    best_y = y_obs.max()
    ei = np.maximum(gp_mean - best_y, 0) + gp_std * 0.5
    ei = ei / ei.max() * 0.3
    ax2_twin = ax2.twinx()
    ax2_twin.fill_between(x_plot, 0, ei, alpha=0.15, color="#FF6F00")
    ax2_twin.plot(x_plot, ei, color="#FF6F00", lw=1.5, label="Expected Improvement")
    ax2_twin.set_ylabel("Acquisition (EI)", color="#FF6F00", fontsize=9)
    ax2_twin.tick_params(axis="y", labelcolor="#FF6F00", labelsize=8)
    ax2_twin.set_ylim(0, 0.6)

    # Mark next candidate
    next_x = x_plot[np.argmax(ei)]
    ax2.axvline(next_x, color="#FF6F00", ls="--", lw=1.5, alpha=0.7)
    ax2.text(next_x + 0.02, ax2.get_ylim()[1] * 0.95, "x_next",
             fontsize=9, color="#FF6F00", fontweight="bold")

    ax2.set_xlabel("Design parameter (normalized)", fontsize=10)
    ax2.set_ylabel("Objective (JI)", fontsize=10)
    ax2.set_title("GP Surrogate + Acquisition Function", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT_DIR, "fig2_bo_loop.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved fig2_bo_loop.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 3: Multi-Objective BO / Pareto Front
# ─────────────────────────────────────────────────────────────
def fig3_mobo_pareto():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        "Phase 2: Multi-Objective Bayesian Optimization — Pareto Front Discovery",
        fontsize=13, fontweight="bold", y=1.0,
    )

    np.random.seed(123)

    # Generate synthetic Pareto data
    n = 80
    asci = np.random.beta(5, 2, n)
    fwhm = 0.2 + 1.5 * (1 - asci) + np.random.normal(0, 0.15, n)
    fwhm = np.clip(fwhm, 0.1, 2.0)
    sens = 0.1 + 0.3 * asci + np.random.normal(0, 0.05, n)
    sens = np.clip(sens, 0.01, 0.5)

    # Identify Pareto front (maximize ASCI, minimize FWHM, maximize sensitivity)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                if (asci[j] >= asci[i] and fwhm[j] <= fwhm[i] and sens[j] >= sens[i] and
                    (asci[j] > asci[i] or fwhm[j] < fwhm[i] or sens[j] > sens[i])):
                    is_pareto[i] = False
                    break

    # Panel 1: ASCI vs FWHM
    ax = axes[0]
    ax.scatter(fwhm[~is_pareto], asci[~is_pareto], c="#90CAF9", s=30, alpha=0.6, label="Dominated")
    ax.scatter(fwhm[is_pareto], asci[is_pareto], c="#D32F2F", s=60, zorder=5,
               edgecolors="white", linewidths=1, label="Pareto-optimal")
    # Connect Pareto points
    pareto_idx = np.where(is_pareto)[0]
    pareto_sorted = pareto_idx[np.argsort(fwhm[pareto_idx])]
    ax.plot(fwhm[pareto_sorted], asci[pareto_sorted], c="#D32F2F", lw=1.5, ls="--", alpha=0.7)
    # Mark baseline
    ax.scatter([0.8], [0.65], c="#FF6F00", s=120, marker="*", zorder=6, label="Baseline\n(a=0.8, b=0.8)")
    ax.set_xlabel("FWHM (mm) \u2190 lower is better", fontsize=9)
    ax.set_ylabel("ASCI \u2192 higher is better", fontsize=9)
    ax.set_title("ASCI vs FWHM", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")
    ax.tick_params(labelsize=8)

    # Panel 2: ASCI vs Sensitivity
    ax = axes[1]
    ax.scatter(sens[~is_pareto], asci[~is_pareto], c="#90CAF9", s=30, alpha=0.6, label="Dominated")
    ax.scatter(sens[is_pareto], asci[is_pareto], c="#D32F2F", s=60, zorder=5,
               edgecolors="white", linewidths=1, label="Pareto-optimal")
    pareto_sorted2 = pareto_idx[np.argsort(sens[pareto_idx])]
    ax.plot(sens[pareto_sorted2], asci[pareto_sorted2], c="#D32F2F", lw=1.5, ls="--", alpha=0.7)
    ax.scatter([0.25], [0.65], c="#FF6F00", s=120, marker="*", zorder=6, label="Baseline")
    ax.set_xlabel("Sensitivity \u2192 higher is better", fontsize=9)
    ax.set_ylabel("ASCI \u2192 higher is better", fontsize=9)
    ax.set_title("ASCI vs Sensitivity", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=8)

    # Panel 3: Scalar JI vs MOBO comparison concept
    ax = axes[2]
    ji = sens * asci / (fwhm ** 2)
    ji_norm = (ji - ji.min()) / (ji.max() - ji.min())

    ax.scatter(range(n), np.sort(ji)[::-1], c="#1976D2", s=20, alpha=0.6)
    ji_best_idx = np.argmax(ji)
    ax.axhline(ji[ji_best_idx], color="#1976D2", ls="--", lw=1, alpha=0.5, label="JI-optimal")

    # Show that Pareto-optimal configs span a range of JI
    pareto_ji = ji[is_pareto]
    for pj in pareto_ji:
        ax.axhline(pj, color="#D32F2F", ls=":", lw=0.5, alpha=0.3)

    ax.fill_between([0, n], [pareto_ji.min()]*2, [pareto_ji.max()]*2,
                     alpha=0.1, color="#D32F2F", label="Pareto JI range")

    ax.set_xlabel("Configuration rank", fontsize=9)
    ax.set_ylabel("Joint Index (JI)", fontsize=9)
    ax.set_title("Scalar JI vs Pareto Diversity", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)

    ax.text(n * 0.5, ji.max() * 0.4,
            "MOBO reveals design\ntrade-offs hidden by\nscalar JI optimization",
            ha="center", va="center", fontsize=8, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF9C4", ec="#F9A825"))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, "fig3_mobo_pareto.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved fig3_mobo_pareto.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 4: 7-Week Timeline / Roadmap
# ─────────────────────────────────────────────────────────────
def fig4_timeline():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-1.5, 4)
    ax.axis("off")
    fig.suptitle(
        "Direction 1: 7-Week Roadmap to IEEE MIC (May 12, 2026)",
        fontsize=14, fontweight="bold", y=0.97,
    )

    weeks = [
        ("Week 1\nMar 23-29", "LHS 150 configs\nPPDF sweep (SLURM)\nBeam analysis", "#E8F5E9", "Phase 1"),
        ("Week 2\nMar 30-Apr 5", "Compute metrics + JI\nTrain GP surrogate\nValidate R²", "#C8E6C9", "Phase 1"),
        ("Week 3\nApr 6-12", "Single-obj BO on JI\n50 iterations\nConvergence analysis", "#A5D6A7", "Phase 1"),
        ("Week 4\nApr 13-19", "MOBO (qNEHVI)\nPareto fronts\n3-objective search", "#E3F2FD", "Phase 2"),
        ("Week 5\nApr 20-26", "Validate top-5 configs\nML-EM reconstruction\nGenerate figures", "#BBDEFB", "Phase 2"),
        ("Week 6\nApr 27-May 3", "Write abstract\nPolish figures\nAblation studies", "#FFF3E0", "Writing"),
        ("Week 7\nMay 4-12", "Revise\nFinal submission\nIEEE MIC deadline", "#FFCCBC", "Submit"),
    ]

    # Draw timeline bar
    ax.plot([0, 7], [2, 2], color="#333", lw=3, zorder=1)

    for i, (week, tasks, color, phase) in enumerate(weeks):
        x = i + 0.5

        # Week box
        box = FancyBboxPatch(
            (x - 0.45, 0.3), 0.9, 1.5,
            boxstyle="round,pad=0.1", facecolor=color,
            edgecolor="#333", linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x, 1.3, tasks, ha="center", va="center", fontsize=7)

        # Week label above timeline
        ax.text(x, 2.3, week, ha="center", va="bottom", fontsize=8, fontweight="bold")

        # Phase label below
        ax.text(x, -0.1, phase, ha="center", va="top", fontsize=8,
                color="#555", fontweight="bold")

        # Dot on timeline
        ax.scatter([x], [2], c=color, s=80, zorder=5, edgecolors="#333", linewidths=1.5)

    # Phase brackets
    ax.annotate("", xy=(0.05, -0.6), xytext=(3.45, -0.6),
                arrowprops=dict(arrowstyle="|-|", color="#2E7D32", lw=2))
    ax.text(1.75, -0.9, "Phase 1: Surrogate + BO", ha="center", fontsize=9,
            color="#2E7D32", fontweight="bold")

    ax.annotate("", xy=(3.55, -0.6), xytext=(5.45, -0.6),
                arrowprops=dict(arrowstyle="|-|", color="#1565C0", lw=2))
    ax.text(4.5, -0.9, "Phase 2: MOBO", ha="center", fontsize=9,
            color="#1565C0", fontweight="bold")

    # Deadline marker
    ax.annotate("IEEE MIC\nDeadline", xy=(7.0, 2), xytext=(7.0, 3.2),
                fontsize=10, fontweight="bold", color="#D32F2F",
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-|>", color="#D32F2F", lw=2))

    fig.savefig(os.path.join(OUT_DIR, "fig4_timeline.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved fig4_timeline.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 5: Design Space & T8 Ellipse Visualization
# ─────────────────────────────────────────────────────────────
def fig5_design_space():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        "T8 Design Space: Elliptical Bed Motion Parameters",
        fontsize=13, fontweight="bold", y=1.0,
    )

    # Panel 1: T8 ellipse for different (a, b) values
    ax = axes[0]
    configs = [
        (0.8, 0.8, 0, "#1976D2", "a=0.8, b=0.8 (baseline)"),
        (0.3, 0.3, 0, "#43A047", "a=0.3, b=0.3"),
        (1.2, 0.4, 0, "#E53935", "a=1.2, b=0.4"),
        (0.5, 1.3, 0, "#FF6F00", "a=0.5, b=1.3"),
    ]
    for a, b, phase, color, label in configs:
        thetas = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.deg2rad(phase)
        dx = a * np.cos(thetas)
        dy = b * np.sin(thetas)
        # Close the ellipse for visualization
        t_fine = np.linspace(0, 2*np.pi, 100)
        ax.plot(a * np.cos(t_fine), b * np.sin(t_fine), color=color, lw=1, alpha=0.4)
        ax.scatter(dx, dy, c=color, s=50, zorder=5, edgecolors="white", linewidths=1, label=label)

    ax.set_xlabel("dx (mm)", fontsize=9)
    ax.set_ylabel("dy (mm)", fontsize=9)
    ax.set_title("T8 Bed Positions\n(8 poses on ellipse)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    # Panel 2: Phase effect
    ax = axes[1]
    a, b = 0.8, 0.8
    for phase_deg, color, label in [(0, "#1976D2", "phase=0\u00b0"),
                                     (15, "#43A047", "phase=15\u00b0"),
                                     (30, "#E53935", "phase=30\u00b0"),
                                     (45, "#FF6F00", "phase=45\u00b0")]:
        thetas = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.deg2rad(phase_deg)
        dx = a * np.cos(thetas)
        dy = b * np.sin(thetas)
        ax.scatter(dx, dy, c=color, s=50, zorder=5, edgecolors="white", linewidths=1, label=label)
        for j in range(8):
            ax.annotate(str(j), (dx[j], dy[j]), fontsize=6, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points", color=color)

    t_fine = np.linspace(0, 2*np.pi, 100)
    ax.plot(a * np.cos(t_fine), b * np.sin(t_fine), color="#999", lw=1, ls="--")
    ax.set_xlabel("dx (mm)", fontsize=9)
    ax.set_ylabel("dy (mm)", fontsize=9)
    ax.set_title("Phase Rotation Effect\n(same a=b=0.8, different phase)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    # Panel 3: LHS sampling visualization (3D projected to 2D pairs)
    ax = axes[2]
    np.random.seed(42)
    from scipy.stats.qmc import LatinHypercube
    sampler = LatinHypercube(d=3, seed=42)
    samples = sampler.random(n=150)
    a_vals = samples[:, 0] * 1.4 + 0.1
    b_vals = samples[:, 1] * 1.4 + 0.1
    phase_vals = samples[:, 2] * 45.0

    scatter = ax.scatter(a_vals, b_vals, c=phase_vals, cmap="viridis",
                          s=25, alpha=0.8, edgecolors="white", linewidths=0.5)
    cb = fig.colorbar(scatter, ax=ax, label="phase (deg)", shrink=0.8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("a (mm)", fontsize=9)
    ax.set_ylabel("b (mm)", fontsize=9)
    ax.set_title("LHS Sampling (150 configs)\nColor = phase", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 1.6)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, "fig5_design_space.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved fig5_design_space.png")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig1_pipeline_overview()
    fig2_bo_loop()
    fig3_mobo_pareto()
    fig4_timeline()
    fig5_design_space()
    print(f"\nAll figures saved to: {OUT_DIR}")
