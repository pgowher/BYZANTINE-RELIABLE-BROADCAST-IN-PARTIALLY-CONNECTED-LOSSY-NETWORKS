#!/usr/bin/env python3
"""
A*-Conference-Grade Publication Figure Generator (Seaborn + SciPy + 3D Surface).

Reads `benchmark_results.csv` produced by `networkPartitioning --benchmark` and
generates three multi-panel, 300-DPI figures suitable for IEEE TPDS / ICDCS / SOSP:
  1. fig1_optimism_bias_landscape.png : 3D interpolated surface of MaxGap(kappa, pLoss)
     with exact observed stems + annotated Seaborn heatmap across topologies/policies.
  2. fig2_adversarial_resilience_matrix.png : Multi-panel latency (Mean + P95 tail
     whiskers + Mean-Field prediction markers) and Delivery Rate heatmap across
     Omission vs. Equivocation adversaries.
  3. fig3_pareto_cost_jointplot.png : Bandwidth-vs-P95 Latency Pareto frontier,
     log-scale cryptographic verification cost under Equivocation, and Max-vs-Mean
     correlation bias scaling with 95% CI regression.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata


def configure_a_star_style() -> None:
    """Apply clean, white-background A*-conference styling via Seaborn + Matplotlib."""
    sns.set_theme(style="whitegrid", context="paper", font="serif")
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.9,
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })


POLICY_ORDER = ["Unrepaired", "2-Connected", "Dolev-(2f+1)"]
TOPO_ORDER = ["GridMesh-16", "ErdosRenyi-16", "BarabasiAlbert-25", "ErdosRenyi-31"]
POLICY_PALETTE = {
    "Unrepaired": "#d55e00",
    "2-Connected": "#0072b2",
    "Dolev-(2f+1)": "#009e73",
}
TOPO_MARKERS = {
    "GridMesh-16": "o",
    "ErdosRenyi-16": "s",
    "BarabasiAlbert-25": "^",
    "ErdosRenyi-31": "D",
}


def plot_fig1_optimism_landscape(df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 1: 3D Optimism Bias Surface + Annotated Seaborn Heatmap."""
    sub = df[(df["adversary"] == "Omission") & (df["mf_rounds"] > 0)].copy()

    fig = plt.figure(figsize=(13.2, 5.2), constrained_layout=True)
    gs = GridSpec(1, 2, width_ratios=[1.18, 1.0], figure=fig)

    # Panel A: 3D Surface over (kappa_after, pLoss) -> max_optimism_gap
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    x = sub["kappa_after"].values.astype(float)
    y = sub["pLoss"].values.astype(float)
    z = sub["max_optimism_gap"].values.astype(float)

    xi = np.linspace(x.min(), x.max(), 40)
    yi = np.linspace(y.min(), y.max(), 40)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((x, y), z, (XI, YI), method="linear")
    # Fill convex-hull exterior smoothly with nearest neighbor for visual continuity
    ZI_near = griddata((x, y), z, (XI, YI), method="nearest")
    ZI = np.where(np.isnan(ZI), ZI_near, ZI)
    ZI = np.clip(ZI, 0.0, 1.0)

    surf = ax3d.plot_surface(
        XI, YI, ZI,
        cmap="viridis",
        alpha=0.82,
        edgecolor="#334155",
        linewidth=0.2,
        antialiased=True,
    )
    ax3d.scatter(x, y, z, color="#e11d48", edgecolor="white", s=42, depthshade=False, zorder=10, label="Observed Runs")
    for xv, yv, zv in zip(x, y, z):
        ax3d.plot([xv, xv], [yv, yv], [0, zv], color="#64748b", linestyle=":", linewidth=0.7, alpha=0.65)

    ax3d.set_xlabel("Vertex Connectivity $\\kappa(G)$", labelpad=8)
    ax3d.set_ylabel("Packet Loss $p_{\\mathrm{loss}}$", labelpad=8)
    ax3d.set_zlabel("Peak Optimism Gap $\\Delta_{\\max}$", labelpad=8)
    ax3d.set_zlim(0.0, 1.05)
    ax3d.view_init(elev=26, azim=-48)
    ax3d.set_title("(a) Mean-Field Optimism Surface $\\Delta_{\\max}(\\kappa, p_{\\mathrm{loss}})$", pad=10)
    fig.colorbar(surf, ax=ax3d, shrink=0.62, aspect=14, pad=0.08, label="$\\Delta_{\\max}$")

    # Panel B: Annotated Seaborn Heatmap across (Topology, Policy) x pLoss
    ax_hm = fig.add_subplot(gs[0, 1])
    om_all = df[df["adversary"] == "Omission"].copy()
    om_all["RowLabel"] = om_all["topology"] + " | " + om_all["policy"] + " ($\\kappa$=" + om_all["kappa_after"].astype(str) + ")"

    ordered_rows = []
    for t in TOPO_ORDER:
        for p in POLICY_ORDER:
            matches = om_all[(om_all["topology"] == t) & (om_all["policy"] == p)]["RowLabel"].unique()
            ordered_rows.extend(matches)

    pivot_gap = om_all.pivot(index="RowLabel", columns="pLoss", values="max_optimism_gap").reindex(ordered_rows)
    pivot_deliv = om_all.pivot(index="RowLabel", columns="pLoss", values="delivery_rate").reindex(ordered_rows)

    annot_labels = pivot_gap.copy().astype(object)
    for r_idx in pivot_gap.index:
        for c_idx in pivot_gap.columns:
            g_val = pivot_gap.loc[r_idx, c_idx]
            d_val = pivot_deliv.loc[r_idx, c_idx]
            if d_val < 0.01:
                annot_labels.loc[r_idx, c_idx] = "Stalled"
            else:
                annot_labels.loc[r_idx, c_idx] = f"{g_val:.3f}"

    sns.heatmap(
        pivot_gap,
        ax=ax_hm,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        annot=annot_labels,
        fmt="",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Peak Optimism Gap $\\Delta_{\\max}$", "shrink": 0.85},
        annot_kws={"size": 8.0, "weight": "medium"},
    )
    ax_hm.set_title("(b) Optimism Bias Across Topologies & Augmentation Policies", pad=10)
    ax_hm.set_xlabel("Link Packet Loss Probability $p_{\\mathrm{loss}}$")
    ax_hm.set_ylabel("")
    ax_hm.set_xticklabels([f"{c:.2f}" for c in pivot_gap.columns])

    fig.suptitle(
        "Topological De-Correlation of Mean-Field Optimism Bias Under Dolev $(2f+1)$ Repair",
        fontsize=12.5,
        fontweight="bold",
    )

    out_path = out_dir / "fig1_optimism_bias_landscape.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig2_adversarial_matrix(df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 2: Multi-Panel Latency (Mean + P95 + MF) & Delivery Heatmap."""
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), constrained_layout=True)

    # Top Row: Mean vs P95 Tail Latency + Mean-Field Prediction under Omission (pLoss=0.15, 0.30)
    for col_idx, p_loss in enumerate([0.15, 0.30]):
        ax = axes[0, col_idx]
        sub = df[(df["adversary"] == "Omission") & np.isclose(df["pLoss"], p_loss)].copy()

        x = np.arange(len(TOPO_ORDER))
        width = 0.25

        for p_idx, pol in enumerate(POLICY_ORDER):
            df_p = sub[sub["policy"] == pol].set_index("topology").reindex(TOPO_ORDER)
            offset = (p_idx - 1) * width
            mc_mean = np.minimum(df_p["mc_mean_rounds"].values, 24.0)
            mc_p95 = np.minimum(df_p["mc_p95_rounds"].values, 24.0)
            mf_rd = df_p["mf_rounds"].values

            ax.bar(
                x + offset, mc_mean, width,
                color=POLICY_PALETTE[pol], alpha=0.88, edgecolor="#1e293b", linewidth=0.7,
                label=f"{pol} (MC Mean)" if col_idx == 0 else None,
            )

            # Plot P95 tail whiskers above mean
            yerr_upper = np.maximum(0.0, mc_p95 - mc_mean)
            ax.errorbar(
                x + offset, mc_mean,
                yerr=[np.zeros_like(yerr_upper), yerr_upper],
                fmt="none", ecolor="#0f172a", elinewidth=1.3, capsize=3.5, zorder=5,
            )

            # Overlay Mean-Field predicted rounds (diamond)
            valid_mf = mf_rd > 0
            ax.scatter(
                (x + offset)[valid_mf],
                mf_rd[valid_mf],
                marker="D", s=34, color="white", edgecolor="#0f172a", linewidth=1.1, zorder=8,
                label="Mean-Field Prediction" if (col_idx == 0 and p_idx == 0) else None,
            )

        ax.axhline(24.0, color="#991b1b", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.text(3.35, 24.3, "Timeout Cap", color="#991b1b", fontsize=7.5, ha="right")
        ax.set_title(f"({'a' if col_idx == 0 else 'b'}) Omission Latency: Mean, P95 Tail & Mean-Field ($p_{{\\mathrm{{loss}}}}={p_loss:.2f}$)")
        ax.set_ylabel("Rounds to Quorum Delivery")
        ax.set_xticks(x)
        ax.set_xticklabels(TOPO_ORDER, rotation=10, ha="right")
        ax.set_ylim(0, 31.5)
        sns.despine(ax=ax)

    axes[0, 0].legend(loc="upper right", frameon=True, edgecolor="#cbd5e1", ncol=2, fontsize=7.6)

    # Bottom Row: Delivery Rate (%) Heatmaps for Omission vs Equivocation
    for col_idx, adv in enumerate(["Omission", "Equivocation"]):
        ax = axes[1, col_idx]
        sub = df[df["adversary"] == adv].copy()
        sub["RowLabel"] = sub["topology"] + " | " + sub["policy"]

        ordered_rows = [f"{t} | {p}" for t in TOPO_ORDER for p in POLICY_ORDER]
        piv = sub.pivot(index="RowLabel", columns="pLoss", values="delivery_rate").reindex(ordered_rows) * 100.0

        sns.heatmap(
            piv,
            ax=ax,
            cmap="crest",
            vmin=0.0,
            vmax=100.0,
            annot=True,
            fmt=".1f",
            linewidths=0.6,
            linecolor="white",
            cbar_kws={"label": "Delivery Rate (%)", "shrink": 0.9},
            annot_kws={"size": 8.0},
        )
        panel_lbl = "c" if col_idx == 0 else "d"
        ax.set_title(f"({panel_lbl}) 3-Phase Bracha Delivery Rate (%) — {adv} Adversary")
        ax.set_xlabel("Link Packet Loss Probability $p_{\\mathrm{loss}}$")
        ax.set_ylabel("")
        ax.set_xticklabels([f"{c:.2f}" for c in piv.columns])

    fig.suptitle(
        "Adversarial Resilience, Tail Latency, and Liveness Boundaries Across Topologies",
        fontsize=12.5,
        fontweight="bold",
    )

    out_path = out_dir / "fig2_adversarial_resilience_matrix.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig3_pareto_and_crypto(df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 3: Bandwidth-Latency Pareto Frontier, Crypto Cost, and Bias Regression."""
    fig = plt.figure(figsize=(13.5, 4.6), constrained_layout=True)
    gs = GridSpec(1, 3, width_ratios=[1.25, 1.0, 1.0], figure=fig)

    # Panel A: Pareto Trade-off (Bandwidth KB vs. P95 Latency)
    ax1 = fig.add_subplot(gs[0, 0])
    deliv = df[(df["delivery_rate"] > 0.90) & (df["adversary"] == "Omission")].copy()

    sns.scatterplot(
        data=deliv,
        x="mc_p95_rounds",
        y="mean_kb",
        hue="policy",
        hue_order=POLICY_ORDER,
        palette=POLICY_PALETTE,
        style="topology",
        markers=TOPO_MARKERS,
        size="pLoss",
        sizes=(55, 135),
        edgecolor="#1e293b",
        linewidth=0.7,
        alpha=0.92,
        ax=ax1,
    )
    ax1.axvspan(4.5, 7.2, color="#10b981", alpha=0.09, zorder=0)
    ax1.set_ylim(70, 390)
    ax1.text(5.85, 370, "Dolev (2f+1)\nLow-Latency Zone", color="#047857", fontsize=7.6, ha="center", fontweight="bold")
    ax1.set_title("(a) Bandwidth vs. P95 Latency Trade-Off")
    ax1.set_xlabel("P95 Convergence Rounds (Monte Carlo)")
    ax1.set_ylabel("Mean Bandwidth per Broadcast (KB)")
    ax1.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", borderaxespad=0, fontsize=7.0, frameon=True)
    sns.despine(ax=ax1)

    # Panel B: Cryptographic Signature Verifications (Omission vs. Equivocation at pLoss=0.15)
    ax2 = fig.add_subplot(gs[0, 1])
    sub_crypto = df[(df["policy"] == "Dolev-(2f+1)") & np.isclose(df["pLoss"], 0.15)].copy()
    sns.barplot(
        data=sub_crypto,
        x="topology",
        y="mean_sig_verifications",
        hue="adversary",
        order=TOPO_ORDER,
        palette={"Omission": "#0072b2", "Equivocation": "#e11d48"},
        edgecolor="#1e293b",
        linewidth=0.7,
        ax=ax2,
    )
    ax2.set_yscale("log")
    ax2.set_title("(b) Crypto Verifications ($p_{\\mathrm{loss}}=0.15$)")
    ax2.set_xlabel("")
    ax2.set_ylabel("Signature Verifications / Run (Log Scale)")
    ax2.tick_params(axis="x", rotation=15)
    ax2.legend(title="Adversary", frameon=True, edgecolor="#cbd5e1", loc="upper left")
    sns.despine(ax=ax2)

    # Panel C: Regression of Mean-Field Peak Gap vs. Network-Wide Mean Gap
    ax3 = fig.add_subplot(gs[0, 2])
    reg_df = df[(df["adversary"] == "Omission") & (df["mf_rounds"] > 0) & (df["pLoss"] > 0)].copy()
    sns.regplot(
        data=reg_df,
        x="mean_optimism_gap",
        y="max_optimism_gap",
        scatter=False,
        color="#334155",
        line_kws={"linewidth": 1.6, "linestyle": "--"},
        ax=ax3,
    )
    sns.scatterplot(
        data=reg_df,
        x="mean_optimism_gap",
        y="max_optimism_gap",
        hue="policy",
        hue_order=POLICY_ORDER,
        palette=POLICY_PALETTE,
        s=75,
        edgecolor="#1e293b",
        linewidth=0.7,
        ax=ax3,
        legend=False,
    )
    ax3.set_title("(c) Peak vs. Mean Bias (95% CI)")
    ax3.set_xlabel("Network-Wide Mean Bias $\\bar{\\Delta}$")
    ax3.set_ylabel("Peak Bottleneck Bias $\\Delta_{\\max}$")
    sns.despine(ax=ax3)

    fig.suptitle(
        "Systems Cost-Latency Frontier, Cryptographic Overhead, and Correlation Scaling",
        fontsize=12.5,
        fontweight="bold",
    )

    out_path = out_dir / "fig3_pareto_cost_jointplot.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate A*-conference figures using Seaborn, SciPy & 3D Matplotlib.")
    parser.add_argument("--csv", type=Path, default=Path("benchmark_results.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    configure_a_star_style()
    df = pd.read_csv(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    f1 = plot_fig1_optimism_landscape(df, args.out_dir)
    f2 = plot_fig2_adversarial_matrix(df, args.out_dir)
    f3 = plot_fig3_pareto_and_crypto(df, args.out_dir)

    print("Generated A*-conference figures:")
    for p in (f1, f2, f3):
        print(f"  - {p.resolve()}")


if __name__ == "__main__":
    main()