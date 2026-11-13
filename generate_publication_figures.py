#!/usr/bin/env python3
"""
Publication-Ready Figure Generator for Byzantine Reliable Broadcast Experiments.

Reads `benchmark_results.csv` produced by `networkPartitioning --benchmark` and
generates three 300-DPI IEEE-formatted figures:
  1. fig1_maxgap_vs_connectivity.png : Collapse of Mean-Field Optimism Gap
     (Delta_max) as vertex connectivity kappa(G) increases toward Dolev's (2f+1).
  2. fig2_delivery_and_rounds.png    : Two-panel comparison of Delivery Rate (%)
     and Mean Convergence Rounds across repair policies and adversary models.
  3. fig3_bandwidth_tradeoff.png     : Per-run Bandwidth (KB) vs. Convergence
     Latency trade-off across topologies and packet-loss regimes.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def configure_ieee_style() -> None:
    """Apply clean, white-background IEEE publication typography and styling."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "grid.color": "#d9d9d9",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
    })


PALETTE = {
    "Unrepaired": "#d55e00",     # Vermilion (Okabe-Ito)
    "2-Connected": "#0072b2",    # Blue
    "Dolev-(2f+1)": "#009e73",   # Bluish green
}

MARKERS = {
    "GridMesh-16": "o",
    "ErdosRenyi-16": "s",
    "BarabasiAlbert-25": "^",
    "ErdosRenyi-31": "D",
}


def plot_fig1_maxgap_vs_connectivity(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot collapse of Mean-Field Optimism Gap (MaxGap) vs. vertex connectivity."""
    sub = df[(df["adversary"] == "Omission") & (df["pLoss"] > 0.0)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, constrained_layout=True)
    loss_vals = [0.15, 0.30]

    for ax, p_loss in zip(axes, loss_vals):
        df_p = sub[np.isclose(sub["pLoss"], p_loss)]
        for topo, marker in MARKERS.items():
            df_t = df_p[df_p["topology"] == topo].sort_values("kappa_after")
            # Filter out configurations that never delivered (where MF didn't converge)
            df_t_valid = df_t[df_t["mf_rounds"] > 0]
            if df_t_valid.empty:
                continue
            ax.plot(
                df_t_valid["kappa_after"],
                df_t_valid["max_optimism_gap"],
                marker=marker,
                markersize=6.5,
                linewidth=1.8,
                label=topo,
            )

        ax.set_title(f"Link Loss $p_{{\\mathrm{{loss}}}} = {p_loss:.2f}$", pad=8)
        ax.set_xlabel("Post-Repair Vertex Connectivity $\\kappa(G)$")
        ax.set_xticks(sorted(df_p["kappa_after"].unique()))
        ax.set_ylim(-0.03, 1.08)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Peak Mean-Field Optimism Gap $\\Delta_{\\max}$")
    axes[0].legend(title="Topology", frameon=True, edgecolor="#cccccc", loc="upper right")
    fig.suptitle(
        "Collapse of Mean-Field Optimism Bias $\\Delta_{\\max}$ Under Dolev $(2f+1)$ Augmentation",
        fontsize=12,
        fontweight="bold",
    )

    out_path = out_dir / "fig1_maxgap_vs_connectivity.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig2_delivery_and_rounds(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot Delivery Rate (%) and Mean Convergence Rounds at pLoss = 0.15."""
    sub = df[np.isclose(df["pLoss"], 0.15) & (df["adversary"] == "Omission")].copy()
    topos = ["GridMesh-16", "ErdosRenyi-16", "BarabasiAlbert-25", "ErdosRenyi-31"]
    policies = ["Unrepaired", "2-Connected", "Dolev-(2f+1)"]

    x = np.arange(len(topos))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    for idx, pol in enumerate(policies):
        df_pol = sub[sub["policy"] == pol].set_index("topology").reindex(topos)
        offset = (idx - 1) * width

        deliv = df_pol["delivery_rate"].values * 100.0
        rounds = df_pol["mc_mean_rounds"].values

        bars1 = ax1.bar(
            x + offset, deliv, width,
            label=pol, color=PALETTE[pol], edgecolor="#222222", linewidth=0.6
        )
        ax2.bar(
            x + offset, np.minimum(rounds, 25.0), width,
            label=pol, color=PALETTE[pol], edgecolor="#222222", linewidth=0.6
        )

        for b, val in zip(bars1, deliv):
            ax1.text(
                b.get_x() + b.get_width() / 2.0,
                val + 2.0,
                f"{val:.0f}%",
                ha="center", va="bottom", fontsize=7.5
            )

        for b_idx, val in enumerate(rounds):
            disp_y = min(val, 25.0)
            lbl = "Timeout" if val >= 199.0 else f"{val:.1f}"
            ax2.text(
                x[b_idx] + offset,
                disp_y + 0.5,
                lbl,
                ha="center", va="bottom", fontsize=7.2, rotation=30 if val >= 199.0 else 0
            )

    ax1.set_title("Quorum Delivery Rate ($p_{\\mathrm{loss}} = 0.15$, Omission)")
    ax1.set_ylabel("Successful Delivery Rate (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(topos, rotation=12, ha="right")
    ax1.set_ylim(0, 115)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.set_title("Mean Convergence Rounds (Capped at 25 for Stalled Runs)")
    ax2.set_ylabel("Mean Rounds to 3-Phase Delivery")
    ax2.set_xticks(x)
    ax2.set_xticklabels(topos, rotation=12, ha="right")
    ax2.set_ylim(0, 29)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(title="Repair Policy", frameon=True, edgecolor="#cccccc", loc="upper right")

    out_path = out_dir / "fig2_delivery_and_rounds.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig3_bandwidth_tradeoff(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot Bandwidth (KB/run) vs. Convergence Latency across delivered runs."""
    delivered = df[(df["delivery_rate"] > 0.90) & (df["adversary"] == "Omission")].copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)

    for pol in ["Unrepaired", "2-Connected", "Dolev-(2f+1)"]:
        df_pol = delivered[delivered["policy"] == pol]
        for topo, marker in MARKERS.items():
            sub = df_pol[df_pol["topology"] == topo]
            if sub.empty:
                continue
            ax.scatter(
                sub["mc_mean_rounds"],
                sub["mean_kb"],
                color=PALETTE[pol],
                marker=marker,
                s=65,
                edgecolors="#222222",
                linewidths=0.6,
                alpha=0.9,
                label=f"{pol} ({topo})" if pol == "Dolev-(2f+1)" else None,
            )

    # Clean legend separating Policy (color) and Topology (shape)
    policy_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[p],
                         markeredgecolor="#222222", markersize=8, label=p)
        for p in ["Unrepaired", "2-Connected", "Dolev-(2f+1)"]
    ]
    topo_handles = [
        mpl.lines.Line2D([0], [0], marker=m, color="w", markerfacecolor="#666666",
                         markeredgecolor="#222222", markersize=7.5, label=t)
        for t, m in MARKERS.items()
    ]

    leg1 = ax.legend(handles=policy_handles, title="Repair Policy", loc="upper right",
                     frameon=True, edgecolor="#cccccc")
    ax.add_artist(leg1)
    ax.legend(handles=topo_handles, title="Topology", loc="center right",
              bbox_to_anchor=(0.99, 0.54), frameon=True, edgecolor="#cccccc")

    ax.set_title("Communication Bandwidth vs. Convergence Latency (Delivered Runs)")
    ax.set_xlabel("Mean Convergence Rounds (Monte Carlo)")
    ax.set_ylabel("Mean Bandwidth per Broadcast (KB)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = out_dir / "fig3_bandwidth_tradeoff.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IEEE-style figures from benchmark_results.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmark_results.csv"),
        help="Path to benchmark_results.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory to write generated PNG figures",
    )
    args = parser.parse_args()

    configure_ieee_style()
    df = pd.read_csv(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    f1 = plot_fig1_maxgap_vs_connectivity(df, args.out_dir)
    f2 = plot_fig2_delivery_and_rounds(df, args.out_dir)
    f3 = plot_fig3_bandwidth_tradeoff(df, args.out_dir)

    print("Generated publication figures:")
    for p in (f1, f2, f3):
        print(f"  - {p.resolve()}")


if __name__ == "__main__":
    main()