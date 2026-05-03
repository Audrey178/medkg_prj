#!/usr/bin/env python3
"""
Generate paper figures from experiment JSON results.
Produces publication-quality figures for draft_v2.tex.
"""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES = PROJECT_ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Consistent color palette
COLORS = {
    "primekg": "#888888",      # Gray
    "hpoa": "#4477AA",         # Blue
    "orphadata": "#EE6677",    # Red-pink
    "phenopackets": "#CCBB44", # Yellow
    "chronomedkg": "#228833", # Green (our contribution)
    "ta_novel": "#117733",     # Darker green
    "struct": "#BBBBBB",       # Light gray for baseline
    "temporal": "#228833",     # Green for temporal
}


# ============================================================================
# Figure 2: Coverage Gap
# ============================================================================
def fig_coverage_gap():
    with open(PROJECT_ROOT / "data/benchmark/coverage_gap_analysis.json") as f:
        data = json.load(f)

    cov = data["coverage"]
    sources = ["PrimeKG", "Phenopackets", "HPOA", "Orphadata", "ChronoMedKG", "TA novel"]
    keys = ["primekg", "phenopackets", "hpoa", "orphadata", "chronomedkg", "ta_novel"]
    diseases = [cov[k]["diseases"] for k in keys]
    pcts = [cov[k]["pct"] for k in keys]
    colors = [COLORS[k] for k in keys]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar(sources, diseases, color=colors, edgecolor="black", linewidth=0.7)

    # Annotate
    for bar, d, p in zip(bars, diseases, pcts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 150,
                f"{d:,}\n({p}%)",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold" if d >= 5000 else "normal")

    ax.set_ylabel("Diseases with onset data")
    ax.set_title("Temporal coverage across biomedical resources (out of 17,080 PrimeKG diseases)")
    ax.set_ylim(0, max(diseases) * 1.18)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticklabels(sources, rotation=15, ha="right")

    # Legend / annotation for granularity
    ax.text(0.02, 0.97,
            "ChronoMedKG provides per-phenotype onset (median 5 pairs/disease)\n"
            "vs. coarse disease-level bins in HPOA/Orphadata (1 range/disease)",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFFBE6", ec="#AAAAAA", lw=0.5))

    plt.tight_layout()
    plt.savefig(FIGURES / "fig2_coverage_gap.png", dpi=300)
    plt.savefig(FIGURES / "fig2_coverage_gap.pdf")
    plt.close()
    print(f"Saved: fig2_coverage_gap (png, pdf)")


# ============================================================================
# Figure 3: Evidence Decay
# ============================================================================
def fig_evidence_decay():
    with open(PROJECT_ROOT / "data/benchmark/evidence_decay_audit.json") as f:
        data = json.load(f)

    year_dist = data["ta_evidence_age_distribution"]["year_distribution"]
    # Flatten and aggregate by 5-year bins
    years = []
    for y, c in year_dist.items():
        years.extend([int(y)] * c)

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    # Use 2-year bins
    bins = range(1960, 2028, 2)
    ax.hist(years, bins=bins, color=COLORS["chronomedkg"], edgecolor="black", linewidth=0.4, alpha=0.85)

    ax.axvline(statistics.median(years), color="red", linestyle="--", linewidth=1.2,
               label=f"Median: {statistics.median(years):.0f}")

    ax.set_xlabel("Publication year of supporting evidence")
    ax.set_ylabel("ChronoMedKG triples")
    ax.set_title(
        "Evidence age distribution across 455,519 ChronoMedKG triples\n"
        "(PrimeKG has zero evidence dates at the edge level)",
        fontsize=11
    )
    ax.set_xlim(1960, 2026)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    # Annotations
    recent = sum(1 for y in years if y >= 2021)
    old = sum(1 for y in years if y < 2006)
    ax.text(0.02, 0.95,
            f"Last 5y (2021-2026): {recent:,} ({100*recent/len(years):.1f}%)\n"
            f">20y old (pre-2006): {old:,} ({100*old/len(years):.1f}%)",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFFBE6", ec="#AAAAAA", lw=0.5))

    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.75))
    plt.tight_layout()
    plt.savefig(FIGURES / "fig3_evidence_decay.png", dpi=300)
    plt.savefig(FIGURES / "fig3_evidence_decay.pdf")
    plt.close()
    print(f"Saved: fig3_evidence_decay (png, pdf)")


# ============================================================================
# Figure 4: Disease Trajectory Clustering (t-SNE)
# ============================================================================
def fig_trajectory_clustering():
    with open(PROJECT_ROOT / "data/benchmark/trajectory_clustering.json") as f:
        data = json.load(f)

    profiles = data.get("cluster_profiles", {})
    diseases = data.get("diseases", {})

    # Build data arrays
    xs, ys, labels = [], [], []
    for did, d in diseases.items():
        if d.get("tsne_x") is not None and d.get("cluster") is not None:
            xs.append(d["tsne_x"])
            ys.append(d["tsne_y"])
            labels.append(d["cluster"])

    cluster_ids = sorted(set(labels))
    cluster_palette = ["#228833", "#EE6677", "#4477AA", "#CCBB44", "#66CCEE", "#AA3377"]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, cid in enumerate(cluster_ids):
        mask = [lab == cid for lab in labels]
        cx = [xs[j] for j in range(len(xs)) if mask[j]]
        cy = [ys[j] for j in range(len(ys)) if mask[j]]
        profile = profiles.get(str(cid), {})
        archetype = profile.get("archetype", f"Cluster {cid}")
        # Rename verbose/duplicated archetype labels for the figure caption.
        # Source JSON stores legacy verbose names; paper text uses the short form.
        archetype_aliases = {
            "Progressive Childhood (wide progression) — Progressive": "Broad-onset Progressive",
        }
        archetype = archetype_aliases.get(archetype, archetype)
        n = profile.get("n_diseases", len(cx))
        ax.scatter(cx, cy, s=5, alpha=0.5, c=cluster_palette[i % len(cluster_palette)],
                   label=f"{archetype} (n={n:,})")

    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title(
        "Disease trajectory archetypes discovered by unsupervised clustering\n"
        "of 8,935 diseases with temporal features (silhouette = 0.362)",
        fontsize=11
    )
    ax.legend(loc="best", markerscale=2.5, fontsize=8.5, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig4_trajectory_clustering.png", dpi=300)
    plt.savefig(FIGURES / "fig4_trajectory_clustering.pdf")
    plt.close()
    print(f"Saved: fig4_trajectory_clustering (png, pdf)")


# ============================================================================
# Figure 5: Link Prediction (KEY NEW RESULT)
# ============================================================================
def fig_link_prediction():
    with open(PROJECT_ROOT / "data/benchmark/link_prediction_v3/link_prediction_seeds.json") as f:
        data = json.load(f)

    s = data["summary"]
    conditions = ["hpoa_struct", "hpoa_temporal", "ta_struct", "ta_temporal"]
    labels = ["HPOA\nstructure", "HPOA\n+ coarse\ntemporal", "TA\nstructure", "TA\n+ fine\ntemporal"]

    mrr_means = [s[c]["mrr_mean"] for c in conditions]
    mrr_stds = [s[c]["mrr_std"] for c in conditions]
    hits_means = [s[c]["hits_at_10_mean"] for c in conditions]
    hits_stds = [s[c]["hits_at_10_std"] for c in conditions]

    colors = [COLORS["struct"], COLORS["hpoa"], COLORS["struct"], COLORS["chronomedkg"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # MRR subplot
    x = np.arange(len(labels))
    bars1 = ax1.bar(x, mrr_means, yerr=mrr_stds, capsize=5, color=colors,
                    edgecolor="black", linewidth=0.7, error_kw=dict(lw=1, capthick=1))
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Mean Reciprocal Rank (MRR)")
    ax1.set_title("Link prediction MRR (TransE, 3 seeds)")
    ax1.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax1.set_axisbelow(True)
    ax1.set_ylim(0, max(mrr_means) * 1.35)

    # Gain arrows
    hpoa_gain = 100 * (mrr_means[1] - mrr_means[0]) / mrr_means[0]
    ta_gain = 100 * (mrr_means[3] - mrr_means[2]) / mrr_means[2]

    # Annotations: bars values
    for i, (bar, m, sd) in enumerate(zip(bars1, mrr_means, mrr_stds)):
        ax1.text(bar.get_x() + bar.get_width() / 2, m + sd + 0.002,
                 f"{m:.4f}", ha="center", va="bottom", fontsize=8.5)

    # Gain annotations
    ax1.annotate("", xy=(1, max(mrr_means) * 1.15), xytext=(0, max(mrr_means) * 1.15),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax1.text(0.5, max(mrr_means) * 1.22, f"+{hpoa_gain:.1f}%\n(p=0.003)",
             ha="center", fontsize=8.5, fontweight="bold", color="#4477AA")

    ax1.annotate("", xy=(3, max(mrr_means) * 1.15), xytext=(2, max(mrr_means) * 1.15),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax1.text(2.5, max(mrr_means) * 1.22, f"+{ta_gain:.1f}%\n(p=0.015)",
             ha="center", fontsize=8.5, fontweight="bold", color=COLORS["chronomedkg"])

    # Hits@10 subplot
    bars2 = ax2.bar(x, hits_means, yerr=hits_stds, capsize=5, color=colors,
                    edgecolor="black", linewidth=0.7, error_kw=dict(lw=1, capthick=1))
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Hits@10")
    ax2.set_title("Link prediction Hits@10")
    ax2.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax2.set_axisbelow(True)

    for bar, m, sd in zip(bars2, hits_means, hits_stds):
        ax2.text(bar.get_x() + bar.get_width() / 2, m + sd + 0.003,
                 f"{m:.3f}", ha="center", va="bottom", fontsize=8.5)

    # Overall figure title
    fig.suptitle(
        "Temporal features significantly improve link prediction in both systems;\n"
        "TA's fine-grained temporal gives larger relative gains than HPOA's coarse bins",
        fontsize=10, y=1.02
    )

    plt.tight_layout()
    plt.savefig(FIGURES / "fig5_link_prediction.png", dpi=300)
    plt.savefig(FIGURES / "fig5_link_prediction.pdf")
    plt.close()
    print(f"Saved: fig5_link_prediction (png, pdf)")


# ============================================================================
# Figure 7: Error Taxonomy Pie
# ============================================================================
def fig_error_taxonomy():
    # Based on error_taxonomy_v2 numbers (from paper draft)
    labels = [
        "Contained (correct)\n50.1%",
        "Adjacent stage\n15.6%",
        "Granularity mismatch\n13.8%",
        "TA wider but overlaps\n6.7%",
        "Single-triple noise\n5.7%",
        "Genuinely wrong\n7.3%",
    ]
    sizes = [50.1, 15.6, 13.8, 6.7, 5.7, 7.3]
    colors_p = ["#228833", "#CCBB44", "#CCBB44", "#CCBB44", "#CCBB44", "#EE6677"]
    explode = (0, 0, 0, 0, 0, 0.08)  # Highlight genuine error

    fig, ax = plt.subplots(figsize=(7, 5.5))
    wedges, texts = ax.pie(sizes, explode=explode, colors=colors_p,
                            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})

    # Custom labels OUTSIDE (more readable)
    for i, (wedge, label) in enumerate(zip(wedges, labels)):
        ang = (wedge.theta2 + wedge.theta1) / 2
        x = 1.25 * np.cos(np.deg2rad(ang))
        y = 1.25 * np.sin(np.deg2rad(ang))
        ha = "left" if x > 0 else "right"
        ax.annotate(label, xy=(np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))),
                    xytext=(x, y), ha=ha, fontsize=9, fontweight="bold" if i == 5 else "normal",
                    color="#CC0000" if i == 5 else "black")

    # Center text
    ax.text(0, 0, "Error Taxonomy v2\n(n=2,563)", ha="center", va="center",
            fontsize=11, fontweight="bold")

    # Legend
    legend_items = [
        mpatches.Patch(color="#228833", label="Correct"),
        mpatches.Patch(color="#CCBB44", label="Granularity / boundary issue (not an error)"),
        mpatches.Patch(color="#EE6677", label="Genuine error"),
    ]
    ax.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=True, fontsize=9)

    ax.set_title("Only 7.3% of disagreements are genuine errors", fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig7_error_taxonomy.png", dpi=300)
    plt.savefig(FIGURES / "fig7_error_taxonomy.pdf")
    plt.close()
    print(f"Saved: fig7_error_taxonomy (png, pdf)")


def main():
    print("Generating paper figures from JSON results...")
    print(f"Output directory: {FIGURES}")
    print()

    fig_coverage_gap()
    fig_evidence_decay()
    fig_trajectory_clustering()
    fig_link_prediction()
    fig_error_taxonomy()

    print()
    print(f"All figures saved to {FIGURES}")
    print("Files:")
    for f in sorted(FIGURES.glob("fig*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
