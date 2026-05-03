#!/usr/bin/env python3
"""
Generate PMC Clinical Case Timelines figure for Appendix A1.

Shows diagnostic delays as horizontal bars: red for misdiagnosis period,
green for correct diagnosis.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES = PROJECT_ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def main():
    with open(PROJECT_ROOT / "data/validation_sources/pmc_clinical_cases_all.json") as f:
        cases = json.load(f)

    # Filter to cases with misdiagnosis and delay
    valid_cases = [c for c in cases
                   if c.get("diagnostic_delay_years") not in (None, "None")
                   and c.get("misdiagnosis") not in (None, "None", "")]

    # Sort by diagnostic delay (ascending for visual impact)
    valid_cases.sort(key=lambda c: c.get("diagnostic_delay_years", 0))

    # Take top 15 for visualization (too many cases = too cluttered)
    # Select: some short, medium, long delays for variety
    if len(valid_cases) > 15:
        # Stratified: take 5 shortest, 5 mid, 5 longest
        short = valid_cases[:5]
        mid = valid_cases[len(valid_cases)//2 - 2:len(valid_cases)//2 + 3]
        long_ = valid_cases[-5:]
        selected = short + mid + long_
    else:
        selected = valid_cases

    selected.sort(key=lambda c: c.get("diagnostic_delay_years", 0))

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(selected) + 1.5)))

    for i, case in enumerate(selected):
        delay = case.get("diagnostic_delay_years", 0)
        try:
            delay = float(delay)
        except (ValueError, TypeError):
            continue
        age_presentation = case.get("patient_age_at_presentation", 0)
        try:
            age_presentation = float(age_presentation)
        except (ValueError, TypeError):
            age_presentation = 0

        # Misdiagnosis period: from presentation to correct dx
        start_age = age_presentation
        end_age = age_presentation + delay

        # Red bar: misdiagnosis period
        ax.barh(i, delay, left=start_age, height=0.55,
                color="#EE6677", edgecolor="black", linewidth=0.5,
                label="Misdiagnosis period" if i == 0 else None)

        # Green tick: correct diagnosis
        ax.plot(end_age, i, marker=">", color="#228833", markersize=10,
                markeredgecolor="black", markeredgewidth=0.8,
                label="Correct diagnosis" if i == 0 else None)

        # Label on left: disease + misdiagnosis
        disease = case["correct_diagnosis"][:38]
        misdx = case.get("misdiagnosis", "?")[:25]
        ax.text(-0.5, i, f"{disease}\n   ← {misdx}",
                ha="right", va="center", fontsize=8,
                fontweight="bold", color="#333333")

        # Label on right: delay + PMC ID
        ax.text(end_age + 0.8, i, f"{delay:.1f}y delay  [{case['pmc_id']}]",
                ha="left", va="center", fontsize=8, color="#555555")

    ax.set_yticks([])
    ax.set_xlabel("Patient age (years)")
    ax.set_title(
        f"Clinical diagnostic odysseys from PubMed Central open-access case reports\n"
        f"(selected {len(selected)} of {len(valid_cases)} cases with misdiagnosis; full list in Table S6)",
        fontsize=10
    )
    ax.set_xlim(-12, max(float(c.get("patient_age_at_presentation", 0) or 0) +
                        float(c.get("diagnostic_delay_years", 0) or 0)
                        for c in selected) + 15)
    ax.set_ylim(-0.6, len(selected) - 0.4)
    ax.axvline(0, color="#999999", linestyle=":", linewidth=0.5)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Legend
    legend_items = [
        mpatches.Patch(color="#EE6677", label="Misdiagnosis period"),
        plt.Line2D([0], [0], marker=">", color="w",
                   markerfacecolor="#228833", markersize=10,
                   label="Correct diagnosis"),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig_a1_pmc_timelines.png", dpi=300)
    plt.savefig(FIGURES / "fig_a1_pmc_timelines.pdf")
    print(f"Saved: {FIGURES / 'fig_a1_pmc_timelines.png'}")
    print(f"       {FIGURES / 'fig_a1_pmc_timelines.pdf'}")


if __name__ == "__main__":
    from matplotlib.lines import Line2D  # noqa
    main()
