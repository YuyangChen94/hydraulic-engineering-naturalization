"""Render the analytical panels of Figs. 1, 3 and 4 from regenerated tables."""
from __future__ import annotations
import importlib
import logging
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

heni = importlib.import_module("02_heni_calculation")
LABELS = {
    "water_resources": "Water resources",
    "sediment": "Sediment",
    "water_environment": "Water environment",
    "aquatic_ecology": "Aquatic ecology",
    "system_heni": "System HENI",
    "inflow_zone": "Inflow zone",
    "reservoir_zone": "Reservoir zone",
    "below_dam_zone": "Below-dam zone",
    "middle_reach_zone": "Middle-reach zone",
}
COLORS = {
    "water_resources": "#4C78A8",
    "sediment": "#B2794F",
    "water_environment": "#5B9A8B",
    "aquatic_ecology": "#C65D5D",
    "system_heni": "#222222",
    "inflow_zone": "#4C78A8",
    "reservoir_zone": "#9BBCE3",
    "below_dam_zone": "#5B9A8B",
    "middle_reach_zone": "#A9CFC2",
}


def export(fig, out, name):
    fig.savefig(out / (name + ".png"), dpi=300)
    fig.savefig(out / (name + ".pdf"))
    plt.close(fig)
    logging.info("Rendered %s", name)


def axis(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7.09, 3.65), layout="constrained")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    return (fig, ax)


def trajectory(data, out, name, title, ylabel):
    fig, ax = axis(title, "Year", ylabel)
    for col in data.columns:
        if col == "year":
            continue
        ax.plot(
            data.year,
            data[col],
            "o-",
            color=COLORS[col],
            lw=1.2,
            ms=3,
            label=LABELS[col],
        )
    for year in [2003, 2011]:
        ax.axvline(year, color="#AAAAAA", ls=":", lw=0.7, zorder=0)
    ax.set_xticks([1998, 2003, 2011, 2017])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    export(fig, out, name)


def fig1(args, out):
    data = pd.read_csv(out / "01_ecdf.csv")
    fig, ax = axis(
        "Fig. 1b | Reservoir proximity (n = 4,262)",
        "Distance to nearest reach (km)",
        "Cumulative reservoirs (%)",
    )
    for label, color, name in [
        ("high_connectivity", "#6F9FB3", "High-connectivity reaches"),
        ("flow_restricted", "#C66E62", "Flow-restricted reaches"),
    ]:
        d = data.loc[data.river_class.eq(label)]
        ax.plot(
            d.distance_km, d.cumulative_reservoirs_pct, color=color, label=name, lw=1.4
        )
    ax.set(xlim=(0, 1000), ylim=(0, 100))
    ax.legend(frameon=False)
    export(fig, out, "Fig1b")
    if args.reservoirs is None or args.rivers is None:
        raise ValueError("Fig. 1a requires --reservoirs and --rivers; see README.")
    geo = importlib.import_module("01_global_reservoir_river_analysis")
    reservoirs, rivers = geo.load_public_layers(
        args.reservoirs, args.rivers, args.river_layer
    )
    fig, ax = plt.subplots(figsize=(7.09, 3.4), layout="constrained")
    for label, color in [
        ("high_connectivity", "#6F9FB3"),
        ("flow_restricted", "#C66E62"),
    ]:
        rivers.loc[rivers.status_binary.eq(label)].to_crs(geo.DISPLAY_CRS).plot(
            ax=ax, color=color, linewidth=0.25
        )
    reservoirs.to_crs(geo.DISPLAY_CRS).plot(
        ax=ax, color="#66767F", markersize=1, alpha=0.6
    )
    ax.legend(
        handles=[
            Line2D([], [], color="#6F9FB3", label="High-connectivity reaches"),
            Line2D([], [], color="#C66E62", label="Flow-restricted reaches"),
            Line2D(
                [],
                [],
                color="#66767F",
                marker="o",
                linestyle="none",
                markersize=3,
                label="Reservoirs",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    ax.set_axis_off()
    ax.set_title("Fig. 1a | Reservoirs and long-river connectivity classes")
    export(fig, out, "Fig1a")


def fig3(out):
    temporal = importlib.import_module("03_temporal_diagnosis")
    data = pd.read_csv(out / "11_source_Fig3_system_and_domains.csv")
    events = pd.read_csv(out / "17_source_segmented_event_tests.csv")
    data = data.rename(
        columns={
            **{v: k for k, v in heni.DOMAIN_COLUMNS.items()},
            "system_heni": "system_heni",
        }
    )
    for letter, names, title in [
        ("b", ["system_heni"], "System HENI"),
        ("c", list(heni.DOMAIN_COLUMNS), "Four domains"),
    ]:
        fig, ax = axis(
            f"Fig. 3{letter} | {title} (n = 20 years)", "Year", "Score (0–1)"
        )
        for name in names:
            fit = temporal.segmented_fit_values(data, name)["fitted"]
            ax.plot(data.year, fit, color=COLORS[name], alpha=0.25, lw=5, zorder=1)
            ax.plot(
                data.year,
                data[name],
                "o-",
                color=COLORS[name],
                lw=1,
                ms=3,
                label=LABELS[name],
                zorder=2,
            )
        for year in [2003, 2011]:
            ax.axvline(year, color="#AAAAAA", ls=":", lw=0.7)
        ax.set_xticks([1998, 2003, 2011, 2017])
        ax.legend(
            frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2
        )
        if letter == "b":
            e = events.loc[events.object.eq("system_heni")]
            ax.text(
                0.98,
                0.97,
                "; ".join(
                    (f"{str(r.event)[:4]}: P = {r.joint_p:.3g}" for r in e.itertuples())
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
        export(fig, out, "Fig3" + letter)
    contributions = pd.read_csv(out / "14_source_event_contributions.csv")
    fig, ax = axis(
        "Fig. 3d | Fixed-weight contributions",
        "Process domain",
        "Contribution to HENI stage-mean change",
    )
    names = list(heni.DOMAIN_COLUMNS)
    for i, (event, label) in enumerate(
        [
            ("2003_impoundment", "2003–2010 minus 1998–2002"),
            ("2011_ecological_operation", "2011–2017 minus 2003–2010"),
        ]
    ):
        d = (
            contributions.loc[contributions.event.eq(event)]
            .set_index("domain")
            .loc[names]
        )
        x = np.arange(4) + (i - 0.5) * 0.36
        ax.bar(
            x,
            d.contribution_value,
            width=0.34,
            color=[COLORS[n] for n in names],
            hatch="///" if i else None,
            label=label,
        )
        for xi, row in zip(x, d.itertuples()):
            ax.annotate(
                f"{100 * row.absolute_contribution_share:.1f}%",
                (xi, row.contribution_value),
                xytext=(0, 4 if row.contribution_value >= 0 else -10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(range(4), [LABELS[n] for n in names])
    ax.axhline(0, color="#AAAAAA", lw=0.7)
    ax.margins(y=0.2)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    export(fig, out, "Fig3d")


def fig4(out):
    trajectory(
        pd.read_csv(out / "40_longitudinal_centered.csv"),
        out,
        "Fig4a",
        "Fig. 4a | Longitudinal responses (1998–2017)",
        "Change from own 1998–2002 mean",
    )
    for filename, letter, title in [
        ("35_Fig.4b_process_response_summary.csv", "b", "Process domains"),
        ("36_Fig.4c_spatial_response_summary.csv", "c", "Longitudinal zones"),
    ]:
        data = pd.read_csv(out / filename)
        fig, ax = axis(
            f"Fig. 4{letter} | {title}",
            "Observed trough lag from 2003 (years)",
            "Post-2011 trend (score/year)",
        )
        for row in data.itertuples():
            x, y = (row.touchdown_lag_year, row.recovery_slope_per_year)
            ax.errorbar(
                x,
                y,
                yerr=[
                    [y - row.recovery_slope_ci_low],
                    [row.recovery_slope_ci_high - y],
                ],
                fmt="D" if row.object == "system_heni" else "o",
                color=COLORS[row.object],
                capsize=3,
                label=LABELS[row.object],
            )
            marker = (
                "*"
                if row.recovery_slope_p < 0.05
                else "†"
                if row.recovery_slope_p < 0.1
                else ""
            )
            if marker:
                ax.annotate(marker, (x, y), xytext=(5, 4), textcoords="offset points")
        ax.axhline(0, color="#AAAAAA", lw=0.7)
        ax.set_xticks(range(0, 8))
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False
        )
        export(fig, out, "Fig4" + letter)


def main():
    parser = heni.cli(
        "Render analysis panels; publication artwork and conceptual illustrations are separate."
    )
    parser.add_argument("--figure", choices=["1", "3", "4", "all"], required=True)
    parser.add_argument("--reservoirs", type=Path)
    parser.add_argument("--rivers", type=Path)
    parser.add_argument("--river-layer")
    args = parser.parse_args()
    out = heni.runtime(args, "06_figures")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )
    if args.figure in ["1", "all"]:
        fig1(args, out)
    if args.figure in ["3", "all"]:
        fig3(out)
    if args.figure in ["4", "all"]:
        fig4(out)


if __name__ == "__main__":
    main()
