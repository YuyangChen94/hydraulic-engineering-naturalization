"""Evaluate domain-weight and data/reference sensitivities."""
from __future__ import annotations
import importlib
import logging
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

heni = importlib.import_module("02_heni_calculation")
CONFIG = heni.CONFIG
DOMAIN_COLUMNS = heni.DOMAIN_COLUMNS
calculate_annual_iha = heni.calculate_annual_iha
score_water_resources = heni.score_water_resources
calculate_sediment = heni.calculate_sediment
ratio_score_high_good = heni.ratio_score_high_good


def weight_grid_sensitivity(domains: pd.DataFrame) -> pd.DataFrame:
    """Enumerate all four-domain weight combinations on a fixed simplex grid.

    This is a robustness analysis only. Equal weights remain the primary HENI;
    no alternative weight is selected based on the desired result direction.
    """
    step = float(CONFIG["weight_grid_step"])
    n = int(round(1.0 / step))
    if not np.isclose(n * step, 1.0):
        raise ValueError("weight_grid_step must divide 1 exactly")
    phases = CONFIG["event_phases"]
    comparisons = [
        (
            "2003_impoundment",
            phases["pre_impoundment"],
            phases["post_impoundment_pre_ecological_operation"],
        ),
        (
            "2011_ecological_operation",
            phases["post_impoundment_pre_ecological_operation"],
            phases["post_ecological_operation"],
        ),
    ]
    rows = []
    cols = list(DOMAIN_COLUMNS.values())
    for event, (pre_start, pre_end), (post_start, post_end) in comparisons:
        pre = (
            domains[domains.year.between(pre_start, pre_end)][cols]
            .mean()
            .to_numpy(float)
        )
        post = (
            domains[domains.year.between(post_start, post_end)][cols]
            .mean()
            .to_numpy(float)
        )
        delta = post - pre
        values = []
        for a in range(n + 1):
            for b in range(n + 1 - a):
                for c in range(n + 1 - a - b):
                    d = n - a - b - c
                    weights = np.array([a, b, c, d], dtype=float) / n
                    values.append(float(weights @ delta))
        arr = np.asarray(values)
        rows.append(
            {
                "event": event,
                "grid_step": step,
                "n_weight_combinations": len(arr),
                "equal_weight_change": float(np.mean(delta)),
                "minimum_change": float(np.min(arr)),
                "q025_change": float(np.quantile(arr, 0.025)),
                "median_change": float(np.median(arr)),
                "q975_change": float(np.quantile(arr, 0.975)),
                "maximum_change": float(np.max(arr)),
                "proportion_positive": float(np.mean(arr > 0)),
                "proportion_negative": float(np.mean(arr < 0)),
                "interpretation": "direction_robust_across_grid"
                if np.all(arr < 0) or np.all(arr > 0)
                else "direction_depends_on_domain_weights",
            }
        )
    return pd.DataFrame(rows)


def sensitivity_analysis(
    data: Dict[str, pd.DataFrame],
    primary_domains: pd.DataFrame,
    sediment_primary: pd.DataFrame,
    water_environment_sensitivity: pd.DataFrame,
    aquatic_ecology_alt: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    alt_iha, _ = calculate_annual_iha(data["flow"], version="alternative_yichang_flow")
    alt_water = score_water_resources(
        alt_iha, tuple(CONFIG["water_resources_sediment_reference_years"])
    )
    alt_sed, _, _ = calculate_sediment(
        data["flow"],
        data["sediment"],
        flow_version="alternative_yichang_flow",
        ref_years=tuple(CONFIG["water_resources_sediment_reference_years"]),
    )
    alt = (
        primary_domains[["year", "water_environment", "aquatic_ecology"]]
        .merge(alt_water[["year", "water_resources"]], on="year")
        .merge(alt_sed[["year", "sediment"]], on="year")
    )
    alt["system_heni_alt_yichang_flow"] = alt[list(DOMAIN_COLUMNS.values())].mean(
        axis=1
    )
    merged = primary_domains[["year", "system_heni"]].merge(
        alt[["year", "system_heni_alt_yichang_flow"]], on="year"
    )
    rows.append(
        {
            "sensitivity": "alternative_Yichang_discharge",
            "n_years": len(merged),
            "mean_absolute_difference": float(
                np.mean(
                    np.abs(merged.system_heni - merged.system_heni_alt_yichang_flow)
                )
            ),
            "max_absolute_difference": float(
                np.max(np.abs(merged.system_heni - merged.system_heni_alt_yichang_flow))
            ),
            "correlation": float(
                np.corrcoef(merged.system_heni, merged.system_heni_alt_yichang_flow)[
                    0, 1
                ]
            ),
        }
    )
    alt_fish = aquatic_ecology_alt[["year", "aquatic_ecology_alternative"]].dropna()
    m = primary_domains.merge(alt_fish, on="year", how="inner")
    m["system_heni_alt_fish"] = (
        m.water_resources
        + m.sediment
        + m.water_environment
        + m.aquatic_ecology_alternative
    ) / 4
    rows.append(
        {
            "sensitivity": "alternative_Jianli_fish_series_without_2007_imputation",
            "n_years": len(m),
            "mean_absolute_difference": float(
                np.mean(np.abs(m.system_heni - m.system_heni_alt_fish))
            ),
            "max_absolute_difference": float(
                np.max(np.abs(m.system_heni - m.system_heni_alt_fish))
            ),
            "correlation": float(
                np.corrcoef(m.system_heni, m.system_heni_alt_fish)[0, 1]
            ),
        }
    )
    m2 = primary_domains.merge(
        water_environment_sensitivity[
            ["year", "water_environment_three_variable_sensitivity"]
        ],
        on="year",
        how="inner",
    ).dropna()
    m2["system_heni_wq3"] = (
        m2.water_resources
        + m2.sediment
        + m2.water_environment_three_variable_sensitivity
        + m2.aquatic_ecology
    ) / 4
    rows.append(
        {
            "sensitivity": "fixed_TN_TP_DO_composition_complete_years_only",
            "n_years": len(m2),
            "mean_absolute_difference": float(
                np.mean(np.abs(m2.system_heni - m2.system_heni_wq3))
            )
            if len(m2)
            else np.nan,
            "max_absolute_difference": float(
                np.max(np.abs(m2.system_heni - m2.system_heni_wq3))
            )
            if len(m2)
            else np.nan,
            "correlation": float(np.corrcoef(m2.system_heni, m2.system_heni_wq3)[0, 1])
            if len(m2) > 2
            else np.nan,
        }
    )
    eco = data["aquatic_ecology"][
        (data["aquatic_ecology"].data_version == "primary_ecology")
        & (data["aquatic_ecology"].station == "Jianli")
    ].copy()
    hist_start, hist_end = CONFIG["aquatic_ecology_historical_reference_years"]
    hist_ref = eco[eco.year.between(hist_start, hist_end)].value.dropna()
    hist_target = float(hist_ref.median())
    eco["aquatic_ecology_historical_target_score"] = eco.value.apply(
        lambda x: ratio_score_high_good(x, hist_target)
    )
    m3 = primary_domains.merge(
        eco[["year", "aquatic_ecology_historical_target_score"]], on="year", how="inner"
    )
    m3["system_heni_aquatic_ecology_historical_target"] = (
        m3.water_resources
        + m3.sediment
        + m3.water_environment
        + m3.aquatic_ecology_historical_target_score
    ) / 4
    rows.append(
        {
            "sensitivity": "sparse_historical_aquatic_ecology_reference_1960_1994_four_years",
            "n_years": len(m3),
            "mean_absolute_difference": float(
                np.mean(
                    np.abs(
                        m3.system_heni
                        - m3.system_heni_aquatic_ecology_historical_target
                    )
                )
            ),
            "max_absolute_difference": float(
                np.max(
                    np.abs(
                        m3.system_heni
                        - m3.system_heni_aquatic_ecology_historical_target
                    )
                )
            ),
            "correlation": float(
                np.corrcoef(
                    m3.system_heni, m3.system_heni_aquatic_ecology_historical_target
                )[0, 1]
            ),
            "reference_value": hist_target,
            "reference_n_years": int(len(hist_ref)),
            "event_2003_change": float(
                m3[
                    m3.year.between(2003, 2010)
                ].system_heni_aquatic_ecology_historical_target.mean()
                - m3[
                    m3.year.between(1998, 2002)
                ].system_heni_aquatic_ecology_historical_target.mean()
            ),
            "event_2011_change": float(
                m3[
                    m3.year.between(2011, 2017)
                ].system_heni_aquatic_ecology_historical_target.mean()
                - m3[
                    m3.year.between(2003, 2010)
                ].system_heni_aquatic_ecology_historical_target.mean()
            ),
        }
    )
    return pd.DataFrame(rows)


def main():
    parser = heni.cli(
        "Evaluate 1,771 simplex weights and four predefined data/reference alternatives."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Authorized input directory; required unless --weights-only",
    )
    parser.add_argument("--weights-only", action="store_true")
    args = parser.parse_args()
    if not args.weights_only and args.input_dir is None:
        parser.error("--input-dir is required for data/reference sensitivity")
    out = heni.runtime(args, "05_sensitivity_analysis")
    domains = pd.read_csv(out / "11_source_Fig3_system_and_domains.csv")
    weights = weight_grid_sensitivity(domains)
    heni.save(weights, out, "19_source_weight_grid_sensitivity.csv")
    records = []
    cols = list(DOMAIN_COLUMNS.values())
    for event, pre, post in [
        ("2003_impoundment", (1998, 2002), (2003, 2010)),
        ("2011_ecological_operation", (2003, 2010), (2011, 2017)),
    ]:
        delta = domains.loc[domains.year.between(*post), cols].mean().to_numpy(
            float
        ) - domains.loc[domains.year.between(*pre), cols].mean().to_numpy(float)
        for a in range(21):
            for b in range(21 - a):
                for c in range(21 - a - b):
                    w = np.array([a, b, c, 20 - a - b - c], dtype=float) / 20
                    records.append(
                        dict(
                            event=event,
                            **dict(zip(cols, w)),
                            system_change=float(w @ delta),
                        )
                    )
    heni.save(pd.DataFrame(records), out, "20_weight_grid_values.csv")
    if not args.weights_only:
        data = heni.read_inputs(args.input_dir)
        summary = sensitivity_analysis(
            data,
            domains,
            pd.read_csv(out / "04_source_sediment_metrics_scores.csv"),
            pd.read_csv(out / "08_source_water_environment_sensitivity.csv"),
            pd.read_csv(out / "10_source_ecology_alternative.csv"),
        )
        heni.save(summary, out, "15_source_sensitivity_summary.csv")
    logging.info("Completed 1771 combinations per event")


if __name__ == "__main__":
    main()
