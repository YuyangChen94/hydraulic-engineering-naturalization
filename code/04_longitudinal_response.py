"""Calculate longitudinal composites, trough timing and post-2011 trends."""
from __future__ import annotations
import importlib
import logging
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon
from statsmodels.nonparametric.smoothers_lowess import lowess

heni = importlib.import_module("02_heni_calculation")
CONFIG = {
    "analysis_years": [1998, 2017],
    "baseline_years": [1998, 2002],
    "post_impoundment_min_window": [2003, 2010],
    "post_ecological_operation_window": [2011, 2017],
    "historical_reference_years": [1960, 1994],
    "touchdown_bootstrap_iterations": 5000,
    "slope_difference_bootstrap_iterations": 5000,
    "random_seed": 20260731,
    "early_late_reference_lag": 4.0,
}


def days_in_year(year: int) -> int:
    return int(pd.Timestamp(f"{year}-12-31").dayofyear)


def calculate_station_water_scores(heni, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    original_station = heni.CONFIG["water_resources_station"]
    for station in ["Cuntan", "Wulong", "Yichang"]:
        heni.CONFIG["water_resources_station"] = station
        iha, _ = heni.calculate_annual_iha(
            data["flow"],
            version="primary_flow",
            ref_years=tuple(CONFIG["historical_reference_years"]),
        )
        score = heni.score_water_resources(
            iha, tuple(CONFIG["historical_reference_years"])
        )[["year", "water_resources"]].rename(
            columns={"water_resources": f"{station}_water_resources"}
        )
        merged = (
            score if merged is None else merged.merge(score, on="year", how="outer")
        )
    heni.CONFIG["water_resources_station"] = original_station
    if merged is None:
        raise RuntimeError("Station water-score calculation failed")
    return merged


def calculate_station_sediment_scores(
    heni, data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Local sediment magnitude-and-seasonality similarity for spatial analysis.

    This is used only in Fig.4c. The system HENI sediment domain remains the
    suspended-sediment continuity proxy defined in the main analysis.
    """
    flow = data["flow"].copy()
    sediment = data["sediment"].copy()
    merged: pd.DataFrame | None = None
    for station in ["Cuntan", "Wulong", "Yichang"]:
        q = flow[
            (flow.station == station) & (flow.data_version == "primary_flow")
        ][["date", "year", "month", "value"]].copy()
        q["date"] = pd.to_datetime(q["date"], errors="coerce")
        q["q"] = pd.to_numeric(q["value"], errors="coerce")
        q.loc[q.q <= 0, "q"] = np.nan
        s = sediment[
            (sediment.station == station)
            & (sediment.data_version == "primary_sediment")
        ][["date", "year", "month", "value"]].copy()
        s["date"] = pd.to_datetime(s["date"], errors="coerce")
        s["ssc"] = pd.to_numeric(s["value"], errors="coerce")
        s.loc[s.ssc <= 0, "ssc"] = np.nan
        d = (
            q[["date", "year", "month", "q"]]
            .merge(
                s[["date", "year", "month", "ssc"]],
                on=["date", "year", "month"],
                how="inner",
            )
            .sort_values("date")
        )
        d["q"] = d["q"].interpolate(limit=2, limit_area="inside")
        d["ssc"] = d["ssc"].interpolate(limit=3, limit_area="inside")
        d["load_t_day"] = d.q * d.ssc * 86.4
        annual_rows: List[dict] = []
        monthly_rows: List[dict] = []
        for year, g in d.groupby("year"):
            year = int(year)
            expected = days_in_year(year)
            valid = int(g.load_t_day.notna().sum())
            coverage = valid / expected
            annual_load = np.nan
            if coverage >= 0.95:
                annual_load = float(g.load_t_day.sum() * expected / valid)
            annual_rows.append({"year": year, "annual_load_t": annual_load})
            for month, gm in g.groupby("month"):
                valid_m = int(gm.load_t_day.notna().sum())
                expected_m = len(gm)
                coverage_m = valid_m / expected_m if expected_m else 0.0
                monthly_load = np.nan
                if coverage_m >= 0.9:
                    monthly_load = float(gm.load_t_day.sum() * expected_m / valid_m)
                monthly_rows.append(
                    {"year": year, "month": int(month), "monthly_load_t": monthly_load}
                )
        annual = pd.DataFrame(annual_rows)
        monthly = pd.DataFrame(monthly_rows).pivot(
            index="year", columns="month", values="monthly_load_t"
        )
        fractions = monthly.div(monthly.sum(axis=1), axis=0)
        ref_start, ref_end = CONFIG["historical_reference_years"]
        reference_annual = annual[annual.year.between(ref_start, ref_end)]
        magnitude_target = float(reference_annual.annual_load_t.median())
        ref_years = [
            y
            for y in fractions.index
            if ref_start <= y <= ref_end and fractions.loc[y].notna().sum() == 12
        ]
        reference_fraction = fractions.loc[ref_years].median(axis=0).to_numpy(float)
        rows = []
        for _, r in annual.iterrows():
            year = int(r.year)
            magnitude_score = heni.ratio_similarity_two_sided(
                r.annual_load_t, magnitude_target
            )
            timing_score = np.nan
            if year in fractions.index and fractions.loc[year].notna().sum() == 12:
                arr = fractions.loc[year].to_numpy(float)
                if np.isfinite(arr).all() and arr.sum() > 0:
                    timing_score = float(
                        1
                        - jensenshannon(
                            arr / arr.sum(),
                            reference_fraction / reference_fraction.sum(),
                            base=2,
                        )
                    )
            valid_scores = [
                v for v in [magnitude_score, timing_score] if np.isfinite(v)
            ]
            local_score = float(np.mean(valid_scores)) if valid_scores else np.nan
            rows.append(
                {
                    "year": year,
                    f"{station}_sediment": local_score,
                    f"{station}_sediment_magnitude": magnitude_score,
                    f"{station}_sediment_seasonality": timing_score,
                }
            )
        score = pd.DataFrame(rows)
        merged = (
            score if merged is None else merged.merge(score, on="year", how="outer")
        )
    if merged is None:
        raise RuntimeError("Station sediment-score calculation failed")
    return merged


def build_process_series() -> pd.DataFrame:
    path = OUT / "11_source_Fig3_system_and_domains.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run code/02_heni_calculation.py, then code/03_temporal_diagnosis.py, before code/04_longitudinal_response.py; see README.md."
        )
    d = pd.read_csv(path)
    d = d.rename(
        columns={
            "water_resources": "water_resources",
            "sediment": "sediment",
            "water_environment": "water_environment",
            "aquatic_ecology": "aquatic_ecology",
            "system_heni": "system_heni",
        }
    )
    return d[
        [
            "year",
            "water_resources",
            "sediment",
            "water_environment",
            "aquatic_ecology",
            "system_heni",
        ]
    ]


def build_spatial_series(
    heni, data: Dict[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    station_water = calculate_station_water_scores(heni, data)
    station_sediment = calculate_station_sediment_scores(heni, data)
    water_environment_path = OUT / "07_source_water_environment_station_scores.csv"
    eco_path = OUT / "09_source_ecology_primary.csv"
    system_path = OUT / "11_source_Fig3_system_and_domains.csv"
    water_environment = (
        pd.read_csv(water_environment_path)
        .pivot(index="year", columns="station", values="TP_score")
        .reset_index()
    )
    aquatic_ecology = pd.read_csv(eco_path)[["year", "aquatic_ecology"]].rename(
        columns={"aquatic_ecology": "Jianli_aquatic_ecology"}
    )
    system = pd.read_csv(system_path)[["year", "system_heni"]].rename(
        columns={"system_heni": "system_heni"}
    )
    d = (
        system.merge(station_water, on="year", how="left")
        .merge(station_sediment, on="year", how="left")
        .merge(water_environment, on="year", how="left")
        .merge(aquatic_ecology, on="year", how="left")
    )
    start, end = CONFIG["analysis_years"]
    d = d[d.year.between(start, end)].copy()
    composition = [
        {
            "zone": "inflow_zone",
            "components": "Cuntan water resources, Wulong water resources, Cuntan sediment, Wulong sediment, Cuntan TP",
            "n_indicators": 5,
            "aggregation": "Equal mean of available standardized indicators",
        },
        {
            "zone": "reservoir_zone",
            "components": "Qingxichang TP, Tuokou TP, Guandukou TP",
            "n_indicators": 3,
            "aggregation": "Equal mean of the three reservoir TP stations",
        },
        {
            "zone": "below_dam_zone",
            "components": "Yichang water resources, Yichang sediment",
            "n_indicators": 2,
            "aggregation": "Equal mean of two standardized indicators",
        },
        {
            "zone": "middle_reach_zone",
            "components": "Hankou TP, Jianli aquatic ecology",
            "n_indicators": 2,
            "aggregation": "Equal mean of two standardized indicators",
        },
    ]
    composition_df = pd.DataFrame(composition)
    d["inflow_zone"] = d[
        [
            "Cuntan_water_resources",
            "Wulong_water_resources",
            "Cuntan_sediment",
            "Wulong_sediment",
            "Cuntan",
        ]
    ].mean(axis=1, skipna=False)
    d["reservoir_zone"] = d[["Qingxichang", "Tuokou", "Guandukou"]].mean(
        axis=1, skipna=False
    )
    d["below_dam_zone"] = d[["Yichang_water_resources", "Yichang_sediment"]].mean(
        axis=1, skipna=False
    )
    d["middle_reach_zone"] = d[["Hankou", "Jianli_aquatic_ecology"]].mean(
        axis=1, skipna=False
    )
    required = [
        "inflow_zone",
        "reservoir_zone",
        "below_dam_zone",
        "middle_reach_zone",
        "system_heni",
    ]
    if d[required].isna().any().any():
        missing = d.loc[d[required].isna().any(axis=1), ["year"] + required]
        raise ValueError(f"Spatial series has missing values:\n{missing}")
    return (d[["year"] + required], composition_df)


def residual_bootstrap_touchdown_lag(
    years: np.ndarray, values: np.ndarray, rng: np.random.Generator, n_boot: int
) -> Tuple[pd.DataFrame, dict]:
    """LOWESS residual bootstrap for minimum-year stability assessment.

    The observed minimum year is retained as the primary estimate. The
    bootstrap is used only to quantify how stable that discrete minimum is;
    its broad interval is not plotted because only eight annual observations
    are available in 2003-2010.
    """
    fit = lowess(values, years, frac=0.75, it=0, return_sorted=False)
    residuals = values - fit
    sampled_idx = rng.integers(0, len(residuals), size=(n_boot, len(residuals)))
    y_boot = fit[None, :] + residuals[sampled_idx]
    min_idx = np.nanargmin(y_boot, axis=1)
    dist = pd.DataFrame(
        {
            "iteration": np.arange(1, n_boot + 1),
            "minimum_year": years[min_idx].astype(int),
            "lag_year": (years[min_idx] - 2003).astype(int),
        }
    )
    observed_idx = int(np.nanargmin(values))
    observed_lag = int(years[observed_idx] - 2003)
    summary = {
        "touchdown_year": int(years[observed_idx]),
        "touchdown_lag_year": observed_lag,
        "bootstrap_modal_lag": int(dist.lag_year.mode().iloc[0]),
        "bootstrap_median_lag": float(dist.lag_year.median()),
        "bootstrap_q25_lag": float(dist.lag_year.quantile(0.25)),
        "bootstrap_q75_lag": float(dist.lag_year.quantile(0.75)),
        "bootstrap_q025_lag": float(dist.lag_year.quantile(0.025)),
        "bootstrap_q975_lag": float(dist.lag_year.quantile(0.975)),
        "observed_lag_bootstrap_probability": float(
            (dist.lag_year == observed_lag).mean()
        ),
    }
    return (dist, summary)


def analyze_response_set(
    series: pd.DataFrame, object_names: Sequence[str], group: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(
        CONFIG["random_seed"] + (0 if group == "process_domains" else 1000)
    )
    baseline_start, baseline_end = CONFIG["baseline_years"]
    min_start, min_end = CONFIG["post_impoundment_min_window"]
    rec_start, rec_end = CONFIG["post_ecological_operation_window"]
    summary_rows = []
    bootstrap_rows = []
    for name in object_names:
        baseline = series[series.year.between(baseline_start, baseline_end)][name]
        post = series[series.year.between(min_start, min_end)][["year", name]].dropna()
        recovery = series[series.year.between(rec_start, rec_end)][
            ["year", name]
        ].dropna()
        if len(post) != min_end - min_start + 1:
            raise ValueError(f"{name}: incomplete post-2003 annual series")
        if len(recovery) != rec_end - rec_start + 1:
            raise ValueError(f"{name}: incomplete post-2011 annual series")
        min_idx = post[name].idxmin()
        touchdown_year = int(post.loc[min_idx, "year"])
        touchdown_lag = touchdown_year - 2003
        maximum_decline = float(post.loc[min_idx, name] - baseline.mean())
        x = recovery.year.to_numpy(float) - rec_start
        X = sm.add_constant(x)
        model = sm.OLS(recovery[name].to_numpy(float), X).fit(
            cov_type="HC3", use_t=True
        )
        ci = model.conf_int(alpha=0.05)[1]
        dist, touch = residual_bootstrap_touchdown_lag(
            post.year.to_numpy(int),
            post[name].to_numpy(float),
            rng,
            int(CONFIG["touchdown_bootstrap_iterations"]),
        )
        dist.insert(0, "object", name)
        dist.insert(0, "group", group)
        bootstrap_rows.append(dist)
        summary_rows.append(
            {
                "group": group,
                "object": name,
                "baseline_mean_1998_2002": float(baseline.mean()),
                "touchdown_year": touchdown_year,
                "touchdown_lag_year": touchdown_lag,
                "maximum_decline_from_baseline": maximum_decline,
                "recovery_slope_per_year": float(model.params[1]),
                "recovery_slope_robust_se": float(model.bse[1]),
                "recovery_slope_p": float(model.pvalues[1]),
                "recovery_slope_ci_low": float(ci[0]),
                "recovery_slope_ci_high": float(ci[1]),
                "recovery_model": "OLS slope for 2011-2017 with HC3 covariance",
                **touch,
            }
        )
    return (pd.DataFrame(summary_rows), pd.concat(bootstrap_rows, ignore_index=True))


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def pairwise_slope_differences(
    series: pd.DataFrame, object_names: Sequence[str], group: str
) -> pd.DataFrame:
    rng = np.random.default_rng(
        CONFIG["random_seed"] + (3000 if group == "process_domains" else 4000)
    )
    rec_start, rec_end = CONFIG["post_ecological_operation_window"]
    d = series[series.year.between(rec_start, rec_end)][
        ["year"] + list(object_names)
    ].dropna()
    years = d.year.to_numpy(float)
    value_matrix = d[list(object_names)].to_numpy(float)
    n_boot = int(CONFIG["slope_difference_bootstrap_iterations"])
    observed_slopes = {}
    for j, name in enumerate(object_names):
        X = sm.add_constant(years - rec_start)
        observed_slopes[name] = float(sm.OLS(value_matrix[:, j], X).fit().params[1])
    idx_matrix = rng.integers(0, len(d), size=(n_boot * 2, len(d)))
    x_boot = years[idx_matrix] - rec_start
    x_centered = x_boot - x_boot.mean(axis=1, keepdims=True)
    denom = np.sum(x_centered**2, axis=1)
    valid = denom > 0
    idx_matrix = idx_matrix[valid][:n_boot]
    x_centered = x_centered[valid][:n_boot]
    denom = denom[valid][:n_boot]
    if len(denom) < n_boot:
        raise RuntimeError("Insufficient non-degenerate paired bootstrap samples")
    boot_slopes = {}
    for j, name in enumerate(object_names):
        y_boot = value_matrix[idx_matrix, j]
        y_centered = y_boot - y_boot.mean(axis=1, keepdims=True)
        boot_slopes[name] = np.sum(x_centered * y_centered, axis=1) / denom
    rows = []
    names_without_system = [x for x in object_names if x != "system_heni"]
    for i in range(len(names_without_system)):
        for j in range(i + 1, len(names_without_system)):
            a, b = (names_without_system[i], names_without_system[j])
            diff = np.asarray(boot_slopes[a]) - np.asarray(boot_slopes[b])
            p = min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
            rows.append(
                {
                    "group": group,
                    "object_a": a,
                    "object_b": b,
                    "observed_slope_difference_a_minus_b": observed_slopes[a]
                    - observed_slopes[b],
                    "bootstrap_ci_low": float(np.quantile(diff, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(diff, 0.975)),
                    "bootstrap_p_two_sided": float(p),
                    "n_bootstrap": len(diff),
                }
            )
    result = pd.DataFrame(rows)
    if len(result):
        result["bootstrap_p_BH"] = bh_adjust(result.bootstrap_p_two_sided)
    return result


def main():
    global OUT
    args = heni.cli(
        "Four longitudinal-zone composites, observed troughs and bootstrap trend comparisons.",
        True,
    ).parse_args()
    OUT = heni.runtime(args, "04_longitudinal_response")
    data = heni.read_inputs(args.input_dir)
    process = build_process_series()
    spatial, composition = build_spatial_series(heni, data)
    process_names = [
        "water_resources",
        "sediment",
        "water_environment",
        "aquatic_ecology",
        "system_heni",
    ]
    spatial_names = [
        "inflow_zone",
        "reservoir_zone",
        "below_dam_zone",
        "middle_reach_zone",
        "system_heni",
    ]
    ps, pb = analyze_response_set(process, process_names, "process_domains")
    ss, sb = analyze_response_set(spatial, spatial_names, "longitudinal_zones")
    pp = pairwise_slope_differences(process, process_names, "process_domains")
    sp = pairwise_slope_differences(spatial, spatial_names, "longitudinal_zones")
    centered = spatial.copy()
    centered[spatial_names] = (
        spatial[spatial_names]
        - spatial.loc[spatial.year.between(1998, 2002), spatial_names].mean()
    )
    for table, name in [
        (process, "33_Fig.4b_process_annual_series.csv"),
        (spatial, "34_Fig.4c_spatial_annual_series.csv"),
        (ps, "35_Fig.4b_process_response_summary.csv"),
        (ss, "36_Fig.4c_spatial_response_summary.csv"),
        (composition, "37_Fig.4c_spatial_composition.csv"),
        (
            pd.concat([pp, sp], ignore_index=True),
            "38_QA_pairwise_recovery_slope_differences.csv",
        ),
        (
            pd.concat([pb, sb], ignore_index=True),
            "39_QA_touchdown_lag_bootstrap_distribution.csv",
        ),
        (centered, "40_longitudinal_centered.csv"),
    ]:
        heni.save(table, OUT, name)
    logging.info("Completed four zones; 5000 replicates per response analysis")


if __name__ == "__main__":
    main()
