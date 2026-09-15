"""Calculate annual HENI from four fixed process domains."""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pandas as pd
import math
from scipy.spatial.distance import jensenshannon
from scipy.stats import theilslopes

CONFIG = {
    "analysis_years": [1998, 2017],
    "water_resources_sediment_reference_years": [1960, 1994],
    "water_quality_reference_years": [1998, 2002],
    "aquatic_ecology_reference_years": [1997, 2002],
    "aquatic_ecology_historical_reference_years": [1960, 1994],
    "event_phases": {
        "pre_impoundment": [1998, 2002],
        "post_impoundment_pre_ecological_operation": [2003, 2010],
        "post_ecological_operation": [2011, 2017],
    },
    "events": {"2003_impoundment": 2003, "2011_ecological_operation": 2011},
    "domain_weights": {
        "water_resources": 0.25,
        "sediment": 0.25,
        "water_environment": 0.25,
        "aquatic_ecology": 0.25,
    },
    "water_quality_primary_variable": "TP",
    "water_quality_fixed_stations": [
        "Cuntan",
        "Qingxichang",
        "Tuokou",
        "Guandukou",
        "Hankou",
    ],
    "sediment_input_stations": ["Cuntan", "Wulong"],
    "sediment_output_station": "Yichang",
    "water_resources_station": "Yichang",
    "min_daily_coverage": 0.95,
    "min_monthly_coverage": 0.9,
    "min_wq_months_per_station_year": 8,
    "min_wq_stations_per_year": 4,
    "random_seed": 20260731,
    "net_change_share_threshold": 0.02,
    "segmented_covariance": "HC3",
    "weight_grid_step": 0.05,
}
DOMAIN_COLUMNS = {
    "water_resources": "water_resources",
    "sediment": "sediment",
    "water_environment": "water_environment",
    "aquatic_ecology": "aquatic_ecology",
}


def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def days_in_year(year: int) -> int:
    return int(pd.Timestamp(f"{year}-12-31").dayofyear)


def ratio_similarity_two_sided(
    value: float, target: float, offset: float = 0.0
) -> float:
    """Scale-free similarity: exp(-|ln((x+c)/(target+c))|).

    Equivalent to min((x+c)/(target+c), (target+c)/(x+c)).
    It is continuous, bounded [0,1], has no tuned shape parameter, and does not
    truncate severe departures to zero.
    """
    if not np.isfinite(value) or not np.isfinite(target):
        return np.nan
    a, b = (value + offset, target + offset)
    if a <= 0 or b <= 0:
        return np.nan
    return float(np.exp(-abs(np.log(a / b))))


def ratio_score_high_good(value: float, target: float, offset: float = 0.0) -> float:
    if not np.isfinite(value) or not np.isfinite(target):
        return np.nan
    a, b = (value + offset, target + offset)
    if a < 0 or b <= 0:
        return np.nan
    return float(min(a / b, 1.0))


def ratio_score_low_good(value: float, target: float, offset: float = 0.0) -> float:
    if not np.isfinite(value) or not np.isfinite(target):
        return np.nan
    a, b = (value + offset, target + offset)
    if a <= 0 or b < 0:
        return np.nan
    return float(min(b / a, 1.0))


def circular_center(values: Iterable[float], period: float = 365.25) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    angles = 2 * np.pi * (values - 1) / period
    mean_angle = math.atan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    return float(1 + mean_angle * period / (2 * np.pi))


def circular_distance(value: float, center: float, period: float = 365.25) -> float:
    d1 = (value - center) % period
    d2 = (center - value) % period
    return float(min(d1, d2))


def run_lengths(mask: np.ndarray) -> List[int]:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return []
    padded = np.r_[False, arr, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return (ends - starts).tolist()


def calculate_annual_iha(
    flow: pd.DataFrame,
    version: str = "primary_flow",
    ref_years: Tuple[int, int] = (1960, 1994),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    station = CONFIG["water_resources_station"]
    d = flow[(flow.station == station) & (flow.data_version == version)].copy()
    require_columns(d, ["date", "year", "month", "value"], "flow")
    d = d.sort_values("date")
    d["q"] = d["value"].astype(float)
    invalid = d["q"] <= 0
    invalid_log = d.loc[invalid, ["date", "station", "q"]].copy()
    invalid_log["issue"] = "nonpositive_discharge_set_missing"
    d.loc[invalid, "q"] = np.nan
    d["q"] = d["q"].interpolate(limit=2, limit_area="inside")
    ref = d[d.year.between(*ref_years)]
    low_thr = float(ref.q.quantile(0.25))
    high_thr = float(ref.q.quantile(0.75))
    rows: List[dict] = []
    for year, g in d.groupby("year"):
        year = int(year)
        g = g.sort_values("date")
        n_expected = days_in_year(year)
        coverage = g.q.notna().sum() / n_expected
        if coverage < CONFIG["min_daily_coverage"]:
            continue
        q = g.q.interpolate(limit=3, limit_direction="both").to_numpy(float)
        if np.isfinite(q).mean() < 0.99:
            continue
        s = pd.Series(q)
        months = g.month.to_numpy()
        row = {"year": year, "flow_coverage": coverage}
        for m in range(1, 13):
            row[f"month_{m:02d}"] = float(np.nanmedian(q[months == m]))
        row.update(
            {
                "mean_flow": float(np.mean(q)),
                "min_1d": float(np.min(q)),
                "min_7d": float(s.rolling(7).mean().min()),
                "max_1d": float(np.max(q)),
                "max_7d": float(s.rolling(7).mean().max()),
                "doy_min": int(g.iloc[int(np.argmin(q))].date.dayofyear),
                "doy_max": int(g.iloc[int(np.argmax(q))].date.dayofyear),
            }
        )
        low_runs = run_lengths(q < low_thr)
        high_runs = run_lengths(q > high_thr)
        row.update(
            {
                "low_pulse_count": len(low_runs),
                "low_pulse_duration": float(np.mean(low_runs)) if low_runs else 0.0,
                "high_pulse_count": len(high_runs),
                "high_pulse_duration": float(np.mean(high_runs)) if high_runs else 0.0,
            }
        )
        diff = np.diff(q)
        rise = diff[diff > 0]
        fall = -diff[diff < 0]
        signs = np.sign(diff)
        signs = signs[signs != 0]
        row.update(
            {
                "rise_rate": float(np.median(rise)) if len(rise) else 0.0,
                "fall_rate": float(np.median(fall)) if len(fall) else 0.0,
                "reversals": int(np.sum(signs[1:] != signs[:-1]))
                if len(signs) > 1
                else 0,
            }
        )
        rows.append(row)
    return (pd.DataFrame(rows), invalid_log)


def score_water_resources(
    iha: pd.DataFrame, ref_years: Tuple[int, int]
) -> pd.DataFrame:
    ref = iha[iha.year.between(*ref_years)].copy()
    centers = {k: circular_center(ref[k]) for k in ["doy_min", "doy_max"]}
    rows = []
    for _, r in iha.iterrows():
        monthly = np.mean(
            [
                ratio_similarity_two_sided(
                    r[f"month_{m:02d}"], ref[f"month_{m:02d}"].median()
                )
                for m in range(1, 13)
            ]
        )
        extremes = np.mean(
            [
                ratio_similarity_two_sided(r[k], ref[k].median())
                for k in ["min_1d", "min_7d", "max_1d", "max_7d"]
            ]
        )
        timing = np.mean(
            [
                max(0.0, 1.0 - circular_distance(r[k], centers[k]) / (365.25 / 2))
                for k in ["doy_min", "doy_max"]
            ]
        )
        pulses = np.mean(
            [
                ratio_similarity_two_sided(r[k], ref[k].median(), offset=1.0)
                for k in [
                    "low_pulse_count",
                    "low_pulse_duration",
                    "high_pulse_count",
                    "high_pulse_duration",
                ]
            ]
        )
        rate = np.mean(
            [
                ratio_similarity_two_sided(r[k], ref[k].median(), offset=1.0)
                for k in ["rise_rate", "fall_rate", "reversals"]
            ]
        )
        rows.append(
            {
                "year": int(r.year),
                "water_monthly_magnitude": monthly,
                "water_extremes": extremes,
                "water_timing": timing,
                "water_pulses": pulses,
                "water_rate_frequency": rate,
                "water_resources": float(
                    np.mean([monthly, extremes, timing, pulses, rate])
                ),
            }
        )
    return pd.DataFrame(rows)


def clean_and_merge_sediment(
    flow: pd.DataFrame, sediment: pd.DataFrame, flow_version: str = "primary_flow"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stations = CONFIG["sediment_input_stations"] + [CONFIG["sediment_output_station"]]
    q = flow[flow.station.isin(stations) & (flow.data_version == flow_version)].copy()
    if flow_version != "primary_flow":
        q_up = flow[
            flow.station.isin(CONFIG["sediment_input_stations"])
            & (flow.data_version == "primary_flow")
        ].copy()
        q = pd.concat(
            [q[q.station == CONFIG["sediment_output_station"]], q_up], ignore_index=True
        )
    q["q"] = q.value.astype(float)
    q.loc[q.q <= 0, "q"] = np.nan
    q = q[["date", "year", "month", "station", "q"]]
    s = sediment[
        sediment.station.isin(stations) & (sediment.data_version == "primary_sediment")
    ].copy()
    s["ssc"] = s.value.astype(float)
    s.loc[s.ssc <= 0, "ssc"] = np.nan
    s = s[["date", "year", "month", "station", "ssc"]]
    d = q.merge(s, on=["date", "year", "month", "station"], how="inner").sort_values(
        ["station", "date"]
    )
    d["q"] = d.groupby("station", observed=True)["q"].transform(
        lambda x: x.interpolate(limit=2, limit_area="inside")
    )
    d["ssc"] = d.groupby("station", observed=True)["ssc"].transform(
        lambda x: x.interpolate(limit=3, limit_area="inside")
    )
    d["load_t_day"] = d.q * d.ssc * 86.4
    qa = (
        d.groupby(["station", "year"], observed=True)
        .agg(
            days=("date", "count"),
            valid_q=("q", "count"),
            valid_ssc=("ssc", "count"),
            valid_load=("load_t_day", "count"),
        )
        .reset_index()
    )
    qa["expected_days"] = qa.year.astype(int).map(days_in_year)
    qa["load_coverage"] = qa.valid_load / qa.expected_days
    return (d, qa)


def calculate_sediment(
    flow: pd.DataFrame,
    sediment: pd.DataFrame,
    flow_version: str = "primary_flow",
    ref_years: Tuple[int, int] = (1960, 1994),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d, qa = clean_and_merge_sediment(flow, sediment, flow_version=flow_version)
    annual_rows = []
    for (station, year), g in d.groupby(["station", "year"], observed=True):
        year = int(year)
        expected = days_in_year(year)
        valid = int(g.load_t_day.notna().sum())
        coverage = valid / expected
        load = np.nan
        if coverage >= CONFIG["min_daily_coverage"]:
            load = float(g.load_t_day.sum() * expected / valid)
        annual_rows.append(
            {
                "station": station,
                "year": year,
                "annual_load_t": load,
                "coverage": coverage,
            }
        )
    annual_station = pd.DataFrame(annual_rows)
    p = annual_station.pivot(
        index="year", columns="station", values="annual_load_t"
    ).reset_index()
    p["input_load_t"] = p[CONFIG["sediment_input_stations"]].sum(axis=1, min_count=2)
    p["output_load_t"] = p[CONFIG["sediment_output_station"]]
    ref = p[p.year.between(*ref_years)].dropna(subset=["input_load_t", "output_load_t"])
    slope, intercept, lower_slope, upper_slope = theilslopes(
        np.log1p(ref.output_load_t.to_numpy()),
        np.log1p(ref.input_load_t.to_numpy()),
        alpha=0.95,
    )
    p["predicted_natural_output_t"] = np.expm1(
        intercept + slope * np.log1p(p.input_load_t)
    )
    p["delivery_index"] = p.output_load_t / p.predicted_natural_output_t
    monthly_rows = []
    out_station = CONFIG["sediment_output_station"]
    for (year, month), g in d[d.station == out_station].groupby(["year", "month"]):
        valid = int(g.load_t_day.notna().sum())
        expected = len(g)
        coverage = valid / expected if expected else 0
        load = np.nan
        if coverage >= CONFIG["min_monthly_coverage"]:
            load = float(g.load_t_day.sum() * expected / valid)
        monthly_rows.append(
            {
                "year": int(year),
                "month": int(month),
                "monthly_load_t": load,
                "coverage": coverage,
            }
        )
    monthly = pd.DataFrame(monthly_rows)
    mp = monthly.pivot(index="year", columns="month", values="monthly_load_t")
    fractions = mp.div(mp.sum(axis=1), axis=0)
    valid_ref_years = [
        y
        for y in fractions.index
        if ref_years[0] <= y <= ref_years[1] and fractions.loc[y].notna().sum() == 12
    ]
    reference_fraction = fractions.loc[valid_ref_years].median(axis=0).to_numpy(float)
    timing_rows = []
    for year, row in fractions.iterrows():
        arr = row.to_numpy(float)
        if np.isfinite(arr).sum() == 12 and arr.sum() > 0:
            js = jensenshannon(
                arr / arr.sum(), reference_fraction / reference_fraction.sum(), base=2
            )
            timing_rows.append(
                {"year": int(year), "seasonal_similarity": float(1 - js)}
            )
    metrics = p.merge(pd.DataFrame(timing_rows), on="year", how="left")
    ref_metrics = metrics[metrics.year.between(*ref_years)]
    delivery_target = float(ref_metrics.delivery_index.median())
    timing_target = float(ref_metrics.seasonal_similarity.median())
    metrics["sediment_continuity"] = metrics.delivery_index.apply(
        lambda x: ratio_similarity_two_sided(x, delivery_target)
    )
    metrics["sediment_timing"] = metrics.seasonal_similarity.apply(
        lambda x: ratio_score_high_good(x, timing_target)
    )
    metrics["sediment"] = metrics[["sediment_continuity", "sediment_timing"]].mean(
        axis=1
    )
    metrics["theil_sen_slope"] = slope
    metrics["theil_sen_intercept"] = intercept
    metrics["delivery_target"] = delivery_target
    metrics["timing_target"] = timing_target
    return (metrics, annual_station, qa)


def calculate_water_environment(
    water_environment: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stations = CONFIG["water_quality_fixed_stations"]
    main_var = CONFIG["water_quality_primary_variable"]
    d = water_environment[
        (water_environment.data_version == "primary_water_quality")
        & water_environment.station.isin(stations)
        & water_environment.variable.isin(["TP", "TN", "DO"])
    ].copy()
    d["value"] = pd.to_numeric(d.value, errors="coerce")
    ref_start, ref_end = CONFIG["water_quality_reference_years"]
    tp_target = float(
        d[(d.variable == main_var) & d.year.between(ref_start, ref_end)].value.median()
    )
    d["primary_score"] = np.nan
    mask = d.variable == main_var
    d.loc[mask, "primary_score"] = d.loc[mask, "value"].apply(
        lambda x: ratio_score_low_good(x, tp_target)
    )
    station_year_rows = []
    for (year, station), g in d[d.variable == main_var].groupby(
        ["year", "station"], observed=True
    ):
        valid = int(g.primary_score.notna().sum())
        station_year_rows.append(
            {
                "year": int(year),
                "station": station,
                "n_months": valid,
                "TP_median_mg_L": float(g.value.median()) if valid else np.nan,
                "TP_score": float(g.primary_score.mean())
                if valid >= CONFIG["min_wq_months_per_station_year"]
                else np.nan,
            }
        )
    station_year = pd.DataFrame(station_year_rows)
    annual = (
        station_year.groupby("year")
        .agg(
            n_stations=("TP_score", "count"),
            water_environment=("TP_score", "mean"),
            TP_system_median_mg_L=("TP_median_mg_L", "median"),
        )
        .reset_index()
    )
    annual.loc[
        annual.n_stations < CONFIG["min_wq_stations_per_year"], "water_environment"
    ] = np.nan
    annual["TP_target_mg_L"] = tp_target
    targets = {
        "TP": float(
            d[(d.variable == "TP") & d.year.between(ref_start, ref_end)].value.median()
        ),
        "TN": float(
            d[(d.variable == "TN") & d.year.between(1997, 2001)].value.median()
        ),
        "DO": float(
            d[(d.variable == "DO") & d.year.between(1997, 2001)].value.median()
        ),
    }

    def supplementary_score(row: pd.Series) -> float:
        if row.variable in ["TP", "TN"]:
            return ratio_score_low_good(row.value, targets[row.variable])
        return ratio_score_high_good(row.value, targets[row.variable])

    d["supplementary_score"] = d.apply(supplementary_score, axis=1)
    sup_rows = []
    for (year, variable), g in d.groupby(["year", "variable"]):
        valid = int(g.supplementary_score.notna().sum())
        required = (
            CONFIG["min_wq_stations_per_year"]
            * CONFIG["min_wq_months_per_station_year"]
        )
        sup_rows.append(
            {
                "year": int(year),
                "variable": variable,
                "score": float(g.supplementary_score.mean())
                if valid >= required
                else np.nan,
                "valid_observations": valid,
            }
        )
    sup = (
        pd.DataFrame(sup_rows)
        .pivot(index="year", columns="variable", values="score")
        .reset_index()
    )
    for col in ["TP", "TN", "DO"]:
        if col not in sup.columns:
            sup[col] = np.nan
    sup["water_environment_three_variable_sensitivity"] = sup[["TP", "TN", "DO"]].mean(
        axis=1, skipna=False
    )
    return (annual, station_year, sup)


def calculate_aquatic_ecology(
    aquatic_ecology: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    primary = aquatic_ecology[
        (aquatic_ecology.data_version == "primary_ecology")
        & (aquatic_ecology.station == "Jianli")
    ].copy()
    primary = primary.sort_values("year")
    ref_start, ref_end = CONFIG["aquatic_ecology_reference_years"]
    target = float(primary[primary.year.between(ref_start, ref_end)].value.median())
    primary["aquatic_ecology"] = primary.value.apply(
        lambda x: ratio_score_high_good(x, target)
    )
    primary["aquatic_ecology_target_1e8_fry"] = target
    alternative = aquatic_ecology[
        (aquatic_ecology.data_version == "alternative_ecology")
        & (aquatic_ecology.station == "Jianli_section")
        & aquatic_ecology.is_four_major_carp
    ].copy()
    alt_target = float(
        alternative[alternative.year.between(ref_start, ref_end)].value.median()
    )
    alternative["aquatic_ecology_alternative"] = alternative.value.apply(
        lambda x: ratio_score_high_good(x, alt_target)
    )
    alternative["alternative_target_1e8_fry"] = alt_target
    return (
        primary.drop(columns="is_four_major_carp"),
        alternative.drop(columns="is_four_major_carp"),
    )


def combine_domains(
    water: pd.DataFrame,
    sediment: pd.DataFrame,
    water_environment: pd.DataFrame,
    aquatic_ecology: pd.DataFrame,
) -> pd.DataFrame:
    d = (
        water[["year", "water_resources"]]
        .merge(sediment[["year", "sediment"]], on="year", how="inner")
        .merge(water_environment[["year", "water_environment"]], on="year", how="inner")
        .merge(aquatic_ecology[["year", "aquatic_ecology"]], on="year", how="inner")
    )
    a, b = CONFIG["analysis_years"]
    d = d[d.year.between(a, b)].sort_values("year").copy()
    required = list(DOMAIN_COLUMNS.values())
    d["valid_domain_count"] = d[required].notna().sum(axis=1)
    d["system_heni"] = d[required].mean(axis=1, skipna=False)
    if d.system_heni.isna().any():
        years = d.loc[d.system_heni.isna(), "year"].tolist()
        raise ValueError(f"System HENI is unavailable for analysis years: {years}")
    return d


def runtime(args, name):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(output / (name + ".log"), mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.info("Starting %s", name)
    return output


def save(table, output, name):
    # Preserve source station-column order in translated pivot tables.
    columns = list(table.columns)
    positions = [i for i, column in enumerate(columns) if column in STATION_ORDER]
    stations = sorted((columns[i] for i in positions), key=STATION_ORDER.index)
    for position, station in zip(positions, stations):
        columns[position] = station
    table.loc[:, columns].to_csv(output / name, index=False, encoding="utf-8-sig")


def cli(description, restricted=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
    )
    if restricted:
        parser.add_argument(
            "--input-dir",
            type=Path,
            required=True,
            help="Authorized restricted CSV inputs; see data/README.md",
        )
    return parser


INPUT_FILES = {
    "flow": "01_flow_daily_selected.csv",
    "sediment": "02_sediment_daily_selected.csv",
    "water_environment": "03_water_quality_monthly_selected.csv",
    "aquatic_ecology": "04_ecology_annual_selected.csv",
}

# Fixed station order preserves grouping and export order.
STATION_ORDER = ('Guandukou', 'Yichang', 'Cuntan', 'Wulong', 'Hankou', 'Tuokou', 'Qingxichang', 'Jianli', 'Jianli_section')


def prepare_standard_input(frame, dataset):
    stations = set(frame['station'].dropna())
    unknown = stations.difference(STATION_ORDER)
    if unknown:
        raise ValueError(f'Unsupported standard station names: {sorted(unknown)}')
    frame['station'] = pd.Categorical(
        frame['station'], categories=[s for s in STATION_ORDER if s in stations],
        ordered=True,
    )
    if dataset == 'aquatic_ecology':
        frame['is_four_major_carp'] = frame['variable'].eq(
            'four_major_carp_eggs_and_fry_runoff'
        )
    return frame


def read_inputs(input_dir):
    data = {}
    for key, filename in INPUT_FILES.items():
        path = input_dir.resolve() / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Required authorized input missing: {filename}. See data/README.md."
            )
        frame = pd.read_csv(path, low_memory=False)
        required = ["date", "year", "station", "value", "unit", "data_version"]
        if key != "aquatic_ecology":
            required.append("month")
        if key in {"water_environment", "aquatic_ecology"}:
            required.append("variable")
        require_columns(frame, required, filename)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
        data[key] = prepare_standard_input(frame, key)
    return data


def main():
    parser = cli(
        "Calculate the four fixed HENI domains and their annual equal-weight aggregate.",
        True,
    )
    args = parser.parse_args()
    output = runtime(args, "02_heni_calculation")
    data = read_inputs(args.input_dir)
    ref = tuple(CONFIG["water_resources_sediment_reference_years"])
    iha, invalid = calculate_annual_iha(
        data["flow"], version="primary_flow", ref_years=ref
    )
    water = score_water_resources(iha, ref)
    sediment, station, sediment_qa = calculate_sediment(
        data["flow"], data["sediment"], flow_version="primary_flow", ref_years=ref
    )
    (
        water_environment,
        water_environment_station,
        water_environment_sensitivity,
    ) = calculate_water_environment(data["water_environment"])
    aquatic_ecology, aquatic_ecology_alt = calculate_aquatic_ecology(
        data["aquatic_ecology"]
    )
    domains = combine_domains(water, sediment, water_environment, aquatic_ecology)
    for table, filename in [
        (iha, "02_source_water_IHA_metrics.csv"),
        (water, "03_source_water_domain_scores.csv"),
        (sediment, "04_source_sediment_metrics_scores.csv"),
        (station, "05_source_sediment_station_annual_loads.csv"),
        (water_environment, "06_source_water_environment_scores.csv"),
        (water_environment_station, "07_source_water_environment_station_scores.csv"),
        (water_environment_sensitivity, "08_source_water_environment_sensitivity.csv"),
        (aquatic_ecology, "09_source_ecology_primary.csv"),
        (aquatic_ecology_alt, "10_source_ecology_alternative.csv"),
        (domains, "11_source_Fig3_system_and_domains.csv"),
        (invalid, "23_QA_invalid_flow.csv"),
        (sediment_qa, "24_QA_sediment_daily_coverage.csv"),
    ]:
        save(table, output, filename)
    logging.info("Completed: %d annual system records", len(domains))


if __name__ == "__main__":
    main()
