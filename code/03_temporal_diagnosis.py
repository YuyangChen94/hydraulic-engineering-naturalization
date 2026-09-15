"""Diagnose predefined temporal changes and decompose their domain contributions."""
from __future__ import annotations
import importlib
import logging
from typing import List, Sequence, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm

heni = importlib.import_module("02_heni_calculation")
CONFIG = heni.CONFIG
DOMAIN_COLUMNS = heni.DOMAIN_COLUMNS


def event_analysis(
    domains: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    phases = CONFIG["event_phases"]
    phase_rows = []
    for phase, (start, end) in phases.items():
        g = domains[domains.year.between(start, end)]
        row = {"phase": phase, "start_year": start, "end_year": end, "n_years": len(g)}
        for col in list(DOMAIN_COLUMNS.values()) + ["system_heni"]:
            row[col] = float(g[col].mean())
        phase_rows.append(row)
    phase_means = pd.DataFrame(phase_rows)
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
    contribution_rows = []
    for event, (pre_start, pre_end), (post_start, post_end) in comparisons:
        pre = domains[domains.year.between(pre_start, pre_end)]
        post = domains[domains.year.between(post_start, post_end)]
        domain_changes = {
            label: float(post[col].mean() - pre[col].mean())
            for label, col in DOMAIN_COLUMNS.items()
        }
        contributions = {
            label: CONFIG["domain_weights"][label] * change
            for label, change in domain_changes.items()
        }
        net = float(sum(contributions.values()))
        abs_sum = float(sum((abs(v) for v in contributions.values())))
        for label, value in contributions.items():
            signed_share = (
                value / net
                if abs(net) >= CONFIG["net_change_share_threshold"]
                else np.nan
            )
            absolute_share = abs(value) / abs_sum if abs_sum > 0 else np.nan
            contribution_rows.append(
                {
                    "event": event,
                    "domain": label,
                    "domain_change": domain_changes[label],
                    "domain_weight": CONFIG["domain_weights"][label],
                    "contribution_value": value,
                    "net_system_heni_change": net,
                    "signed_share_of_net_change": signed_share,
                    "absolute_contribution_share": absolute_share,
                    "signed_share_status": "stable"
                    if np.isfinite(signed_share)
                    else "not_reported_net_change_near_zero",
                }
            )
    return phase_means, pd.DataFrame(contribution_rows)


def segmented_regression_analysis(
    domains: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pre-specified interrupted time-series model with 2003 and 2011 knots.

    The model separates immediate level changes from slope changes. HC3
    heteroskedasticity-robust covariance with finite-sample t/F inference is
    used because only 20 annual observations are available. Results are
    event-aligned associations, not causal attribution to the reservoir alone.
    """
    years = domains["year"].astype(int)
    x = pd.DataFrame(
        {
            "const": 1.0,
            "time": years - int(years.min()),
            "step2003": (years >= 2003).astype(int),
            "slope2003": np.where(years >= 2003, years - 2003, 0),
            "step2011": (years >= 2011).astype(int),
            "slope2011": np.where(years >= 2011, years - 2011, 0),
        }
    )
    coefficient_rows: List[dict] = []
    event_rows: List[dict] = []
    slope_rows: List[dict] = []
    objects = list(DOMAIN_COLUMNS.items()) + [("system_heni", "system_heni")]
    for label, col in objects:
        model = sm.OLS(domains[col].to_numpy(float), x).fit(
            cov_type=CONFIG["segmented_covariance"], use_t=True
        )
        ci = model.conf_int(alpha=0.05)
        for term in model.params.index:
            coefficient_rows.append(
                {
                    "object": label,
                    "column": col,
                    "term": term,
                    "estimate": float(model.params[term]),
                    "robust_se": float(model.bse[term]),
                    "p_value": float(model.pvalues[term]),
                    "ci_low": float(ci.loc[term, 0]),
                    "ci_high": float(ci.loc[term, 1]),
                    "covariance": CONFIG["segmented_covariance"],
                    "n_years": int(model.nobs),
                    "r_squared": float(model.rsquared),
                }
            )
        for event, terms in {
            "2003_impoundment": ["step2003", "slope2003"],
            "2011_ecological_operation": ["step2011", "slope2011"],
        }.items():
            r = np.zeros((2, len(model.params)))
            for i, term in enumerate(terms):
                r[i, list(model.params.index).index(term)] = 1.0
            test = model.wald_test(r, use_f=True, scalar=True)
            event_rows.append(
                {
                    "object": label,
                    "column": col,
                    "event": event,
                    "joint_F": float(test.fvalue),
                    "joint_p": float(test.pvalue),
                    "df_num": float(test.df_num),
                    "df_denom": float(test.df_denom),
                    "test_definition": "joint_test_of_level_and_slope_change",
                    "covariance": CONFIG["segmented_covariance"],
                }
            )
        for phase, terms in {
            "pre_impoundment": ["time"],
            "post_impoundment_pre_ecological_operation": ["time", "slope2003"],
            "post_ecological_operation": ["time", "slope2003", "slope2011"],
        }.items():
            contrast = np.zeros(len(model.params))
            for term in terms:
                contrast[list(model.params.index).index(term)] = 1.0
            test = model.t_test(contrast)
            slope_rows.append(
                {
                    "object": label,
                    "column": col,
                    "phase": phase,
                    "slope_per_year": float(np.asarray(test.effect).ravel()[0]),
                    "robust_se": float(np.asarray(test.sd).ravel()[0]),
                    "t_value": float(np.asarray(test.tvalue).ravel()[0]),
                    "p_value": float(np.asarray(test.pvalue).ravel()[0]),
                    "ci_low": float(np.asarray(test.conf_int()).reshape(-1, 2)[0, 0]),
                    "ci_high": float(np.asarray(test.conf_int()).reshape(-1, 2)[0, 1]),
                    "covariance": CONFIG["segmented_covariance"],
                }
            )
    return (
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(event_rows),
        pd.DataFrame(slope_rows),
    )


def build_segmented_design(years: Sequence[int]) -> pd.DataFrame:
    years = pd.Series(years, dtype=int)
    return pd.DataFrame(
        {
            "const": 1.0,
            "time": years - int(years.min()),
            "step2003": (years >= 2003).astype(int),
            "slope2003": np.where(years >= 2003, years - 2003, 0),
            "step2011": (years >= 2011).astype(int),
            "slope2011": np.where(years >= 2011, years - 2011, 0),
        }
    )


def segmented_fit_values(df: pd.DataFrame, column: str) -> pd.DataFrame:
    x = build_segmented_design(df["year"].astype(int))
    model = sm.OLS(df[column].to_numpy(float), x).fit()
    pred = model.predict(x)
    out = df[["year", column]].copy()
    out["fitted"] = pred
    return out


def main():
    args = heni.cli(
        "Predefined 2003/2011 temporal diagnosis and stage contribution decomposition."
    ).parse_args()
    out = heni.runtime(args, "03_temporal_diagnosis")
    domains = pd.read_csv(out / "11_source_Fig3_system_and_domains.csv")
    phase, contributions = event_analysis(domains)
    coefficients, events, slopes = segmented_regression_analysis(domains)
    for table, name in [
        (phase, "12_source_stage_means.csv"),
        (contributions, "14_source_event_contributions.csv"),
        (coefficients, "16_source_segmented_regression_coefficients.csv"),
        (events, "17_source_segmented_event_tests.csv"),
        (slopes, "18_source_segment_slopes.csv"),
    ]:
        heni.save(table, out, name)
    logging.info("Completed both predefined temporal diagnoses")


if __name__ == "__main__":
    main()
