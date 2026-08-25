"""
Derivation layer: patient records -> indicator numerators and denominators.
=====================================================================

The separation that matters
---------------------------
Three things happen here, deliberately kept apart:

1. **Flagging.** Turn each patient record into a set of booleans whose
   names match the registry exactly (``flag_ct_60``, ``den_ischaemic``).
   All the clinical inclusion logic lives in one place, and every
   indicator that says "within 4 hours" means the same four hours.

2. **Aggregation.** Sum those booleans by period and by group. Nothing
   clinical happens here -- it is arithmetic, and it is the same
   arithmetic for every indicator.

3. **Charting.** Hand the aggregated numerator/denominator to the SPC
   engine. It knows nothing about stroke.

Getting this ordering wrong is the most common failure in home-built
audit dashboards: inclusion logic gets rewritten inside each chart, the
definitions quietly diverge, and two indicators on the same page end up
using different denominators for "stroke admission".

Denominator discipline
----------------------
Every denominator here is a *cohort*, expressed as a boolean per patient.
That makes the exclusion visible and auditable:

    den_all_stroke        every confirmed stroke admission
    den_ischaemic         ischaemic strokes only
    den_thrombolysed      patients who actually received thrombolysis
    den_af_survivors      AF, survived to discharge (a dead patient
                          cannot be discharged on an anticoagulant, and
                          including them makes the service look worse
                          for a reason unrelated to prescribing)
    den_pt_needed         recorded as requiring physiotherapy
    den_inpatient_rehab   survived and stayed >= 3 days
    den_immobile          NIHSS >= 6 -- a crude proxy; replace with your
                          local mobility coding if you have it

The "needed" denominators are clinical judgements, which means a service
can improve those indicators by narrowing who it says needs input. The
dashboard therefore always shows the denominator trend beside the rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import spc
from .standards import METRICS, MetricSpec, Standard, Threshold, metric

FREQ_LABELS = {"MS": "Month", "QS": "Quarter", "W-MON": "Week"}


# ---------------------------------------------------------------------------
# 1. Flagging
# ---------------------------------------------------------------------------
def derive_flags(adm: pd.DataFrame) -> pd.DataFrame:
    """Add every ``flag_*`` and ``den_*`` column the registry refers to."""
    d = adm.copy()
    ischaemic = d["stroke_type"] == "Ischaemic"
    survived = ~d["died_inpatient"]

    # ---- denominators (cohorts) ----------------------------------------
    d["den_all_stroke"] = True
    d["den_ischaemic"] = ischaemic
    d["den_thrombolysed"] = d["thrombolysed"]
    d["den_survivors"] = survived
    d["den_ischaemic_survivors"] = ischaemic & survived
    d["den_af_survivors"] = d["af"] & survived
    d["den_survivors_prehome"] = survived & d["admitted_from_home"]
    d["den_esd_eligible"] = d["esd_eligible"]
    d["den_inpatient_rehab"] = survived & (d["los_days"] >= 3)
    d["den_immobile"] = d["immobile"]
    d["den_pt_needed"] = d["pt_needed"]
    d["den_ot_needed"] = d["ot_needed"]
    d["den_slt_needed"] = d["slt_needed"]

    # ---- hyperacute -----------------------------------------------------
    # Guard against implausible values rather than letting them score as
    # compliant. A door-to-imaging time of -180 minutes (arrival and scan
    # timestamped against different dates) satisfies "<= 60" and would be
    # silently counted as excellent performance. Impossible values must
    # fail the numerator, and the Data Quality page reports how many.
    plausible_ct = d["door_to_ct_min"].between(0, 20000)
    d["flag_ct_60"] = plausible_ct & (d["door_to_ct_min"] <= 60)
    d["flag_ct_720"] = plausible_ct & (d["door_to_ct_min"] <= 720)
    d["flag_thrombolysed"] = ischaemic & d["thrombolysed"]
    d["flag_thrombectomy"] = ischaemic & d["thrombectomy"]
    d["flag_dtn_60"] = d["thrombolysed"] & (d["door_to_needle_min"] <= 60)
    d["flag_nihss_recorded"] = d["nihss_recorded"]

    # ---- stroke unit ----------------------------------------------------
    d["flag_su_4h"] = d["time_to_su_hours"] <= 4
    # "90% of stay on a stroke unit": time before the stroke unit counts
    # against the patient, so express it as a share of total stay in days.
    share_off_unit = (d["time_to_su_hours"] / 24.0) / d["los_days"].clip(lower=1)
    d["flag_su_90pct_stay"] = share_off_unit <= 0.10
    d["flag_swallow_4h"] = d["swallow_screen_hours"] <= 4
    d["flag_swallow_24h"] = d["swallow_screen_hours"] <= 24
    d["flag_mdt_goals_5d"] = d["mdt_goals_5d"]
    d["flag_vte_ipc"] = d["vte_ipc_24h"]

    # ---- therapy access -------------------------------------------------
    d["flag_pt_72h"] = d["pt_needed"] & (d["pt_assess_hours"] <= 72)
    d["flag_ot_72h"] = d["ot_needed"] & (d["ot_assess_hours"] <= 72)
    d["flag_slt_72h"] = d["slt_needed"] & (d["slt_assess_hours"] <= 72)

    # ---- safety ---------------------------------------------------------
    d["flag_hap"] = d["hap"]
    d["bed_days"] = d["los_days"]

    # ---- discharge and outcome -----------------------------------------
    d["flag_af_anticoag"] = d["af"] & survived & d["af_anticoagulated"]
    d["flag_antiplatelet"] = ischaemic & survived & d["antiplatelet"]
    d["flag_esd"] = d["esd_referral"]
    d["flag_home"] = d["discharge_destination"] == "Usual residence"
    d["flag_mood_screen"] = d["mood_screened"]
    d["flag_died"] = d["died_inpatient"]
    d["flag_mrs_0_2"] = d["mrs_discharge"] <= 2
    d["flag_readmit_30d"] = d["readmitted_30d"]

    # Every flag must be nested inside its denominator, or the proportion
    # can exceed 1 and the p-chart limits become nonsense. Enforce it
    # rather than trusting the definitions above.
    for key, m in METRICS.items():
        if m.chart == "p" and m.numerator in d.columns and m.denominator in d.columns:
            d[m.numerator] = d[m.numerator].fillna(False).astype(bool) & \
                d[m.denominator].fillna(False).astype(bool)
    return d


# ---------------------------------------------------------------------------
# 2. Therapy day-level derivation
# ---------------------------------------------------------------------------
def therapy_patient_level(sessions: pd.DataFrame) -> pd.DataFrame:
    """Per admission and discipline: applicable days, days delivered, dose.

    ``minutes_per_day`` divides by *applicable* days, not by days on which
    therapy happened. This is the definition that matters clinically: a
    patient who gets 60 minutes on two days out of ten has received 12
    minutes a day of rehabilitation, not 60. Dividing by delivered days
    is the most common way therapy dose gets overstated.
    """
    s = sessions
    if "day_45min" not in s.columns:
        s = s.assign(day_45min=(s["minutes"] >= 45))
    g = s.groupby(["admission_id", "discipline"], observed=True)
    out = g.agg(
        days_applicable=("applicable", "sum"),
        days_with_therapy=("attended", "sum"),
        days_45min=("day_45min", "sum"),
        total_minutes=("minutes", "sum"),
        sessions_delivered=("n_sessions", "sum"),
    ).reset_index()
    out["minutes_per_day"] = out["total_minutes"] / out["days_applicable"].clip(lower=1)
    out["pct_days"] = 100 * out["days_with_therapy"] / out["days_applicable"].clip(lower=1)
    return out


def _therapy_wide(adm: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Patient-level therapy dose columns, one row per admission."""
    pl = therapy_patient_level(sessions)
    wide = pl.pivot(index="admission_id", columns="discipline",
                    values="minutes_per_day")
    wide.columns = [f"{c.lower()}_min_per_day" for c in wide.columns]
    return wide.reset_index()


def therapy_day_counts(sessions: pd.DataFrame, freq: str,
                       group_col: str | None) -> pd.DataFrame:
    """Period-level therapy-day numerators and denominators.

    Attributed by *session date*, not by admission date, because these
    indicators are about what the service delivered in that month.
    """
    s = sessions.copy()
    s["period"] = s["date"].dt.to_period(_pandas_freq(freq)).dt.to_timestamp()
    s["day_45min"] = s["minutes"] >= 45
    keys = ["period"] + ([group_col] if group_col else [])

    frames = []
    for disc in ("PT", "OT", "SLT"):
        sub = s[s["discipline"] == disc]
        agg = sub.groupby(keys, observed=True).agg(
            **{f"{disc.lower()}_days_applicable": ("applicable", "sum"),
               f"{disc.lower()}_days_with_therapy": ("attended", "sum"),
               f"{disc.lower()}_days_45min": ("day_45min", "sum")}
        )
        frames.append(agg)

    weekend = s[s["weekend"]]
    any_day = weekend.groupby(keys + ["admission_id", "date"], observed=True)["attended"].max()
    we = any_day.groupby(keys, observed=True).agg(
        weekend_days_applicable="size", weekend_days_with_therapy="sum")
    frames.append(we)

    out = pd.concat(frames, axis=1).fillna(0).reset_index()
    for c in out.columns:
        if c not in keys:
            out[c] = out[c].astype(float)
    return out


def _pandas_freq(freq: str) -> str:
    return {"MS": "M", "QS": "Q", "W-MON": "W-MON"}.get(freq, "M")


# ---------------------------------------------------------------------------
# 3. Aggregation to a period table
# ---------------------------------------------------------------------------
_VALUE_COLS = ["door_to_needle_min", "onset_to_door_min", "time_to_su_hours",
               "los_days", "door_to_puncture_min", "door_to_ct_min",
               "pt_min_per_day", "ot_min_per_day", "slt_min_per_day"]

_COUNT_COLS = ["falls", "pressure_ulcers", "bed_days"]


def build_period_table(adm: pd.DataFrame, sessions: pd.DataFrame, *,
                       freq: str = "MS", group_col: str | None = None) -> pd.DataFrame:
    """One row per period (and group), carrying every column the registry needs.

    Continuous indicators are summarised by the **median**, not the mean.
    Clinical timing data is strongly right-skewed -- a handful of
    patients found late, or one 14-hour scan delay, drags a mean far
    above anything a typical patient experiences. The median describes
    the typical patient, which is what a service-improvement chart is
    for. Where the tail itself is the concern, plot the 90th centile as
    a separate series rather than switching summary statistic.
    """
    d = adm if "den_all_stroke" in adm.columns else derive_flags(adm)
    d = d.merge(_therapy_wide(d, sessions), on="admission_id", how="left")

    d = d.copy()
    d["period"] = d["arrival_datetime"].dt.to_period(_pandas_freq(freq)).dt.to_timestamp()
    keys = ["period"] + ([group_col] if group_col else [])

    bool_cols = [c for c in d.columns if c.startswith(("flag_", "den_"))]
    agg_spec: dict[str, tuple] = {c: (c, "sum") for c in bool_cols}
    for c in _COUNT_COLS:
        if c in d.columns:
            agg_spec[c] = (c, "sum")
    for c in _VALUE_COLS:
        if c in d.columns:
            agg_spec[c] = (c, "median")
    agg_spec["n_admissions"] = ("admission_id", "count")
    agg_spec["median_age"] = ("age", "median")
    agg_spec["median_nihss"] = ("nihss", "median")
    agg_spec["pct_haemorrhage"] = ("stroke_type",
                                   lambda s: 100 * float((s != "Ischaemic").mean()))

    tbl = d.groupby(keys, observed=True).agg(**agg_spec).reset_index()

    therapy = therapy_day_counts(sessions, freq, group_col)
    tbl = tbl.merge(therapy, on=keys, how="left")
    for c in therapy.columns:
        if c not in keys:
            tbl[c] = tbl[c].fillna(0.0)

    # A complete period axis matters: a month with no admissions must
    # appear as a gap, not silently close up and make a 3-month drift
    # look like a 2-month one.
    full = pd.date_range(tbl["period"].min(), tbl["period"].max(), freq=freq)
    if group_col:
        idx = pd.MultiIndex.from_product([full, sorted(tbl[group_col].unique())],
                                         names=["period", group_col])
        tbl = tbl.set_index(["period", group_col]).reindex(idx).reset_index()
    else:
        tbl = tbl.set_index("period").reindex(full).rename_axis("period").reset_index()
    return tbl.sort_values(keys).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Registry -> control chart
# ---------------------------------------------------------------------------
def metric_series(tbl: pd.DataFrame, key: str, *,
                  standard: Standard = "SSNAP",
                  overrides: dict[str, float] | None = None,
                  rule_set: str = "nhs",
                  laney: bool | str = "auto",
                  baseline: int | None = None,
                  phases: list[int] | None = None) -> spc.ChartResult:
    """Build the control chart for one registry key from a period table."""
    spec, thr = metric(key, standard, overrides)
    x = tbl["period"]

    if spec.chart == "p":
        num = tbl[spec.numerator].astype(float)
        den = tbl[spec.denominator].astype(float)
        # Suppress periods below the minimum denominator rather than
        # plotting a 0% or 100% built from three patients. The point is
        # dropped from the chart, not from the pooled centre line -- the
        # cases still count toward the process rate.
        num = num.where(den >= spec.min_denominator)
        den = den.where(den >= spec.min_denominator)
        res = spc.p_chart(num, den, x, laney=laney, target=thr.target,
                          higher_is_better=spec.higher_is_better,
                          baseline=baseline, phases=phases, rule_set=rule_set,
                          unit=spec.unit, label=spec.label)
    elif spec.chart == "xmr":
        lower = 0.0 if spec.unit in ("minutes", "hours", "days", "minutes/day") else None
        res = spc.xmr(tbl[spec.value_col], x, target=thr.target,
                      higher_is_better=spec.higher_is_better,
                      baseline=baseline, phases=phases, rule_set=rule_set,
                      lower_bound=lower, unit=spec.unit, label=spec.label)
    elif spec.chart == "u":
        res = spc.u_chart(tbl[spec.count_col], tbl[spec.exposure_col], x,
                          multiplier=spec.multiplier, laney=laney,
                          target=thr.target, higher_is_better=spec.higher_is_better,
                          baseline=baseline, rule_set=rule_set,
                          unit=spec.unit, label=spec.label)
    else:  # pragma: no cover
        raise ValueError(f"unknown chart type {spec.chart!r}")

    res.meta["metric_key"] = key
    res.meta["spec"] = spec
    res.meta["threshold"] = thr
    res.meta["ambition"] = thr.ambition
    return res


def current_value(tbl: pd.DataFrame, key: str, periods: int = 3) -> float | None:
    """Rolling value over the last N periods.

    Pooled over the window, not averaged across periods -- otherwise a
    quiet month gets the same weight as a busy one. This is the number
    to put on a tile; the chart is what says whether it means anything.
    """
    spec = METRICS[key]
    tail = tbl.tail(periods)
    if spec.chart == "p":
        den = tail[spec.denominator].sum()
        return None if not den else 100 * float(tail[spec.numerator].sum()) / float(den)
    if spec.chart == "u":
        exp = tail[spec.exposure_col].sum()
        return None if not exp else spec.multiplier * float(tail[spec.count_col].sum()) / float(exp)
    vals = tail[spec.value_col].dropna()
    return None if vals.empty else float(vals.median())


def funnel_for_metric(adm: pd.DataFrame, key: str, group_col: str = "site",
                      standard: Standard = "SSNAP") -> dict | None:
    """Cross-sectional funnel for any proportion metric."""
    spec, _thr = metric(key, standard)
    if spec.chart != "p":
        return None
    d = adm if "den_all_stroke" in adm.columns else derive_flags(adm)
    g = d.groupby(group_col, observed=True).agg(
        num=(spec.numerator, "sum"), den=(spec.denominator, "sum")).reset_index()
    g = g[g["den"] > 0]
    if len(g) < 3:
        return None
    return spc.funnel_plot(g["num"], g["den"], g[group_col])


def mrs_distribution(adm: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    """mRS 0-6 distribution, for a shift ('Grotta bar') plot.

    The dichotomy at 0-2 throws away most of the information in the
    scale. A patient moving from mRS 5 to mRS 4 -- from bedbound to
    needing assistance to walk -- is a large clinical gain that the
    dichotomised indicator scores as zero. The shift plot shows the whole
    distribution, which is how stroke trials have reported for two decades.
    """
    keys = [group_col] if group_col else []
    counts = (adm.groupby(keys + ["mrs_discharge"], observed=True)
              .size().rename("n").reset_index())
    total = counts.groupby(keys, observed=True)["n"].transform("sum") if keys else counts["n"].sum()
    counts["pct"] = 100 * counts["n"] / total
    return counts
