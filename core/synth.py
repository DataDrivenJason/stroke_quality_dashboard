"""
Synthetic stroke service data generator.
=====================================================================

Why synthetic data has to be *built*, not sampled
-------------------------------------------------
A dashboard demonstrated on random noise teaches the wrong lesson: every
chart shows common cause, every icon is grey, and the reader concludes
SPC finds nothing. Equally, data drawn from a single stationary
distribution cannot exercise the run rules at all.

So this generator constructs a service with a *history*. Specific,
nameable things happen to specific sites at specific times, and the
charts should find them. If you change the SPC code and these stop being
detected, you have broken something.

Planted signals (the ground truth for `tests/test_spc.py`)
----------------------------------------------------------
1. Riverside General, Feb 2025 -- thrombolysis pathway redesign
   (pre-alert, CT-first, drug on the scanner table). Door-to-needle
   steps down ~20 minutes. Expect: XmR shift, special cause improvement.
2. Northbay Regional, from Oct 2025 -- sustained medical bed pressure.
   Time to stroke unit drifts upward month on month; the 4-hour
   proportion falls. Expect: XmR trend and a p-chart shift, and the
   CUSUM should catch it earlier than the Shewhart chart.
3. Lakeview General, Jun 2025 - Feb 2026 -- two physiotherapy vacancies.
   Percentage of days with PT falls; the "no therapist available"
   share of missed sessions roughly doubles. Recovers on recruitment.
   Expect: a clear p'-chart shift, and demand-capacity gap on the
   Caseload page.
4. St Brendan's, Nov 2024 -- a cluster of falls on the rehab ward
   following a bay reconfiguration. Expect: u-chart single-point signal.
5. All sites, Apr 2024 -- swallow screening moved to the emergency
   department triage nurse. Screening within 4 hours improves. Expect:
   p-chart shift upward across all sites.
6. Structural, always on -- weekends. Therapy delivery drops sharply on
   Saturday and Sunday, which is the single largest determinant of total
   rehabilitation dose and is visible in every therapy indicator.
7. Structural, always on -- winter. Admissions rise, length of stay
   rises, therapy minutes fall, pneumonia rises. Seasonality is common
   cause: a well-behaved chart should NOT flag December every year, and
   if yours does, your limits are too tight.

Everything else is drawn from distributions chosen to match the shape,
not the exact values, of published stroke audit data: right-skewed
timings (lognormal), skewed severity (gamma), and outcome probabilities
that respond monotonically to age, severity, stroke type and prior
function.

This is simulated data. It is realistic in structure and must never be
quoted as evidence about any real service.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20260824

SITES = pd.DataFrame(
    [
        # name,                     admissions/month, thrombectomy centre, PT WTE, OT WTE, SLT WTE
        ("St Brendan's University Hospital", 62, True, 6.4, 5.6, 3.4),
        ("Riverside General", 44, False, 4.2, 3.8, 2.2),
        ("Northbay Regional", 33, False, 3.0, 2.6, 1.6),
        ("Lakeview General", 26, False, 2.6, 2.2, 1.4),
    ],
    columns=["site", "monthly_admissions", "thrombectomy_centre", "pt_wte", "ot_wte", "slt_wte"],
)

START = pd.Timestamp("2023-09-01")
END = pd.Timestamp("2026-08-31")

DISCIPLINES = ["PT", "OT", "SLT"]
MISS_REASONS = [
    "Patient medically unwell",
    "No therapist available",
    "Patient declined",
    "Off ward for investigation",
    "Patient fatigued",
]


# ---------------------------------------------------------------------------
def _month_index(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period("M").dt.to_timestamp()


def _lognormal(rng, median: float, sigma: float, size) -> np.ndarray:
    """Lognormal parameterised by its median, which is how clinical
    timing data is actually reported. mu = ln(median)."""
    return rng.lognormal(mean=np.log(median), sigma=sigma, size=size)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
def generate(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    months = pd.date_range(START, END, freq="MS")
    rows = []
    for _, s in SITES.iterrows():
        for m in months:
            # Winter surge: cosine peaking in January, +/- 12%.
            seasonal = 1.0 + 0.12 * np.cos(2 * np.pi * (m.month - 1) / 12)
            n = rng.poisson(s.monthly_admissions * seasonal)
            rows.append((s.site, m, n))
    plan = pd.DataFrame(rows, columns=["site", "month", "n"])

    total = int(plan["n"].sum())
    site = np.repeat(plan["site"].to_numpy(), plan["n"].to_numpy())
    month = np.repeat(plan["month"].to_numpy(), plan["n"].to_numpy())
    month = pd.Series(month)

    # ---- arrival timestamp: uniform within month, with a real-world
    # time-of-day profile (daytime peak, small hours trough).
    day_offset = rng.integers(0, 28, total)
    hour_weights = np.array([2, 1.5, 1.2, 1.1, 1.2, 1.8, 3.0, 4.2, 5.4, 6.0, 6.2, 5.9,
                             5.6, 5.4, 5.3, 5.2, 5.0, 4.7, 4.3, 3.9, 3.4, 3.0, 2.6, 2.2])
    hour = rng.choice(24, size=total, p=hour_weights / hour_weights.sum())
    minute = rng.integers(0, 60, total)
    arrival = (month + pd.to_timedelta(day_offset, "D")
               + pd.to_timedelta(hour, "h") + pd.to_timedelta(minute, "m"))
    arrival = pd.Series(arrival)
    out_of_hours = (hour < 8) | (hour >= 18) | (arrival.dt.dayofweek >= 5)

    df = pd.DataFrame({
        "admission_id": [f"A{i:06d}" for i in range(total)],
        "site": site,
        "month": month.to_numpy(),
        "arrival_datetime": arrival.to_numpy(),
        "out_of_hours": out_of_hours.to_numpy(),
    })
    df["month_num"] = ((df["month"].dt.year - START.year) * 12
                       + df["month"].dt.month - START.month)
    is_tx_centre = df["site"].map(dict(zip(SITES["site"], SITES["thrombectomy_centre"])))

    # ---- demographics and severity -------------------------------------
    df["age"] = np.clip(rng.normal(74, 13, total), 28, 101).round().astype(int)
    df["sex"] = rng.choice(["Female", "Male"], total, p=[0.51, 0.49])
    df["stroke_type"] = rng.choice(
        ["Ischaemic", "Intracerebral haemorrhage"], total, p=[0.855, 0.145])
    ischaemic = (df["stroke_type"] == "Ischaemic").to_numpy()

    # NIHSS: gamma is the right family -- a long right tail of severe
    # strokes over a mode of 3-5. Haemorrhage shifted higher.
    nihss = rng.gamma(1.7, 4.2, total) + np.where(ischaemic, 0, 3.5)
    df["nihss"] = np.clip(nihss, 0, 42).round().astype(int)
    df["prestroke_mrs"] = rng.choice([0, 1, 2, 3, 4, 5], total,
                                     p=[0.44, 0.20, 0.14, 0.12, 0.08, 0.02])
    df["af"] = rng.random(total) < _logistic(-2.4 + 0.030 * (df["age"] - 74))
    df["diabetes"] = rng.random(total) < 0.21
    df["hypertension"] = rng.random(total) < 0.62
    df["previous_stroke"] = rng.random(total) < 0.19
    df["arrived_by_ambulance"] = rng.random(total) < 0.73
    df["lives_alone"] = rng.random(total) < 0.31
    df["admitted_from_home"] = rng.random(total) < 0.91

    # ---- onset to door --------------------------------------------------
    onset_known = rng.random(total) < 0.67
    o2d = _lognormal(rng, 145, 0.95, total)
    o2d = np.where(df["arrived_by_ambulance"], o2d * 0.72, o2d)
    df["onset_known"] = onset_known
    df["onset_to_door_min"] = np.where(onset_known, np.round(o2d), np.nan)

    # ---- door to CT -----------------------------------------------------
    # SIGNAL 5 raises the whole front-door tempo slightly from Apr 2024;
    # out-of-hours arrivals wait longer for reporting.
    ct_median = np.full(total, 52.0)
    ct_median += np.where(out_of_hours.to_numpy(), 16.0, 0.0)
    ct_median -= np.where(df["month_num"] >= 7, 5.0, 0.0)
    # Suspected thrombolysis candidates are prioritised.
    fast_track = ischaemic & onset_known & (o2d < 210) & (df["nihss"].to_numpy() >= 3)
    ct_median = np.where(fast_track, ct_median * 0.38, ct_median)
    df["door_to_ct_min"] = np.round(np.clip(
        _lognormal(rng, 1.0, 0.62, total) * ct_median, 4, 4000))

    # ---- thrombolysis ---------------------------------------------------
    eligible = (ischaemic & onset_known & (o2d < 240)
                & (df["nihss"].to_numpy() >= 4) & (df["prestroke_mrs"].to_numpy() <= 3))
    df["thrombolysis_eligible"] = eligible
    # Calibrated so the overall thrombolysis rate lands around 12% of
    # ischaemic strokes, which is where mature European services sit.
    # About a third of ischaemic strokes clear the eligibility screen
    # above, and roughly 40% of those are treated once contraindications,
    # rapid improvement and patient choice are accounted for.
    p_lyse = np.where(eligible, 0.40, 0.0) * np.where(out_of_hours.to_numpy(), 0.88, 1.0)
    df["thrombolysed"] = rng.random(total) < p_lyse

    # SIGNAL 1: Riverside General door-to-needle step change, Feb 2025.
    riverside_after = (df["site"] == "Riverside General") & (df["month_num"] >= 17)
    dtn_median = np.full(total, 49.0)
    dtn_median += np.where(out_of_hours.to_numpy(), 9.0, 0.0)
    dtn_median += np.where(df["site"] == "Lakeview General", 7.0, 0.0)
    dtn_median -= np.where(riverside_after, 20.0, 0.0)
    dtn = np.round(_lognormal(rng, 1.0, 0.38, total) * dtn_median)
    df["door_to_needle_min"] = np.where(df["thrombolysed"], np.clip(dtn, 12, 300), np.nan)

    # ---- thrombectomy ---------------------------------------------------
    lvo_proxy = ischaemic & (df["nihss"].to_numpy() >= 10)
    p_evt = np.where(lvo_proxy, np.where(is_tx_centre.to_numpy(), 0.42, 0.24), 0.0)
    p_evt = p_evt * np.where(onset_known, 1.0, 0.45)
    df["thrombectomy"] = rng.random(total) < p_evt
    df["thrombectomy_transferred"] = df["thrombectomy"] & ~is_tx_centre.to_numpy()
    d2p = _lognormal(rng, 1.0, 0.42, total) * np.where(is_tx_centre.to_numpy(), 96.0, 176.0)
    df["door_to_puncture_min"] = np.where(df["thrombectomy"], np.round(d2p), np.nan)

    # ---- stroke unit access --------------------------------------------
    # SIGNAL 2: Northbay bed pressure -- a *ramp*, not a step, from Oct 2025.
    northbay_drift = np.where(
        (df["site"] == "Northbay Regional") & (df["month_num"] >= 25),
        (df["month_num"] - 25).clip(lower=0) * 0.46, 0.0)
    su_median = np.full(total, 3.1)
    su_median += np.where(out_of_hours.to_numpy(), 1.5, 0.0)
    su_median += northbay_drift
    su_median += np.where(df["site"] == "St Brendan's University Hospital", -0.5, 0.0)
    # Winter bed pressure applies everywhere.
    winter = df["month"].dt.month.isin([12, 1, 2]).to_numpy()
    su_median += np.where(winter, 0.85, 0.0)
    t2su = _lognormal(rng, 1.0, 0.72, total) * su_median
    df["time_to_su_hours"] = np.round(np.clip(t2su, 0.2, 300), 2)
    df["su_direct"] = df["time_to_su_hours"] < 4

    # ---- swallow screening ---------------------------------------------
    # SIGNAL 5: ED triage screening from Apr 2024 (month_num >= 7).
    sw_median = np.where(df["month_num"] >= 7, 1.7, 3.4)
    sw_median = sw_median + np.where(out_of_hours.to_numpy(), 0.9, 0.0)
    df["swallow_screen_hours"] = np.round(
        np.clip(_lognormal(rng, 1.0, 0.78, total) * sw_median, 0.1, 200), 2)
    df["dysphagia"] = rng.random(total) < _logistic(-1.9 + 0.115 * df["nihss"].to_numpy())

    df["nihss_recorded"] = rng.random(total) < np.where(
        df["month_num"] >= 5, 0.965, 0.90)

    # ---- mortality and length of stay -----------------------------------
    # Intercept calibrated to an overall in-hospital mortality near 12%,
    # the usual range for a mixed acute stroke population. The covariate
    # slopes are chosen to give the right *ordering* and rough gradient --
    # age, severity, haemorrhage and prior dependency all raise risk --
    # not to reproduce any published model's coefficients.
    lp_death = (-3.68 + 0.041 * (df["age"] - 74) + 0.115 * df["nihss"]
                + 0.55 * (~ischaemic).astype(int) + 0.30 * df["prestroke_mrs"]
                - 0.28 * df["thrombectomy"].astype(int))
    df["died_inpatient"] = rng.random(total) < _logistic(lp_death.to_numpy())

    los_median = (5.5 + 0.62 * df["nihss"].to_numpy()
                  + 1.2 * df["prestroke_mrs"].to_numpy()
                  + 0.055 * (df["age"].to_numpy() - 74)
                  + 2.4 * df["lives_alone"].to_numpy())
    los_median = np.clip(los_median, 2.0, 90)
    los_median = los_median * np.where(winter, 1.12, 1.0)
    los = _lognormal(rng, 1.0, 0.55, total) * los_median
    # Deaths concentrate early in the stay.
    los = np.where(df["died_inpatient"], np.minimum(los, rng.gamma(2.0, 4.0, total) + 1), los)
    df["los_days"] = np.round(np.clip(los, 1, 240)).astype(int)
    df["discharge_datetime"] = df["arrival_datetime"] + pd.to_timedelta(df["los_days"], "D")

    # ---- complications ---------------------------------------------------
    lp_hap = (-3.25 + 0.072 * df["nihss"] + 1.05 * df["dysphagia"].astype(int)
              + 0.024 * (df["age"] - 74)
              + 0.35 * (df["swallow_screen_hours"] > 24).astype(int))
    lp_hap = lp_hap + np.where(winter, 0.28, 0.0)
    df["hap"] = rng.random(total) < _logistic(lp_hap.to_numpy())

    # SIGNAL 4: falls cluster at St Brendan's in Nov 2024 (month_num == 14).
    falls_lambda = (df["los_days"].to_numpy() / 1000.0 * 5.2
                    * (1 + 0.05 * np.clip(df["nihss"].to_numpy() - 4, 0, None)))
    fall_boost = ((df["site"] == "St Brendan's University Hospital")
                  & (df["month_num"] == 14)).to_numpy()
    falls_lambda = falls_lambda * np.where(fall_boost, 3.6, 1.0)
    df["falls"] = rng.poisson(falls_lambda)
    df["pressure_ulcers"] = rng.poisson(df["los_days"].to_numpy() / 1000.0 * 0.62
                                        * (1 + 0.09 * df["prestroke_mrs"].to_numpy()))

    # ---- ward-care process reliability -----------------------------------
    df["immobile"] = df["nihss"] >= 6
    df["vte_ipc_24h"] = df["immobile"] & (rng.random(total) < 0.885)
    stays = ~df["died_inpatient"] & (df["los_days"] >= 3)
    df["mdt_goals_5d"] = stays & (rng.random(total) < 0.83)
    df["mood_screened"] = stays & (rng.random(total) < np.where(df["month_num"] >= 12, 0.84, 0.71))

    # ---- therapy need and first assessment -------------------------------
    p_pt = np.where(stays, _logistic(0.9 + 0.16 * df["nihss"].to_numpy()), 0.0)
    p_ot = np.where(stays, _logistic(0.6 + 0.15 * df["nihss"].to_numpy()), 0.0)
    p_slt = np.where(stays, _logistic(-1.5 + 0.16 * df["nihss"].to_numpy()
                                      + 1.6 * df["dysphagia"].to_numpy()), 0.0)
    df["pt_needed"] = rng.random(total) < p_pt
    df["ot_needed"] = rng.random(total) < p_ot
    df["slt_needed"] = rng.random(total) < p_slt

    # SIGNAL 3: Lakeview PT vacancies, Jun 2025 (21) to Feb 2026 (29).
    lakeview_gap = ((df["site"] == "Lakeview General")
                    & df["month_num"].between(21, 29)).to_numpy()
    df["staffing_gap"] = lakeview_gap

    for disc, needed, base_h in (("pt", "pt_needed", 26.0),
                                 ("ot", "ot_needed", 34.0),
                                 ("slt", "slt_needed", 30.0)):
        med = np.full(total, base_h)
        med += np.where(df["arrival_datetime"].dt.dayofweek >= 4, 12.0, 0.0)
        if disc == "pt":
            med += np.where(lakeview_gap, 16.0, 0.0)
        if disc == "slt":
            med -= np.where(df["dysphagia"].to_numpy(), 14.0, 0.0)
        hrs = _lognormal(rng, 1.0, 0.55, total) * med
        df[f"{disc}_assess_hours"] = np.where(df[needed], np.round(hrs, 1), np.nan)

    # ---- discharge and outcome -------------------------------------------
    surv = ~df["died_inpatient"]
    lp_home = (2.4 - 0.108 * df["nihss"] - 0.030 * (df["age"] - 74)
               - 0.62 * df["prestroke_mrs"] - 0.55 * df["lives_alone"].astype(int))
    home = surv & df["admitted_from_home"] & (rng.random(total) < _logistic(lp_home.to_numpy()))
    dest = np.where(df["died_inpatient"], "Died",
                    np.where(home, "Usual residence",
                             np.where(rng.random(total) < 0.55, "Inpatient rehabilitation",
                                      np.where(rng.random(total) < 0.5, "Nursing home", "Other"))))
    df["discharge_destination"] = dest

    esd_eligible = home & (df["nihss"] <= 15) & (df["prestroke_mrs"] <= 3)
    df["esd_eligible"] = esd_eligible
    esd_p = np.where(df["site"] == "Northbay Regional", 0.58, 0.79)
    esd_p = esd_p * np.where(df["month_num"] >= 14, 1.08, 1.0)
    df["esd_referral"] = esd_eligible & (rng.random(total) < np.clip(esd_p, 0, 1))

    df["af_anticoagulated"] = df["af"] & surv & (rng.random(total) < 0.905)
    df["antiplatelet"] = ischaemic & surv & (rng.random(total) < 0.965)

    # mRS at discharge: ordinal, driven by severity, age, prior function,
    # stroke type and reperfusion. Deaths are mRS 6 by definition.
    lp_mrs = (-1.0 + 0.148 * df["nihss"] + 0.030 * (df["age"] - 74)
              + 0.70 * df["prestroke_mrs"] + 0.35 * (~ischaemic).astype(int)
              - 0.45 * df["thrombectomy"].astype(int)
              - 0.22 * df["thrombolysed"].astype(int))
    latent = lp_mrs.to_numpy() + rng.normal(0, 1.25, total)
    # Cut points chosen so that roughly 42% of the whole cohort is mRS 0-2
    # at discharge, with a long tail through 3-5. Discharge mRS is worse
    # than 90-day mRS -- many patients leave still dependent and improve
    # afterwards -- so this should not be read against trial endpoints.
    cuts = np.array([-1.0, -0.12, 0.68, 1.5, 2.55])
    mrs = np.searchsorted(cuts, latent)
    df["mrs_discharge"] = np.where(df["died_inpatient"], 6, np.clip(mrs, 0, 5))

    lp_readmit = -2.55 + 0.030 * df["nihss"] + 0.018 * (df["age"] - 74) \
        + 0.30 * df["lives_alone"].astype(int) - 0.25 * df["esd_referral"].astype(int)
    df["readmitted_30d"] = surv & (rng.random(total) < _logistic(lp_readmit.to_numpy()))

    df = df.sort_values("arrival_datetime").reset_index(drop=True)

    # ---------------------------------------------------------------------
    # Therapy sessions: one row per patient-day-discipline that was
    # *applicable*, whether or not therapy happened. Building it this way
    # (rather than only recording delivered sessions) is what makes the
    # percentage-of-days and 45-minute indicators computable at all --
    # you cannot recover a denominator of opportunity from a table of
    # attendances.
    # ---------------------------------------------------------------------
    sessions = _generate_sessions(df, rng)
    therapists = _generate_therapists(rng)
    staffing = _generate_staffing(df, rng)

    df = _inject_recording_defects(df, rng)

    return {"admissions": df, "sessions": sessions,
            "therapists": therapists, "staffing": staffing}


def _inject_recording_defects(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Corrupt the *recorded* values, after the clinical truth is settled.

    This is the right order: the patient's actual course generated their
    outcomes; what the extract contains is a lossy transcription of it.
    Injecting defects earlier would let a data-entry error change a
    patient's mortality, which is not how reality works and would make
    the data-quality page an artefact of its own demonstration.

    Four defects, all of them things you will meet in a real extract:

    D1. **Digit preference.** About 35% of door-to-needle times are
        recorded to the nearest 5 minutes, because someone read a wall
        clock rather than the pump timestamp. Barely moves the median,
        but it compresses the variance, which narrows control limits,
        which manufactures special-cause signals out of nothing.

    D2. **Sentinel leakage.** A handful of NIHSS values arrive as 99 —
        a "not recorded" code from an upstream system that nobody
        mapped to null. The classic silent corruption: it is numeric,
        so it passes every type check and quietly inflates every mean
        it touches.

    D3. **Clock errors.** A few negative door-to-imaging times, from
        arrival and scan timestamps recorded against different dates.

    D4. **A feed break.** Northbay Regional stops capturing SLT
        assessment times for three months in early 2025 during a system
        migration. This is the important one: it will present itself as
        a sudden collapse in SLT access, and the only way to tell the
        difference is to look at completeness beside the indicator.
    """
    d = df.copy()
    n = len(d)

    # D1 -- digit preference on door-to-needle
    lysed = d["door_to_needle_min"].notna().to_numpy()
    round5 = lysed & (rng.random(n) < 0.35)
    d.loc[round5, "door_to_needle_min"] = (
        (d.loc[round5, "door_to_needle_min"] / 5).round() * 5)

    # D2 -- "not recorded" sentinel leaking through as a number
    sentinel = rng.random(n) < 0.003
    d.loc[sentinel, "nihss"] = 99

    # D3 -- timestamps recorded against the wrong date
    clock = rng.random(n) < 0.0018
    d.loc[clock, "door_to_ct_min"] = -d.loc[clock, "door_to_ct_min"].abs()

    # D4 -- three months with no SLT assessment times captured at one site
    feed_break = ((d["site"] == "Northbay Regional")
                  & d["month"].between("2025-02-01", "2025-04-30"))
    d.loc[feed_break, "slt_assess_hours"] = np.nan

    return d


# ---------------------------------------------------------------------------
def _generate_sessions(adm: pd.DataFrame, rng) -> pd.DataFrame:
    frames = []
    cfg = {
        "PT": dict(need="pt_needed", assess="pt_assess_hours", mins=39, sd=15,
                   wd=0.83, we=0.20),
        "OT": dict(need="ot_needed", assess="ot_assess_hours", mins=37, sd=16,
                   wd=0.78, we=0.09),
        "SLT": dict(need="slt_needed", assess="slt_assess_hours", mins=31, sd=13,
                    wd=0.71, we=0.14),
    }

    for disc, c in cfg.items():
        sub = adm[adm[c["need"]].to_numpy()].copy()
        if sub.empty:
            continue
        start_day = np.floor(sub[c["assess"]].to_numpy() / 24.0).astype(int)
        n_days = np.clip(sub["los_days"].to_numpy() - start_day, 0, 90).astype(int)
        keep = n_days > 0
        sub, start_day, n_days = sub[keep], start_day[keep], n_days[keep]
        if sub.empty:
            continue

        idx = np.repeat(np.arange(len(sub)), n_days)
        day_no = np.concatenate([np.arange(d) for d in n_days]) + np.repeat(start_day, n_days)
        adm_id = sub["admission_id"].to_numpy()[idx]
        site = sub["site"].to_numpy()[idx]
        date = (sub["arrival_datetime"].to_numpy()[idx]
                + pd.to_timedelta(day_no, "D")).astype("datetime64[ns]")
        date = pd.Series(date).dt.normalize()
        dow = date.dt.dayofweek.to_numpy()
        weekend = dow >= 5

        gap = sub["staffing_gap"].to_numpy()[idx] & (disc == "PT")
        winter = pd.Series(date).dt.month.isin([12, 1, 2]).to_numpy()

        p_attend = np.where(weekend, c["we"], c["wd"]).astype(float)
        p_attend *= np.where(gap, 0.62, 1.0)          # SIGNAL 3
        p_attend *= np.where(winter, 0.94, 1.0)       # SIGNAL 7
        # Therapy tails off as patients approach discharge and plateau.
        rel = day_no / np.maximum(np.repeat(sub["los_days"].to_numpy(), n_days), 1)
        p_attend *= np.clip(1.08 - 0.28 * rel, 0.35, 1.0)
        attended = rng.random(len(idx)) < np.clip(p_attend, 0, 1)

        n_sessions = np.where(attended, 1 + (rng.random(len(idx)) < 0.24).astype(int), 0)
        mins = np.round(rng.normal(c["mins"], c["sd"], len(idx)))
        mins = np.clip(mins, 8, 95) * n_sessions
        mins = np.where(weekend & attended, np.round(mins * 0.72), mins)
        mins = np.where(winter & attended, np.round(mins * 0.95), mins)

        # Missed-session reasons. During the staffing gap, "no therapist
        # available" roughly doubles its share -- this is the trace that
        # separates a capacity problem from a clinical one, and it is the
        # single most useful field a therapy service can start recording.
        base_p = np.array([0.30, 0.24, 0.16, 0.20, 0.10])
        gap_p = np.array([0.20, 0.53, 0.11, 0.11, 0.05])
        we_p = np.array([0.18, 0.62, 0.08, 0.07, 0.05])
        reason = np.full(len(idx), "", dtype=object)
        miss = ~attended
        for mask, probs in ((miss & gap, gap_p),
                            (miss & ~gap & weekend, we_p),
                            (miss & ~gap & ~weekend, base_p)):
            k = int(mask.sum())
            if k:
                reason[mask] = rng.choice(MISS_REASONS, size=k, p=probs)

        frames.append(pd.DataFrame({
            "admission_id": adm_id,
            "site": site,
            "discipline": disc,
            "date": date.to_numpy(),
            "day_of_stay": day_no,
            "weekend": weekend,
            "applicable": True,
            "attended": attended,
            "n_sessions": n_sessions,
            "minutes": mins.astype(int),
            "missed_reason": reason,
        }))

    out = pd.concat(frames, ignore_index=True)
    out["month"] = out["date"].dt.to_period("M").dt.to_timestamp()
    return out.sort_values(["date", "admission_id", "discipline"]).reset_index(drop=True)


def _generate_therapists(rng) -> pd.DataFrame:
    grades = ["Staff grade", "Senior", "Clinical specialist", "Therapy manager"]
    grade_p = [0.42, 0.40, 0.13, 0.05]
    rows = []
    tid = 0
    for _, s in SITES.iterrows():
        for disc, col in (("PT", "pt_wte"), ("OT", "ot_wte"), ("SLT", "slt_wte")):
            wte_total = float(s[col])
            remaining = wte_total
            while remaining > 0.05:
                w = float(min(remaining, rng.choice([1.0, 1.0, 1.0, 0.8, 0.6, 0.5])))
                rows.append({
                    "therapist_id": f"T{tid:03d}",
                    "site": s.site,
                    "discipline": disc,
                    "grade": rng.choice(grades, p=grade_p),
                    "wte": round(w, 2),
                })
                tid += 1
                remaining -= w
    return pd.DataFrame(rows)


def _generate_staffing(adm: pd.DataFrame, rng) -> pd.DataFrame:
    """Weekly funded vs available WTE by site and discipline.

    'Available' is funded minus vacancy minus leave and sickness. The gap
    between the two is the honest denominator for any capacity claim: a
    service is not short of therapists against its establishment, it is
    short against the establishment it can actually field this week.
    """
    weeks = pd.date_range(START, END, freq="W-MON")
    rows = []
    for _, s in SITES.iterrows():
        for disc, col in (("PT", "pt_wte"), ("OT", "ot_wte"), ("SLT", "slt_wte")):
            funded = float(s[col])
            for w in weeks:
                mnum = (w.year - START.year) * 12 + w.month - START.month
                vacancy = 0.0
                if s.site == "Lakeview General" and disc == "PT" and 21 <= mnum <= 29:
                    vacancy = 2.0                      # SIGNAL 3
                elif rng.random() < 0.16:
                    vacancy = float(rng.choice([0.5, 1.0]))
                # Leave and sickness: higher in summer and over Christmas.
                seasonal = 0.16 if w.month in (7, 8, 12) else 0.10
                absence = funded * rng.beta(2, 2) * seasonal
                available = max(funded - vacancy - absence, 0.0)
                rows.append({
                    "week": w, "site": s.site, "discipline": disc,
                    "wte_funded": funded,
                    "wte_vacant": vacancy,
                    "wte_absent": round(absence, 2),
                    "wte_available": round(available, 2),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    t0 = time.time()
    data = generate()
    for name, frame in data.items():
        print(f"{name:12s} {len(frame):>8,} rows  {list(frame.columns)[:6]}")
    print(f"generated in {time.time() - t0:.1f}s")
