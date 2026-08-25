"""Therapy dose and intensity for PT, OT and SLT."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import mdc, metrics, ui, viz
from core.standards import metric

ui.setup("Therapy dose")
f = ui.sidebar()
data = ui.filtered(f)
adm = data["admissions"]
ses = data["sessions"]
tbl = ui.period_table(f)

DISC = {"PT": "Physiotherapy", "OT": "Occupational therapy",
        "SLT": "Speech & language therapy"}

st.title("Therapy dose and intensity")
st.caption("Assessment-within-72-hours says contact was made. Only minutes per day, "
           "and the proportion of days they land on, say whether a therapeutic dose "
           "followed.")

# ---------------------------------------------------------------------------
# The dose cascade
# ---------------------------------------------------------------------------
st.subheader("Where the dose is lost")

pl = metrics.therapy_patient_level(ses)
cascade_rows = []
for d in ("PT", "OT", "SLT"):
    sub = pl[pl["discipline"] == d]
    if sub.empty:
        continue
    applicable = float(sub["days_applicable"].sum())
    delivered = float(sub["days_with_therapy"].sum())
    at45 = float(sub["days_45min"].sum())
    cascade_rows.append({
        "Discipline": DISC[d],
        "Patient-days applicable": applicable,
        "Days with any therapy": delivered,
        "Days meeting 45 minutes": at45,
        "% of days delivered": 100 * delivered / max(applicable, 1),
        "% of days at 45 min": 100 * at45 / max(applicable, 1),
        "Median min/day": float(sub["minutes_per_day"].median()),
    })
cascade = pd.DataFrame(cascade_rows)

c1, c2 = st.columns([3, 2], gap="large")
with c1:
    p = viz.palette()
    fig = go.Figure()
    stages = ["Patient-days applicable", "Days with any therapy", "Days meeting 45 minutes"]
    shades = ["#86b6ef", "#3987e5", "#184f95"] if p is viz.LIGHT else ["#9ec5f4", "#3987e5", "#184f95"]
    for i, stage in enumerate(stages):
        fig.add_trace(go.Bar(
            x=cascade["Discipline"], y=cascade[stage], name=stage,
            marker=dict(color=shades[i], line=dict(color=p["surface"], width=2)),
            hovertemplate="<b>%{y:,.0f}</b> patient-days<extra>%{x} · " + stage + "</extra>"))
    lay = viz._layout(p, 340, "", ytitle="Patient-days")
    lay["barmode"] = "group"
    lay["bargap"] = 0.28
    lay["bargroupgap"] = 0.06
    fig.update_layout(**lay)
    viz.show(fig)
with c2:
    st.markdown(
        "**The denominator is the whole argument.** These bars start from "
        "*applicable* patient-days — every day on which therapy was indicated, "
        "whether or not any happened. A service that records only delivered "
        "sessions cannot draw this chart at all, and so can never measure its own "
        "reliability. That is usually discovered at the moment someone asks it to.\n\n"
        "The gap between the first and second bar is **frequency** — days lost "
        "entirely. The gap between the second and third is **intensity** — days "
        "where therapy happened but fell short of a therapeutic dose. They have "
        "completely different causes and completely different fixes, and a single "
        "'minutes per day' average hides both.")

ui.df(
    cascade[["Discipline", "Patient-days applicable", "Days with any therapy",
             "Days meeting 45 minutes", "% of days delivered",
             "% of days at 45 min", "Median min/day"]].round(1),
    hide_index=True,
    column_config={
        "Patient-days applicable": st.column_config.NumberColumn(format="%d"),
        "Days with any therapy": st.column_config.NumberColumn(format="%d"),
        "Days meeting 45 minutes": st.column_config.NumberColumn(format="%d"),
        "% of days delivered": st.column_config.NumberColumn(format="%.1f%%"),
        "% of days at 45 min": st.column_config.NumberColumn(format="%.1f%%"),
        "Median min/day": st.column_config.NumberColumn(format="%.0f min"),
    })

# ---------------------------------------------------------------------------
# Control charts by discipline
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Is the dose changing?")

dc1, dc2 = st.columns([1, 3])
with dc1:
    disc = st.radio("Discipline", list(DISC.keys()),
                    format_func=lambda d: DISC[d], key="td_disc")
    measure = st.radio(
        "Measure", ["Days with therapy", "Minutes per day", "Days at 45 minutes",
                    "Assessment within 72 hours"], key="td_measure")

key = {
    ("Days with therapy",): f"{disc.lower()}_pct_days",
    ("Minutes per day",): f"{disc.lower()}_min_per_day",
    ("Days at 45 minutes",): f"{disc.lower()}_45min",
    ("Assessment within 72 hours",): f"{disc.lower()}_72h",
}[(measure,)]

res = ui.series(tbl, key, f)
spec, thr = metric(key, f.standard, f.overrides)
v = mdc.verdict(res)

with dc2:
    viz.show(viz.control_chart(res, title=f"{DISC[disc]} — {spec.label}", height=360))
ui.verdict_banner(v, res)
ui.methods_block(res, key, f)

if res.meta.get("overdispersed"):
    st.markdown(
        f'<div class="sqi-note"><b>This is the indicator that needs the Laney '
        f'correction most.</b> The denominator here is patient-<i>days</i> — '
        f'{res.frame["denominator"].mean():,.0f} of them in a typical period. '
        f'Binomial limits on a denominator that large sit within a fraction of a '
        f'percentage point of the centre line, and every period signals. Measured '
        f'σz here is {res.meta["sigma_z"]:.2f}, so the limits are that much wider. '
        f'The overdispersion itself is real information: patient-days are not '
        f'independent trials, because the same patient contributes many of them '
        f'and one understaffed week contributes hundreds.</div>',
        unsafe_allow_html=True)

ui.table_view(res)

# ---------------------------------------------------------------------------
# The weekend
# ---------------------------------------------------------------------------
st.divider()
st.subheader("The weekend")

s = ses.copy()
s["dow"] = pd.Categorical(s["date"].dt.day_name().str[:3],
                          categories=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                          ordered=True)
by_dow = (s.groupby(["dow", "discipline"], observed=True)
          .agg(applicable=("applicable", "sum"), attended=("attended", "sum"),
               minutes=("minutes", "sum")).reset_index())
by_dow["pct_days"] = 100 * by_dow["attended"] / by_dow["applicable"].clip(lower=1)
by_dow["min_per_day"] = by_dow["minutes"] / by_dow["applicable"].clip(lower=1)

wc1, wc2 = st.columns([3, 2], gap="large")
with wc1:
    pivot = by_dow.pivot(index="discipline", columns="dow", values="min_per_day")
    pivot.index = [DISC[i] for i in pivot.index]
    viz.show(viz.heatmap(pivot.round(1), height=250,
                         title="Mean therapy minutes per applicable day",
                         unit="min/day"))
with wc2:
    wk = by_dow.assign(weekend=by_dow["dow"].isin(["Sat", "Sun"]))
    summ = (wk.groupby(["discipline", "weekend"], observed=True)
            .agg(minutes=("minutes", "sum"), applicable=("applicable", "sum"))
            .reset_index())
    summ["min_per_day"] = summ["minutes"] / summ["applicable"].clip(lower=1)
    piv = summ.pivot(index="discipline", columns="weekend", values="min_per_day")
    piv.columns = ["Weekday", "Weekend"]
    piv["Weekend as % of weekday"] = 100 * piv["Weekend"] / piv["Weekday"]
    piv.index = [DISC[i] for i in piv.index]
    ui.df(piv.round(1))

    lost = float(by_dow.loc[by_dow["dow"].isin(["Sat", "Sun"]), "applicable"].sum())
    wkday_rate = float(
        by_dow.loc[~by_dow["dow"].isin(["Sat", "Sun"]), "minutes"].sum() /
        max(by_dow.loc[~by_dow["dow"].isin(["Sat", "Sun"]), "applicable"].sum(), 1))
    actual_we = float(by_dow.loc[by_dow["dow"].isin(["Sat", "Sun"]), "minutes"].sum())
    st.markdown(
        f'<div class="sqi-note"><b>Two sevenths of every stay.</b> If weekend days '
        f'were delivered at the weekday rate, this cohort would have received '
        f'roughly <b>{lost * wkday_rate - actual_we:,.0f}</b> additional therapy '
        f'minutes over the period — a {100 * (lost * wkday_rate - actual_we) / max(by_dow["minutes"].sum(), 1):.0f}% '
        f'increase in total dose, without changing anything that happens Monday to '
        f'Friday. That figure is the business case for seven-day working, and it '
        f'is the single largest structural lever on rehabilitation dose.<br><br>'
        f'The counter-argument worth taking seriously: weekend cover is often a '
        f'deliberate prioritisation — new admissions and dysphagia review — rather '
        f'than a gap. Whether that is what is happening here is visible in the '
        f'missed-session reasons on the Caseload page.</div>',
        unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Time to first assessment
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Time to first assessment")

ac1, ac2 = st.columns([3, 2], gap="large")
with ac1:
    p = viz.palette()
    fig = go.Figure()
    for i, d in enumerate(("PT", "OT", "SLT")):
        col = f"{d.lower()}_assess_hours"
        vals = pd.to_numeric(adm.loc[adm[f"{d.lower()}_needed"], col],
                             errors="coerce").dropna()
        if vals.empty:
            continue
        vals = vals.clip(upper=200)
        fig.add_trace(go.Violin(
            y=vals, name=DISC[d], side="positive", width=0.85,
            line=dict(color=p["series"][i], width=1.8),
            fillcolor=p["series"][i], opacity=0.30, points=False,
            meanline=dict(visible=True, color=p["series"][i], width=2),
            hovertemplate="%{y:.0f} hours<extra>" + DISC[d] + "</extra>"))
    fig.add_hline(y=72, line=dict(color=p["target"], width=1.8, dash="dot"),
                  annotation_text="72 hour standard",
                  annotation_font=dict(size=10.5, color=p["muted"]))
    lay = viz._layout(p, 340, "", ytitle="Hours from clock start")
    lay["hovermode"] = "closest"
    lay["showlegend"] = False
    fig.update_layout(**lay)
    viz.show(fig)
with ac2:
    rows = []
    for d in ("PT", "OT", "SLT"):
        need = adm[adm[f"{d.lower()}_needed"]]
        vals = pd.to_numeric(need[f"{d.lower()}_assess_hours"], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append({
            "Discipline": DISC[d],
            "Needing input": f"{len(need):,}",
            "% of admissions": f"{100 * len(need) / max(len(adm), 1):.0f}%",
            "Median (h)": f"{vals.median():.0f}",
            "Within 72h": f"{100 * (vals <= 72).mean():.1f}%",
        })
    ui.df(pd.DataFrame(rows), hide_index=True)
    st.markdown(
        "**Watch the denominator, not just the rate.** 'Needing input' is a "
        "clinical judgement recorded by the service being measured. A team under "
        "pressure can improve its 72-hour compliance simply by recording fewer "
        "patients as requiring input, and the indicator will applaud. The "
        "percentage-of-admissions column is the check: if compliance rises while "
        "that column falls, the improvement is in the coding.\n\n"
        "The distributions are also worth reading as shapes. A sharp cliff just "
        "before 72 hours is the fingerprint of a threshold being managed rather "
        "than a process being improved.")

# ---------------------------------------------------------------------------
# Dose and outcome
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Dose and outcome — and why the obvious analysis is wrong")

wide = pl.pivot(index="admission_id", columns="discipline", values="minutes_per_day")
wide.columns = [f"{c}_mpd" for c in wide.columns]
joined = adm.merge(wide.reset_index(), on="admission_id", how="inner")
joined = joined[joined["died_inpatient"] == False]  # noqa: E712
joined["mrs_gain"] = joined["prestroke_mrs"] - joined["mrs_discharge"]
joined["nihss_band"] = pd.cut(joined["nihss"], [-1, 4, 9, 15, 42],
                              labels=["Minor (0–4)", "Moderate (5–9)",
                                      "Moderate-severe (10–15)", "Severe (16+)"])

dose_col = f"{disc}_mpd"
if dose_col in joined.columns:
    sc1, sc2 = st.columns([3, 2], gap="large")
    with sc1:
        stratify = st.checkbox("Stratify by stroke severity", value=True, key="td_strat")
        p = viz.palette()
        fig = go.Figure()
        grp = joined.dropna(subset=[dose_col, "mrs_discharge"])
        if stratify:
            for i, band in enumerate(grp["nihss_band"].cat.categories):
                sub = grp[grp["nihss_band"] == band]
                if len(sub) < 20:
                    continue
                binned = sub.groupby(pd.cut(sub[dose_col], 6), observed=True).agg(
                    dose=(dose_col, "median"), mrs=("mrs_discharge", "mean"),
                    n=("admission_id", "count")).dropna()
                fig.add_trace(go.Scatter(
                    x=binned["dose"], y=binned["mrs"], mode="lines+markers", name=str(band),
                    line=dict(color=p["series"][i], width=2),
                    marker=dict(size=np.clip(binned["n"] / 12, 6, 18),
                                line=dict(color=p["surface"], width=1.5)),
                    customdata=binned["n"],
                    hovertemplate="%{y:.2f} mean mRS<br>%{customdata:,.0f} patients"
                                  "<extra>" + str(band) + "</extra>"))
        else:
            binned = grp.groupby(pd.cut(grp[dose_col], 10), observed=True).agg(
                dose=(dose_col, "median"), mrs=("mrs_discharge", "mean"),
                n=("admission_id", "count")).dropna()
            fig.add_trace(go.Scatter(
                x=binned["dose"], y=binned["mrs"], mode="lines+markers",
                name="All patients", line=dict(color=p["series"][0], width=2),
                marker=dict(size=np.clip(binned["n"] / 25, 7, 20),
                            line=dict(color=p["surface"], width=1.5)),
                customdata=binned["n"],
                hovertemplate="%{y:.2f} mean mRS<br>%{customdata:,.0f} patients<extra></extra>"))
        lay = viz._layout(p, 360, f"{DISC[disc]} dose against disability at discharge",
                          ytitle="Mean mRS at discharge (lower is better)",
                          xtitle="Median minutes per applicable day")
        lay["hovermode"] = "closest"
        fig.update_layout(**lay)
        viz.show(fig)
    with sc2:
        pooled = joined[[dose_col, "mrs_discharge"]].dropna()
        r_pooled = pooled.corr().iloc[0, 1] if len(pooled) > 30 else np.nan
        within = []
        for band in joined["nihss_band"].cat.categories:
            sub = joined[joined["nihss_band"] == band][[dose_col, "mrs_discharge"]].dropna()
            if len(sub) > 30:
                within.append(sub.corr().iloc[0, 1])
        st.metric("Pooled correlation, dose vs mRS", f"{r_pooled:+.2f}")
        if within:
            st.metric("Mean within-severity correlation", f"{np.mean(within):+.2f}")
        st.markdown(
            "**Confounding by indication.** Sicker patients receive more therapy "
            "*and* have worse outcomes, so the pooled association between dose and "
            "disability comes out positive — apparently, more therapy makes people "
            "worse. Severity is driving both variables, and the pooled correlation "
            "is measuring severity, not therapy.\n\n"
            "Stratifying by NIHSS band pulls the two apart: within a band, patients "
            "are far more comparable and the relationship is closer to the causal "
            "one. That the pooled and within-band correlations can point in "
            "opposite directions is Simpson's paradox, and it is the reason no dose "
            "claim should ever be made from an unadjusted scatter.\n\n"
            "Even stratified, this is observational. Therapy allocation responds to "
            "how the patient is progressing, and a patient recovering well is "
            "discharged sooner and receives less total therapy. Doing this properly "
            "needs the time-varying nature of both dose and prognosis handled "
            "explicitly — marginal structural models or a target-trial emulation — "
            "not a cross-sectional regression with severity thrown in as a "
            "covariate.")
