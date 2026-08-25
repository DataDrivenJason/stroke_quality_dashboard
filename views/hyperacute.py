"""Hyperacute pathway: the clock from door to treatment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import mdc, metrics, spc, ui, viz
from core.standards import METRICS, metric, metrics_in_domain

ui.setup("Hyperacute pathway")
f = ui.sidebar()
data = ui.filtered(f)
adm = data["admissions"]
tbl = ui.period_table(f)

st.title("Hyperacute pathway")
st.caption("Everything on this page is measured from **clock start** — arrival at "
           "the first hospital, or symptom onset for in-hospital strokes. Services "
           "that measure from 'first seen by the stroke team' post better numbers "
           "and treat patients no faster.")

# ---------------------------------------------------------------------------
# The pathway as a sequence of medians
# ---------------------------------------------------------------------------
st.subheader("Where the time goes")

lysed = adm[adm["thrombolysed"]]
stages = [
    ("Onset → arrival", adm.loc[adm["onset_known"], "onset_to_door_min"], "Ambulance and public awareness"),
    ("Arrival → imaging", adm["door_to_ct_min"], "ED triage, portering, scanner access"),
    ("Arrival → needle", lysed["door_to_needle_min"], "The whole in-hospital decision"),
    ("Arrival → puncture", adm.loc[adm["thrombectomy"], "door_to_puncture_min"], "Includes inter-hospital transfer"),
    ("Arrival → stroke unit", adm["time_to_su_hours"] * 60, "Bed availability, not clinical decision"),
]
rows = []
for name, s, driver in stages:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        continue
    rows.append({"Stage": name, "n": len(s), "Median (min)": s.median(),
                 "75th centile": s.quantile(0.75), "90th centile": s.quantile(0.90),
                 "Main driver": driver})
stage_df = pd.DataFrame(rows)

c1, c2 = st.columns([3, 2], gap="large")
with c1:
    p = viz.palette()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=stage_df["Stage"], x=stage_df["Median (min)"], orientation="h", name="Median",
        marker=dict(color=p["series"][0], line=dict(color=p["surface"], width=2)),
        hovertemplate="<b>%{x:,.0f} min</b> median<extra>%{y}</extra>"))
    fig.add_trace(go.Scatter(
        y=stage_df["Stage"], x=stage_df["90th centile"], mode="markers", name="90th centile",
        marker=dict(color=p["series"][1], size=11, symbol="line-ns-open",
                    line=dict(width=3, color=p["series"][1])),
        hovertemplate="<b>%{x:,.0f} min</b> at the 90th centile<extra>%{y}</extra>"))
    lay = viz._layout(p, 320, "", xtitle="Minutes from the start of that stage")
    lay["yaxis"]["gridcolor"] = "rgba(0,0,0,0)"
    lay["yaxis"]["autorange"] = "reversed"
    lay["xaxis"]["showgrid"] = True
    lay["xaxis"]["gridcolor"] = p["grid"]
    lay["hovermode"] = "closest"
    fig.update_layout(**lay)
    viz.show(fig)
with c2:
    st.markdown(
        "**Read the gap between the median and the 90th centile, not the median "
        "alone.** A stage where the two are close is a reliable process. A stage "
        "where the 90th centile is three times the median has a tail — a subset "
        "of patients experiencing something quite different, usually out of "
        "hours, usually for a nameable reason.\n\n"
        "Improving a median moves everyone slightly. Removing a tail moves a few "
        "patients enormously, and is almost always the better return.")
    ui.df(stage_df[["Stage", "n", "Median (min)", "90th centile"]].round(0),
                 hide_index=True)

# ---------------------------------------------------------------------------
# Control chart for a selected hyperacute indicator
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Is it changing?")

hyper_keys = metrics_in_domain("Hyperacute pathway", f.standard)
key = ui.metric_picker(hyper_keys, f, "Indicator", key="hyper_metric", default="dtn_median")
res = ui.series(tbl, key, f)
spec, thr = metric(key, f.standard, f.overrides)
v = mdc.verdict(res)

left, right = st.columns([3, 1], gap="large")
with left:
    viz.show(viz.control_chart(res, title=spec.label, height=390))
    if res.chart_type == "xmr":
        viz.show(viz.mr_chart(res))
with right:
    ui.verdict_banner(v, res)
    cp = spc.suggest_phase_break(res.frame["value"])
    if cp and cp["worth_splitting"]:
        when = pd.to_datetime(res.frame["x"].iloc[cp["index"]])
        st.markdown(
            f'<div class="sqi-note"><b>Possible step change around '
            f'{when:%b %Y}.</b> Mean {cp["mean_before"]:.1f} → '
            f'{cp["mean_after"]:.1f} ({cp["shift"]:+.1f}). '
            f'Only split the limits here if you can name what changed and date '
            f'it independently of this chart.</div>', unsafe_allow_html=True)

ui.methods_block(res, key, f)
ui.table_view(res)

# ---------------------------------------------------------------------------
# Out-of-hours analysis
# ---------------------------------------------------------------------------
st.divider()
st.subheader("The out-of-hours tail")

metric_choice = st.radio(
    "Measure", ["Door-to-needle", "Door-to-imaging", "Time to stroke unit"],
    horizontal=True, key="ooh_measure")
col_map = {"Door-to-needle": ("door_to_needle_min", "minutes", adm["thrombolysed"]),
           "Door-to-imaging": ("door_to_ct_min", "minutes", pd.Series(True, index=adm.index)),
           "Time to stroke unit": ("time_to_su_hours", "hours", pd.Series(True, index=adm.index))}
col, unit, mask = col_map[metric_choice]

sub = adm[mask].copy()
sub["dow"] = sub["arrival_datetime"].dt.day_name().str[:3]
sub["hour_band"] = pd.cut(sub["arrival_datetime"].dt.hour,
                          bins=[-1, 3, 7, 11, 15, 19, 23],
                          labels=["00–04", "04–08", "08–12", "12–16", "16–20", "20–24"])
order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
pivot = (sub.pivot_table(index="dow", columns="hour_band", values=col,
                         aggfunc="median", observed=False)
         .reindex(order))
counts = (sub.pivot_table(index="dow", columns="hour_band", values=col,
                          aggfunc="count", observed=False).reindex(order))

hc1, hc2 = st.columns([3, 2], gap="large")
with hc1:
    viz.show(viz.heatmap(pivot.round(1), height=300,
                         title=f"Median {metric_choice.lower()} by arrival time",
                         unit=unit))
    st.caption("Cells are medians; cells built on very few patients are unstable — "
               "the counts are in the table beside this.")
with hc2:
    inh = sub.loc[~sub["out_of_hours"], col].dropna()
    ooh = sub.loc[sub["out_of_hours"], col].dropna()
    if len(inh) > 5 and len(ooh) > 5:
        d1, d2 = st.columns(2)
        d1.metric("In hours (median)", f"{inh.median():.1f} {unit}")
        d2.metric("Out of hours (median)", f"{ooh.median():.1f} {unit}",
                  delta=f"{ooh.median() - inh.median():+.1f}", delta_color="inverse")
        # Hodges-Lehmann shift: the median of all pairwise differences.
        # Robust, and it estimates a quantity people actually care about --
        # "how much longer, typically" -- rather than a difference in means.
        samp_a = inh.sample(min(len(inh), 400), random_state=1).to_numpy()
        samp_b = ooh.sample(min(len(ooh), 400), random_state=1).to_numpy()
        hl = float(np.median(samp_b[:, None] - samp_a[None, :]))
        st.markdown(
            f'<div class="sqi-note"><b>Hodges–Lehmann shift: {hl:+.1f} {unit}.</b> '
            f'The median of all pairwise differences between an out-of-hours and an '
            f'in-hours patient — a robust estimate of how much longer a typical '
            f'patient waits out of hours, unaffected by the extreme tail that would '
            f'drag a difference in means.</div>', unsafe_allow_html=True)
    ui.df(counts.fillna(0).astype(int))

# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------
st.divider()
st.subheader("The whole distribution, not just the threshold")

dc1, dc2 = st.columns([3, 2], gap="large")
with dc1:
    p = viz.palette()
    vals = pd.to_numeric(lysed["door_to_needle_min"], errors="coerce").dropna()
    if len(vals) > 10:
        fig = go.Figure(go.Histogram(
            x=vals, nbinsx=40,
            marker=dict(color=p["series"][0], line=dict(color=p["surface"], width=1.5)),
            hovertemplate="<b>%{y}</b> patients<extra>%{x} min</extra>"))
        fig.add_vline(x=60, line=dict(color=p["target"], width=1.8, dash="dot"),
                      annotation_text="60 min", annotation_font=dict(size=10.5))
        fig.add_vline(x=float(vals.median()), line=dict(color=p["ink2"], width=1.6),
                      annotation_text=f"median {vals.median():.0f}",
                      annotation_font=dict(size=10.5), annotation_position="top left")
        lay = viz._layout(p, 320, "Door-to-needle time", ytitle="Patients",
                          xtitle="Minutes from clock start")
        lay["showlegend"] = False
        lay["hovermode"] = "closest"
        fig.update_layout(**lay)
        viz.show(fig)
with dc2:
    st.markdown(
        "**Why the threshold indicator is not enough.** 'Percentage treated within "
        "60 minutes' cannot tell the difference between a service whose failures "
        "miss by five minutes and one whose failures miss by fifty. Both score the "
        "same, and only one is close to fixing.\n\n"
        "It also creates a real perverse incentive. Once a patient has passed 60 "
        "minutes they no longer affect the indicator, and the urgency that the "
        "evidence says should persist — benefit decays continuously, with no "
        "plateau — has no measurement behind it.\n\n"
        "The distribution shows both: where the bulk sits, and how long the tail "
        "runs past the threshold.")
    if len(vals) > 10:
        ui.df(pd.DataFrame({
            "Statistic": ["n", "Median", "75th centile", "90th centile",
                          "Within 60 min", "Beyond 90 min"],
            "Value": [f"{len(vals):,}", f"{vals.median():.0f} min",
                      f"{vals.quantile(.75):.0f} min", f"{vals.quantile(.90):.0f} min",
                      f"{100 * (vals <= 60).mean():.1f}%",
                      f"{100 * (vals > 90).mean():.1f}%"],
        }), hide_index=True)

# ---------------------------------------------------------------------------
# Site comparison
# ---------------------------------------------------------------------------
if adm["site"].nunique() > 1:
    st.divider()
    st.subheader("By site")
    site_tbl = ui.period_table(f, group_col="site")
    if METRICS[key].chart == "p":
        plot = site_tbl.assign(
            value=100 * site_tbl[METRICS[key].numerator] /
            site_tbl[METRICS[key].denominator].replace(0, np.nan))
    elif METRICS[key].chart == "xmr":
        plot = site_tbl.assign(value=site_tbl[METRICS[key].value_col])
    else:
        plot = site_tbl.assign(
            value=METRICS[key].multiplier * site_tbl[METRICS[key].count_col] /
            site_tbl[METRICS[key].exposure_col].replace(0, np.nan))
    viz.show(viz.multi_line(plot, "period", "value", "site", height=340,
                            title=spec.label, ytitle=spec.unit, target=thr.target))
    st.caption("Colour follows the site, not its current rank — deselecting a site "
               "in the legend never repaints the others. Individual site charts "
               "with their own limits are on the Control charts page.")
