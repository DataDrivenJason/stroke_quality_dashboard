"""Drill from a signal on a chart down to the patients behind it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui, viz

ui.setup("Patient explorer")
f = ui.sidebar()
data = ui.filtered(f)
adm = data["admissions"]
ses = data["sessions"]

st.title("Patient explorer")
st.caption("A control chart tells you *that* something changed. This page is how "
           "you find out *what* — by going back to the records that produced the "
           "signal, which is the step most dashboards leave out.")

# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------
st.subheader("Define a cohort")

c1, c2, c3, c4 = st.columns(4)
with c1:
    types = st.multiselect("Stroke type", sorted(adm["stroke_type"].unique()),
                           default=sorted(adm["stroke_type"].unique()), key="pe_type")
    treat = st.multiselect("Treatment", ["Thrombolysis", "Thrombectomy", "Neither"],
                           default=["Thrombolysis", "Thrombectomy", "Neither"],
                           key="pe_treat")
with c2:
    age_rng = st.slider("Age", int(adm["age"].min()), int(adm["age"].max()),
                        (int(adm["age"].min()), int(adm["age"].max())), key="pe_age")
    nihss_rng = st.slider(
        "NIHSS", 0, 42, (0, 42), key="pe_nihss",
        help="The scale runs 0–42. Records holding a value outside it are "
             "excluded by this filter and counted below, rather than being "
             "silently dropped.")
    n_bad_nihss = int((~adm["nihss"].between(0, 42)).sum())
with c3:
    dest = st.multiselect("Discharge destination",
                          sorted(adm["discharge_destination"].unique()),
                          default=sorted(adm["discharge_destination"].unique()),
                          key="pe_dest")
    mrs_rng = st.slider("mRS at discharge", 0, 6, (0, 6), key="pe_mrs")
with c4:
    breaches = st.multiselect(
        "Pathway breaches only",
        ["Imaging beyond 1 hour", "Needle beyond 60 minutes",
         "Stroke unit beyond 4 hours", "Swallow screen beyond 4 hours",
         "PT assessment beyond 72 hours", "OT assessment beyond 72 hours",
         "SLT assessment beyond 72 hours"], key="pe_breach")
    complications = st.multiselect("Complications",
                                   ["Pneumonia", "Any fall", "Pressure ulcer"],
                                   key="pe_comp")

m = (adm["stroke_type"].isin(types)
     & adm["age"].between(*age_rng)
     & adm["nihss"].between(*nihss_rng)
     & adm["discharge_destination"].isin(dest)
     & adm["mrs_discharge"].between(*mrs_rng))

treat_mask = pd.Series(False, index=adm.index)
if "Thrombolysis" in treat:
    treat_mask |= adm["thrombolysed"]
if "Thrombectomy" in treat:
    treat_mask |= adm["thrombectomy"]
if "Neither" in treat:
    treat_mask |= ~(adm["thrombolysed"] | adm["thrombectomy"])
m &= treat_mask

breach_map = {
    "Imaging beyond 1 hour": adm["door_to_ct_min"] > 60,
    "Needle beyond 60 minutes": adm["thrombolysed"] & (adm["door_to_needle_min"] > 60),
    "Stroke unit beyond 4 hours": adm["time_to_su_hours"] > 4,
    "Swallow screen beyond 4 hours": adm["swallow_screen_hours"] > 4,
    "PT assessment beyond 72 hours": adm["pt_needed"] & (adm["pt_assess_hours"] > 72),
    "OT assessment beyond 72 hours": adm["ot_needed"] & (adm["ot_assess_hours"] > 72),
    "SLT assessment beyond 72 hours": adm["slt_needed"] & (adm["slt_assess_hours"] > 72),
}
for b in breaches:
    m &= breach_map[b].fillna(False)

comp_map = {"Pneumonia": adm["hap"], "Any fall": adm["falls"] > 0,
            "Pressure ulcer": adm["pressure_ulcers"] > 0}
for cpx in complications:
    m &= comp_map[cpx].fillna(False)

cohort = adm[m]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Patients", f"{len(cohort):,}")
k2.metric("Share of selection", f"{100 * len(cohort) / max(len(adm), 1):.1f}%")
k3.metric("Median LOS", f"{cohort['los_days'].median():.0f} d" if len(cohort) else "—")
k4.metric("Died in hospital",
          f"{100 * cohort['died_inpatient'].mean():.1f}%" if len(cohort) else "—")
k5.metric("Home at discharge",
          f"{100 * (cohort['discharge_destination'] == 'Usual residence').mean():.1f}%"
          if len(cohort) else "—")

if len(cohort) == 0:
    st.info("No patients match this cohort. Loosen a filter.")
    st.stop()

if n_bad_nihss:
    st.caption(
        f"⚠ {n_bad_nihss:,} record(s) hold an NIHSS outside the 0–42 scale — a "
        f"'not recorded' sentinel leaking through from an upstream system. They "
        f"are excluded by the severity filter above. This is exactly the kind of "
        f"silent corruption the Data Quality page exists to surface: the value is "
        f"numeric, so it passes every type check, and it inflates every mean it "
        f"touches.")

st.markdown(
    '<div class="sqi-note"><b>A caution on cohorts built from breaches.</b> '
    'Selecting on a breach and then comparing outcomes with everyone else is '
    'a case-control design assembled after the fact, and it will confidently '
    'reproduce whatever confounding drove the breach in the first place — '
    'out-of-hours arrivals breach more and are sicker. Use this to read notes '
    'and find themes, not to estimate the effect of the breach.</div>',
    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cohort table
# ---------------------------------------------------------------------------
st.divider()
st.subheader("The records")

show_cols = ["admission_id", "site", "arrival_datetime", "age", "sex", "stroke_type",
             "nihss", "prestroke_mrs", "door_to_ct_min", "thrombolysed",
             "door_to_needle_min", "thrombectomy", "time_to_su_hours",
             "swallow_screen_hours", "pt_assess_hours", "ot_assess_hours",
             "slt_assess_hours", "los_days", "discharge_destination", "mrs_discharge"]
disp = cohort[show_cols].copy()
disp["arrival_datetime"] = disp["arrival_datetime"].dt.strftime("%Y-%m-%d %H:%M")
st.dataframe(disp.sort_values("arrival_datetime", ascending=False).round(1),
             hide_index=True, width="stretch", height=320)
st.download_button("Download this cohort as CSV", cohort.to_csv(index=False).encode(),
                   file_name="stroke_cohort.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# One patient
# ---------------------------------------------------------------------------
st.divider()
st.subheader("One patient's pathway")

pid = st.selectbox("Admission", cohort["admission_id"].tolist(), key="pe_patient")
pt = cohort[cohort["admission_id"] == pid].iloc[0]

p1, p2 = st.columns([2, 3], gap="large")
with p1:
    st.markdown(
        f"**{pt['admission_id']}** · {pt['site']}  \n"
        f"{pt['age']}-year-old {str(pt['sex']).lower()}, {str(pt['stroke_type']).lower()}  \n"
        f"NIHSS **{pt['nihss']}** · pre-stroke mRS **{pt['prestroke_mrs']}**  \n"
        f"Arrived {pt['arrival_datetime']:%d %b %Y, %H:%M}"
        f"{' (out of hours)' if pt['out_of_hours'] else ''}  \n"
        f"Length of stay **{pt['los_days']} days** · discharged to "
        f"**{str(pt['discharge_destination']).lower()}** · mRS **{pt['mrs_discharge']}**")

    events = []
    for label, minutes in (("Symptom onset", -pt["onset_to_door_min"] if pd.notna(pt["onset_to_door_min"]) else None),
                           ("Arrival (clock start)", 0.0),
                           ("Brain imaging", pt["door_to_ct_min"]),
                           ("Thrombolysis", pt["door_to_needle_min"] if pt["thrombolysed"] else None),
                           ("Arterial puncture", pt["door_to_puncture_min"] if pt["thrombectomy"] else None),
                           ("Swallow screen", pt["swallow_screen_hours"] * 60),
                           ("Stroke unit", pt["time_to_su_hours"] * 60),
                           ("PT assessment", pt["pt_assess_hours"] * 60 if pt["pt_needed"] else None),
                           ("OT assessment", pt["ot_assess_hours"] * 60 if pt["ot_needed"] else None),
                           ("SLT assessment", pt["slt_assess_hours"] * 60 if pt["slt_needed"] else None)):
        if minutes is not None and pd.notna(minutes):
            events.append({"Event": label, "Minutes from clock start": float(minutes),
                           "Hours": float(minutes) / 60})
    ev = pd.DataFrame(events).sort_values("Minutes from clock start")
    st.dataframe(ev.round(1), hide_index=True, width="stretch")

with p2:
    p = viz.palette()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ev["Hours"], y=ev["Event"], mode="markers",
        marker=dict(size=13, color=p["series"][0],
                    line=dict(color=p["surface"], width=2)),
        hovertemplate="%{x:.1f} hours from clock start<extra>%{y}</extra>"))
    for _, r in ev.iterrows():
        fig.add_shape(type="line", x0=0, x1=r["Hours"], y0=r["Event"], y1=r["Event"],
                      line=dict(color=p["grid"], width=1.5))
    fig.add_vline(x=0, line=dict(color=p["axis"], width=1.4))
    lay = viz._layout(p, 330, "Pathway timing", xtitle="Hours from clock start")
    lay["showlegend"] = False
    lay["hovermode"] = "closest"
    lay["yaxis"]["gridcolor"] = "rgba(0,0,0,0)"
    lay["yaxis"]["autorange"] = "reversed"
    fig.update_layout(**lay)
    viz.show(fig)

# ---- therapy record for this patient --------------------------------------
psess = ses[ses["admission_id"] == pid].copy()
if not psess.empty:
    st.markdown("##### Therapy record")
    pivot = psess.pivot_table(index="discipline", columns="day_of_stay",
                              values="minutes", aggfunc="sum", observed=True)
    if not pivot.empty:
        viz.show(viz.heatmap(pivot.fillna(0), height=190,
                             title="Attended minutes by day of stay", unit="min"))
    reasons = psess[(~psess["attended"]) & (psess["missed_reason"] != "")]
    tot = psess.groupby("discipline", observed=True).agg(
        applicable=("applicable", "sum"), attended=("attended", "sum"),
        minutes=("minutes", "sum")).reset_index()
    tot["min_per_applicable_day"] = tot["minutes"] / tot["applicable"].clip(lower=1)
    tot["% of days delivered"] = 100 * tot["attended"] / tot["applicable"].clip(lower=1)
    t1, t2 = st.columns([2, 3], gap="large")
    with t1:
        st.dataframe(tot.round(1), hide_index=True, width="stretch")
    with t2:
        if not reasons.empty:
            rc = reasons.groupby(["discipline", "missed_reason"], observed=True).size() \
                .rename("days").reset_index()
            st.dataframe(rc, hide_index=True, width="stretch")
        else:
            st.caption("No missed sessions recorded for this admission.")

# ---------------------------------------------------------------------------
# Cohort outcomes
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Cohort against the rest")

rest = adm[~adm.index.isin(cohort.index)]
rows = []
for label, fn in (
    ("n", lambda d: f"{len(d):,}"),
    ("Median age", lambda d: f"{d['age'].median():.0f}"),
    ("Median NIHSS", lambda d: f"{d['nihss'].median():.0f}"),
    ("Haemorrhagic", lambda d: f"{100 * (d['stroke_type'] != 'Ischaemic').mean():.1f}%"),
    ("Median door-to-imaging (min)", lambda d: f"{d['door_to_ct_min'].median():.0f}"),
    ("Median time to stroke unit (h)", lambda d: f"{d['time_to_su_hours'].median():.1f}"),
    ("Median LOS (days)", lambda d: f"{d['los_days'].median():.0f}"),
    ("Pneumonia", lambda d: f"{100 * d['hap'].mean():.1f}%"),
    ("Died in hospital", lambda d: f"{100 * d['died_inpatient'].mean():.1f}%"),
    ("mRS 0–2 at discharge", lambda d: f"{100 * (d['mrs_discharge'] <= 2).mean():.1f}%"),
):
    rows.append({"Measure": label, "Cohort": fn(cohort),
                 "Everyone else": fn(rest) if len(rest) else "—"})

o1, o2 = st.columns([2, 3], gap="large")
with o1:
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
with o2:
    both = pd.concat([cohort.assign(_grp="Cohort"), rest.assign(_grp="Everyone else")])
    from core.metrics import mrs_distribution
    viz.show(viz.stacked_shift(mrs_distribution(both, "_grp"), "_grp", height=210,
                               title="Disability at discharge"))
    st.markdown(
        "**This table is descriptive, and only descriptive.** The cohort was "
        "defined by hand, usually on characteristics that also predict the "
        "outcomes underneath it. Every row here is confounded by construction. "
        "It earns its place by generating hypotheses and by pointing at records "
        "worth reading — not by supporting a comparison.")
