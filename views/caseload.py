"""Caseload, demand and capacity for the therapy services."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import mdc, spc, ui, viz

ui.setup("Caseload & capacity")
f = ui.sidebar()
data = ui.filtered(f)
adm = data["admissions"]
ses = data["sessions"]
staffing = data["staffing"]
therapists = data["therapists"]

DISC = {"PT": "Physiotherapy", "OT": "Occupational therapy",
        "SLT": "Speech & language therapy"}

st.title("Caseload and capacity")
st.caption("The therapy manager's view: how many patients, how many therapists, "
           "and the arithmetic that connects them.")

# ---------------------------------------------------------------------------
# Capacity assumptions -- exposed, not buried
# ---------------------------------------------------------------------------
with st.expander("Capacity assumptions (change these — they drive every number below)",
                 expanded=False):
    a1, a2, a3 = st.columns(3)
    with a1:
        hours_week = st.number_input("Contracted hours per WTE per week", 30.0, 42.0,
                                     37.5, 0.5, key="cap_hours")
    with a2:
        contact_ratio = st.slider(
            "Clinical contact ratio", 0.35, 0.90, 0.60, 0.05, key="cap_ratio",
            help="The share of contracted time that becomes face-to-face therapy. "
                 "The rest goes to documentation, MDT and board rounds, handover, "
                 "travel between wards, supervision, students and annual training. "
                 "Published figures for inpatient therapy typically land between "
                 "0.50 and 0.65; measure yours before arguing from it.")
    with a3:
        dose_target = st.number_input("Guideline dose (minutes per applicable day)",
                                      15, 90, 45, 5, key="cap_dose")
    st.markdown(
        "**Why the contact ratio is the argument.** A business case built on "
        "contracted hours assumes a therapist spends every minute of the working "
        "day in front of a patient, which nobody has ever done. The ratio is the "
        "honest conversion between establishment and delivered care, and it is "
        "the number a finance director will challenge first — so measure it "
        "locally from a diary study rather than adopting a published figure.")

MINUTES_PER_WTE_WEEK = hours_week * 60 * contact_ratio

# ---------------------------------------------------------------------------
# Active caseload
# ---------------------------------------------------------------------------
st.subheader("Active caseload")

daily = (ses.groupby(["date", "site", "discipline"], observed=True)
         .agg(caseload=("admission_id", "nunique"),
              seen=("attended", "sum"),
              minutes=("minutes", "sum")).reset_index())
daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()

staff_m = staffing.copy()
staff_m["month"] = staff_m["week"].dt.to_period("M").dt.to_timestamp()
staff_m = (staff_m.groupby(["month", "site", "discipline"], observed=True)
           .agg(wte_funded=("wte_funded", "mean"),
                wte_available=("wte_available", "mean"),
                wte_vacant=("wte_vacant", "mean")).reset_index())

monthly = (daily.groupby(["month", "site", "discipline"], observed=True)
           .agg(mean_caseload=("caseload", "mean"),
                applicable_days=("caseload", "sum"),
                delivered_minutes=("minutes", "sum"),
                days_seen=("seen", "sum")).reset_index())
monthly = monthly.merge(staff_m, on=["month", "site", "discipline"], how="left")
monthly["caseload_per_wte"] = monthly["mean_caseload"] / monthly["wte_available"].replace(0, np.nan)

cc1, cc2 = st.columns([1, 3])
with cc1:
    disc = st.radio("Discipline", list(DISC.keys()), format_func=lambda d: DISC[d],
                    key="cap_disc")
    view = st.radio("View", ["Caseload per available WTE", "Active caseload",
                             "Available vs funded WTE"], key="cap_view")
sub = monthly[monthly["discipline"] == disc]

with cc2:
    if view == "Caseload per available WTE":
        viz.show(viz.multi_line(sub, "month", "caseload_per_wte", "site", height=330,
                                title=f"{DISC[disc]}: patients per available WTE",
                                ytitle="Patients per WTE"))
        st.caption(
            "Divided by **available** WTE, not funded. A service is not short "
            "against its establishment — it is short against the establishment it "
            "can actually field this week, after vacancy, leave and sickness. "
            "Funded-establishment figures make a staffing crisis invisible.")
    elif view == "Active caseload":
        viz.show(viz.multi_line(sub, "month", "mean_caseload", "site", height=330,
                                title=f"{DISC[disc]}: mean daily active caseload",
                                ytitle="Patients"))
    else:
        long = sub.melt(id_vars=["month", "site"],
                        value_vars=["wte_funded", "wte_available"],
                        var_name="measure", value_name="wte")
        p = viz.palette()
        fig = go.Figure()
        for i, site in enumerate(sorted(sub["site"].unique())):
            s2 = sub[sub["site"] == site].sort_values("month")
            fig.add_trace(go.Scatter(x=s2["month"], y=s2["wte_funded"], mode="lines",
                                     name=f"{site} — funded",
                                     line=dict(color=p["series"][i % 8], width=1.4,
                                               dash="dot"),
                                     hovertemplate="%{y:.1f} WTE<extra>funded</extra>"))
            fig.add_trace(go.Scatter(x=s2["month"], y=s2["wte_available"], mode="lines",
                                     name=f"{site} — available",
                                     line=dict(color=p["series"][i % 8], width=2.2),
                                     hovertemplate="%{y:.1f} WTE<extra>available</extra>"))
        fig.update_layout(**viz._layout(p, 330, f"{DISC[disc]}: funded vs available WTE",
                                        ytitle="Whole-time equivalents"))
        viz.show(fig)

# ---------------------------------------------------------------------------
# Demand vs capacity
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Demand against capacity")

weeks_per_month = 365.25 / 12 / 7
dc = (monthly[monthly["discipline"] == disc]
      .groupby("month", observed=True)
      .agg(applicable_days=("applicable_days", "sum"),
           delivered_minutes=("delivered_minutes", "sum"),
           wte_available=("wte_available", "sum"),
           wte_funded=("wte_funded", "sum")).reset_index())
dc["required_minutes"] = dc["applicable_days"] * dose_target
dc["available_minutes"] = dc["wte_available"] * MINUTES_PER_WTE_WEEK * weeks_per_month
dc["funded_minutes"] = dc["wte_funded"] * MINUTES_PER_WTE_WEEK * weeks_per_month
dc["shortfall"] = dc["required_minutes"] - dc["available_minutes"]
dc["wte_needed"] = dc["required_minutes"] / (MINUTES_PER_WTE_WEEK * weeks_per_month)
dc = dc.rename(columns={"month": "period"})

d1, d2 = st.columns([3, 2], gap="large")
with d1:
    viz.show(viz.demand_capacity(dc, height=360,
                                 title=f"{DISC[disc]}: minutes required, available and delivered"))
    st.caption(
        "The most recent period is right-censored: patients admitted near the "
        "end of the window are still in hospital, so their remaining applicable "
        "days have not happened yet. Expect the last point to rise as the month "
        "completes, and do not read a trend into it. Every live extract has this "
        "property; most dashboards do not mention it.")
with d2:
    recent = dc.tail(6)
    need = float(recent["wte_needed"].mean())
    have = float(recent["wte_available"].mean())
    funded = float(recent["wte_funded"].mean())
    m1, m2, m3 = st.columns(3)
    m1.metric("WTE available", f"{have:.1f}")
    m2.metric("WTE funded", f"{funded:.1f}")
    m3.metric("WTE to meet dose", f"{need:.1f}", delta=f"{have - need:+.1f}",
              delta_color="normal")
    st.markdown(
        f'<div class="sqi-note"><b>The gap, stated plainly.</b> Over the last six '
        f'months this service carried an average applicable caseload requiring '
        f'<b>{need:.1f} WTE</b> at {dose_target} minutes per applicable day and a '
        f'{contact_ratio:.0%} clinical contact ratio. It had <b>{have:.1f} WTE</b> '
        f'available against <b>{funded:.1f} WTE</b> funded.<br><br>'
        f'{"The establishment is adequate and the shortfall is vacancy and absence — a recruitment and retention problem, not a funding one." if funded >= need > have else ""}'
        f'{"Even fully staffed to establishment, this service cannot deliver the guideline dose. That is a commissioning conversation, and no amount of productivity improvement closes it." if funded < need else ""}'
        f'{"Capacity is sufficient at these assumptions; if dose is still short, the constraint is scheduling and job planning rather than headcount." if have >= need else ""}'
        f'</div>', unsafe_allow_html=True)
    st.caption(
        "Every number here inherits the contact-ratio assumption above. Move that "
        "slider and watch the conclusion move with it — which is exactly why it "
        "belongs on the page rather than buried in a spreadsheet.")

# ---------------------------------------------------------------------------
# Missed sessions
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Why sessions do not happen")

missed = ses[(~ses["attended"]) & (ses["missed_reason"] != "")].copy()
missed_disc = missed[missed["discipline"] == disc]

CAPACITY_REASONS = {"No therapist available"}

mc1, mc2 = st.columns([3, 2], gap="large")
with mc1:
    if missed_disc.empty:
        st.info("No missed sessions recorded for this discipline in the current selection.")
    else:
        counts = missed_disc["missed_reason"].value_counts()
        viz.show(viz.pareto(counts.index, counts.to_numpy(), height=330,
                            title=f"{DISC[disc]}: missed sessions by reason",
                            ytitle="Patient-days"))
with mc2:
    st.markdown(
        "**The most valuable field a therapy service can record.** Split the "
        "missed sessions into two buckets and they lead to completely different "
        "places:\n\n"
        "*Service-side* — no therapist available. This is capacity, and it is "
        "the only bucket that responds to establishment, job planning or "
        "seven-day rostering.\n\n"
        "*Patient-side* — medically unwell, declined, off ward for an "
        "investigation, fatigued. These respond to scheduling around ward "
        "rounds and imaging slots, to fatigue management, and sometimes to "
        "nothing at all. A service with a high patient-side share and a low "
        "service-side share is already running at its ceiling, and a business "
        "case built on its missed-session total will not survive scrutiny.")

if not missed_disc.empty:
    trend = missed_disc.copy()
    trend["month"] = trend["date"].dt.to_period("M").dt.to_timestamp()
    trend["capacity"] = trend["missed_reason"].isin(CAPACITY_REASONS)
    by_month = (trend.groupby(["month", "site"], observed=True)
                .agg(capacity_missed=("capacity", "sum"),
                     total_missed=("capacity", "size")).reset_index())
    by_month["pct_capacity"] = 100 * by_month["capacity_missed"] / by_month["total_missed"].clip(lower=1)
    viz.show(viz.multi_line(
        by_month, "month", "pct_capacity", "site", height=320,
        title="Share of missed sessions attributable to no therapist being available",
        ytitle="% of missed sessions"))
    st.caption(
        "This series is the trace a staffing gap leaves behind. A rise here "
        "without a rise in total missed sessions means the *mix* changed — the "
        "service absorbed the same number of losses but for a different reason, "
        "which is usually the first visible sign of a vacancy.")

    # Control chart on the capacity share -- it is a proportion with a
    # variable denominator, so it gets a p-chart like anything else.
    agg = (trend.groupby("month", observed=True)
           .agg(num=("capacity", "sum"), den=("capacity", "size")).reset_index())
    res = spc.p_chart(agg["num"], agg["den"], agg["month"], laney=f.laney,
                      target=None, higher_is_better=False, rule_set=f.rule_set,
                      unit="%", label="Capacity-attributable missed sessions")
    with st.expander("Control chart on the capacity share"):
        viz.show(viz.control_chart(res, height=320,
                                   title="Missed for lack of a therapist (% of all misses)"))
        v = mdc.verdict(res)
        ui.verdict_banner(v, res)

# ---------------------------------------------------------------------------
# Waiting to be seen
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Waiting for a first contact")

first = (ses[ses["attended"]].groupby(["admission_id", "discipline"], observed=True)["day_of_stay"]
         .min().rename("first_contact_day").reset_index())
first = first.merge(adm[["admission_id", "site", "arrival_datetime", "nihss"]],
                    on="admission_id", how="left")
first["month"] = first["arrival_datetime"].dt.to_period("M").dt.to_timestamp()
fd = first[first["discipline"] == disc]

w1, w2 = st.columns([3, 2], gap="large")
with w1:
    med = (fd.groupby(["month", "site"], observed=True)["first_contact_day"]
           .median().reset_index())
    viz.show(viz.multi_line(med, "month", "first_contact_day", "site", height=320,
                            title=f"{DISC[disc]}: median days from admission to first "
                                  f"delivered session",
                            ytitle="Days"))
with w2:
    rows = []
    for d in DISC:
        sub2 = first[first["discipline"] == d]["first_contact_day"]
        if sub2.empty:
            continue
        rows.append({"Discipline": DISC[d], "Median (days)": f"{sub2.median():.1f}",
                     "90th centile": f"{sub2.quantile(0.9):.1f}",
                     "Seen day 0–1": f"{100 * (sub2 <= 1).mean():.0f}%",
                     "Not seen before day 5": f"{100 * (sub2 > 5).mean():.0f}%"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(
        "This measures the wait to a *delivered* session, which is a harder and "
        "more honest test than time to assessment. A patient can be assessed "
        "inside 72 hours and then wait a further week for therapy to start; only "
        "the second delay is visible to the patient.")

# ---------------------------------------------------------------------------
# Establishment
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Establishment and skill mix")

e1, e2 = st.columns([2, 3], gap="large")
with e1:
    mix = (therapists.groupby(["discipline", "grade"], observed=True)["wte"]
           .sum().reset_index())
    mixd = mix[mix["discipline"] == disc]
    viz.show(viz.bar(mixd["grade"], mixd["wte"], horizontal=True, height=250,
                     title=f"{DISC[disc]}: funded WTE by grade", ytitle="WTE"))
with e2:
    vac = (staffing[staffing["discipline"] == disc]
           .groupby(["week", "site"], observed=True)["wte_vacant"].sum().reset_index())
    viz.show(viz.multi_line(vac, "week", "wte_vacant", "site", height=250,
                            title=f"{DISC[disc]}: vacant WTE", ytitle="WTE vacant"))
    st.caption(
        "Vacancy is the honest bridge between an establishment figure and a "
        "delivered-dose figure. When a dose indicator drops and vacancy rose in "
        "the same window, the causal story writes itself — and it is a far "
        "stronger business case than the dose chart on its own.")
