"""
Stroke Quality Intelligence — service overview and board scorecard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import mdc, metrics, ui, viz
from core.standards import HEADLINE_METRICS, DOMAINS, METRICS, metric, metrics_in_domain

ui.setup("Service overview")
f = ui.sidebar()
data = ui.filtered(f)
adm = data["admissions"]
tbl = ui.period_table(f)

# ---------------------------------------------------------------------------
st.title("Stroke service quality")
st.caption(
    f"{len(adm):,} admissions across {adm['site'].nunique()} site(s), "
    f"{adm['arrival_datetime'].min():%b %Y} – {adm['arrival_datetime'].max():%b %Y} · "
    f"indicators and targets on the **{f.standard}** standard"
)

# ---- context strip --------------------------------------------------------
c = st.columns(5)
with c[0]:
    st.metric("Stroke admissions", f"{len(adm):,}")
with c[1]:
    st.metric("Median age", f"{adm['age'].median():.0f}")
with c[2]:
    st.metric("Median NIHSS", f"{adm['nihss'].median():.0f}")
with c[3]:
    st.metric("Haemorrhagic", f"{100 * (adm['stroke_type'] != 'Ischaemic').mean():.0f}%")
with c[4]:
    st.metric("Stroke bed days", f"{adm['los_days'].sum():,.0f}")

st.markdown(
    '<div class="sqi-note">These five are <b>context, not performance</b>. '
    'They describe who walked through the door, and they are the first thing to '
    'check when an indicator moves: a rising median NIHSS will move mortality, '
    'length of stay and therapy dose all at once without anything in the service '
    'having changed.</div>', unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Board scorecard
# ---------------------------------------------------------------------------
st.subheader("Board scorecard")
st.caption("Two questions per indicator, answered separately: has anything changed, "
           "and can this process be relied on to meet its target?")

results: dict[str, object] = {}
verdicts: dict[str, mdc.MDCVerdict] = {}

keys = [k for k in HEADLINE_METRICS if k in METRICS]
cols_per_row = 4
for row_start in range(0, len(keys), cols_per_row):
    row_keys = keys[row_start:row_start + cols_per_row]
    cols = st.columns(cols_per_row, gap="medium")
    for col, key in zip(cols, row_keys):
        with col:
            try:
                res = ui.series(tbl, key, f)
                results[key] = res
                verdicts[key] = ui.metric_card(res, tbl, key, f)
            except Exception as exc:  # a broken indicator must not kill the page
                st.warning(f"{METRICS[key].label}: {exc}")
    st.write("")

# ---------------------------------------------------------------------------
with st.expander("How to read the two icons"):
    st.markdown(
        "Most performance reports answer one question badly: *are we above or "
        "below target this month?* That conflates two different things and gets "
        "both wrong. A stable process crosses its target at random; a single "
        "month above it is not achievement, and a single month below it is not "
        "failure.\n\n"
        "**Variation** (left icon) asks only whether the process has changed. It "
        "has nothing to do with the target. It comes from the control chart "
        "rules — a point outside the limits, a run on one side of the mean, a "
        "sustained trend.\n\n"
        "**Assurance** (right icon) asks whether the process, behaving as it "
        "normally does, can be relied on to meet the target. It compares the "
        "*process limits* with the target, not the latest point with the target. "
        "That is why a service can post a good month and still be told the "
        "target is unreliable."
    )

    grid = [
        ("common", "pass", "Stable and reliably meeting target. Nothing to do — and "
                           "specifically, do not investigate individual months."),
        ("common", "fail", "Stable and reliably missing target. The system is doing "
                           "exactly what it was designed to do. Redesign is the only "
                           "route; performance management of the current process "
                           "will not close the gap."),
        ("common", "hit_miss", "Stable, but the target sits inside natural variation. "
                               "Whether it is met in any month is close to chance. "
                               "Reduce variation before chasing the mean."),
        ("high_improve", "hit_miss", "Something has changed for the better. Find out "
                                     "what and hold it. Do not reset the limits until "
                                     "the change is confirmed over several periods."),
        ("low_concern", "fail", "Genuine deterioration against a target already "
                                "unreachable. This is the combination that warrants "
                                "an urgent look."),
    ]
    rows = []
    for var, ass, text in grid:
        rows.append(
            '<tr>'
            f'<td style="padding:9px 12px 9px 0;vertical-align:top;white-space:nowrap;">'
            f'{mdc.variation_icon(var, 26)} {mdc.assurance_icon(ass, 26)}</td>'
            f'<td style="padding:9px 0;font-size:0.85rem;line-height:1.45;">{text}</td>'
            '</tr>')
    st.markdown('<table style="border-collapse:collapse;">' + "".join(rows) + "</table>",
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Signals needing attention
# ---------------------------------------------------------------------------
st.divider()
st.subheader("What actually needs attention")

rows = []
for domain in DOMAINS:
    for key in metrics_in_domain(domain, f.standard):
        try:
            res = results.get(key) or ui.series(tbl, key, f)
            results[key] = res
            v = verdicts.get(key) or mdc.verdict(res)
            verdicts[key] = v
        except Exception:
            continue
        spec, thr = metric(key, f.standard, f.overrides)
        rows.append({
            "Domain": domain,
            "Indicator": spec.label,
            "Latest": ui.fmt(metrics.current_value(tbl, key, 3), spec.unit),
            "Unit": ui.unit_suffix(spec.unit),
            "Variation": v.variation_label,
            "Assurance": v.assurance_label,
            "_priority": (0 if v.variation.endswith("concern") else
                          1 if v.assurance == "fail" else
                          2 if v.assurance == "hit_miss" else
                          3 if v.variation.endswith("improve") else 4),
        })

summary = pd.DataFrame(rows).sort_values(["_priority", "Domain"])
concern = summary[summary["_priority"] == 0]
failing = summary[summary["_priority"] == 1]
improving = summary[summary["_priority"] == 3]

cc = st.columns(3, gap="large")
with cc[0]:
    st.markdown(f"##### {len(concern)} deteriorating")
    st.caption("Special-cause variation in the wrong direction. Real change, "
               "not noise — worth a conversation this week.")
    if concern.empty:
        st.caption("— none —")
    else:
        for _, r in concern.iterrows():
            st.markdown(f"**{r['Indicator']}** — {r['Latest']}{r['Unit']}")
with cc[1]:
    st.markdown(f"##### {len(failing)} stable but failing")
    st.caption("No signal, and the whole range of natural variation sits on the "
               "wrong side of target. These need redesign, not chasing.")
    if failing.empty:
        st.caption("— none —")
    else:
        for _, r in failing.iterrows():
            st.markdown(f"**{r['Indicator']}** — {r['Latest']}{r['Unit']}")
with cc[2]:
    st.markdown(f"##### {len(improving)} improving")
    st.caption("Special cause in the right direction. Find the cause and hold it "
               "before the limits are recalculated.")
    if improving.empty:
        st.caption("— none —")
    else:
        for _, r in improving.iterrows():
            st.markdown(f"**{r['Indicator']}** — {r['Latest']}{r['Unit']}")

with st.expander("Full indicator summary"):
    st.dataframe(summary.drop(columns="_priority"), hide_index=True, width="stretch")
    st.download_button("Download scorecard as CSV",
                       summary.drop(columns="_priority").to_csv(index=False).encode(),
                       file_name="stroke_scorecard.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Between-site comparison
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Comparing sites without a league table")

left, right = st.columns([1, 1], gap="large")

p_keys = [k for k in metrics_in_domain("Hyperacute pathway", f.standard)
          + metrics_in_domain("Stroke unit & ward care", f.standard)
          + metrics_in_domain("Therapy & rehabilitation", f.standard)
          + metrics_in_domain("Outcomes", f.standard)
          if METRICS[k].chart == "p"]

with left:
    fkey = ui.metric_picker(p_keys, f, "Indicator to compare", key="funnel_metric",
                            default="su_4h")
    fn = metrics.funnel_for_metric(adm, fkey, "site", f.standard)
    if fn is None:
        st.info("Need at least three sites in the current selection.")
    else:
        spec, thr = metric(fkey, f.standard, f.overrides)
        viz.show(viz.funnel_chart(fn, title=spec.label, unit="%"))
        st.caption(
            f"Dispersion φ = {fn['phi']:.2f}. "
            + ("Close to 1: binomial sampling alone explains the spread between "
               "sites, so the limits are the plain exact-binomial ones."
               if fn["phi"] < 1.3 else
               f"Above 1: the sites vary by more than binomial sampling can "
               f"produce, so the limits are widened by √φ = {fn['inflation']:.2f}. "
               "The honest reading of a large φ is 'these units are not running "
               "the same process', not 'most of these units are outliers'."))

with right:
    st.markdown("**Why a funnel and not a bar chart**")
    st.markdown(
        "A ranked bar chart of sites answers a question nobody should ask. "
        "Someone is always bottom, and with small denominators their position is "
        "mostly sampling noise — rank a set of identical sites on 30 patients "
        "each and you still get a convincing-looking spread.\n\n"
        "The funnel plots the indicator against the number of patients it was "
        "measured on. The limits widen as volume falls, which is the whole "
        "point: a small unit has to be much further from the mean before the "
        "difference is real. Sites inside the funnel are consistent with a "
        "common rate. There is no ranking, and no implied ordering.\n\n"
        "Two limits are drawn. The inner pair (95%) is a screening threshold — "
        "expect roughly one site in twenty outside it by chance alone. The outer "
        "pair (99.8%) is the one to act on.")

    mrs = metrics.mrs_distribution(adm, "site")
    viz.show(viz.stacked_shift(mrs, "site", height=250,
                               title="Disability at discharge (mRS distribution)"))
    st.caption(
        "The 'independent at discharge' indicator dichotomises this distribution "
        "at mRS 2 and throws away everything else. A patient moving from mRS 5 to "
        "4 — bedbound to needing help to walk — is a large clinical gain that the "
        "dichotomy scores as nothing. Read the shift, not just the cut point.")

st.divider()
st.caption(
    "Simulated data, generated to exercise the statistical machinery. Planted "
    "changes include a thrombolysis pathway redesign, a period of bed pressure, "
    "and a physiotherapy staffing gap — the charts should find all three. "
    "Nothing here describes a real service."
)
