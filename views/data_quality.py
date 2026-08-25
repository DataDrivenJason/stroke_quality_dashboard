"""Data quality, the schema contract, and loading your own extract."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import loaders, ui, viz

ui.setup("Data quality")
f = ui.sidebar()
tables = ui.get_tables()
data = ui.filtered(f)
adm = data["admissions"]
ses = data["sessions"]

st.title("Data quality")
st.caption("Every indicator in this dashboard inherits the quality of the fields "
           "underneath it. This page is where you find out what you are actually "
           "measuring before anyone puts it in a board paper.")

tab_report, tab_complete, tab_digit, tab_load = st.tabs(
    ["Validation report", "Completeness over time", "Digit preference", "Load your own data"])

# ===========================================================================
with tab_report:
    issues = loaders.validate(tables)
    if not issues:
        st.success("No validation issues found.")
    else:
        df = pd.DataFrame([{"Table": i.table, "Column": i.column,
                            "Severity": i.severity.title(), "Issue": i.message,
                            "Rows affected": i.n_affected or ""} for i in issues])
        sev_order = {"Error": 0, "Warning": 1, "Info": 2}
        df = df.sort_values("Severity", key=lambda s: s.map(sev_order))
        n_err = int((df["Severity"] == "Error").sum())
        n_warn = int((df["Severity"] == "Warning").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Errors", n_err)
        c2.metric("Warnings", n_warn)
        c3.metric("Notes", int((df["Severity"] == "Info").sum()))
        st.dataframe(df, hide_index=True, width="stretch")

    st.markdown(
        "**Validation here is advisory, never blocking.** Real extracts are "
        "always imperfect, and a dashboard that refuses to open until the data is "
        "clean gets abandoned within a fortnight. So problems are reported, "
        "quantified, and left visible — which also means the person reading the "
        "chart can see the caveat rather than being protected from it.\n\n"
        "The checks that catch the most are not the missing-value counts. They "
        "are the **impossible values** (an NIHSS of 47, a negative door-to-needle "
        "time) and the **internal contradictions** (a needle time on a patient not "
        "flagged as thrombolysed, attended therapy recorded with zero minutes). A "
        "missing value usually means a field was not collected; an impossible one "
        "usually means it is being populated from the wrong source, which is a "
        "much bigger problem and a much easier one to fix.")

    st.divider()
    st.markdown("### Rule 6 and rule 7 as data-quality alarms")
    st.markdown(
        "Two of the Nelson rules almost never indicate clinical excellence in "
        "audit data, and are worth more attention than a point outside the limits:\n\n"
        "**Rule 6 — fifteen consecutive points hugging the centre line.** A "
        "process that looks *too* well behaved. In practice this means "
        "stratification, a wrongly-defined subgroup, heavy rounding, or a figure "
        "that has been smoothed somewhere upstream before it reached you.\n\n"
        "**Rule 7 — eight consecutive points all away from the centre line with "
        "none near it.** The signature of two different processes averaged "
        "together. Weekday and weekend. Two sites reported as one. Two coders "
        "applying a definition differently.\n\n"
        "Switch the rule set to Nelson in the sidebar to enable both.")

# ===========================================================================
with tab_complete:
    st.markdown("### Completeness over time")
    st.markdown(
        "A step change in completeness is a step change in every indicator built "
        "on that field, and it will present itself as a clinical improvement. "
        "This is the most common way an audit dashboard misleads a board: the "
        "service did not get better, the recording did.")

    fields = ["nihss", "onset_to_door_min", "door_to_ct_min", "door_to_needle_min",
              "time_to_su_hours", "swallow_screen_hours", "pt_assess_hours",
              "ot_assess_hours", "slt_assess_hours", "mrs_discharge", "prestroke_mrs"]
    fields = [c for c in fields if c in adm.columns]
    chosen = st.multiselect("Fields", fields,
                            default=["nihss", "onset_to_door_min", "door_to_needle_min",
                                     "mrs_discharge"], key="dq_fields")

    d = adm.copy()
    d["period"] = d["arrival_datetime"].dt.to_period("M").dt.to_timestamp()
    rows = []
    for c in chosen:
        # Only count completeness within the cohort the field applies to,
        # otherwise "door_to_needle_min is 92% missing" is meaningless --
        # it is missing because 92% of patients were not thrombolysed.
        applicable = pd.Series(True, index=d.index)
        if c == "door_to_needle_min":
            applicable = d["thrombolysed"]
        elif c == "onset_to_door_min":
            applicable = d["onset_known"]
        elif c.endswith("_assess_hours"):
            applicable = d[c.replace("_assess_hours", "_needed")]
        g = (d[applicable].groupby("period", observed=True)[c]
             .agg(complete="count", total="size").reset_index())
        g["pct"] = 100 * g["complete"] / g["total"].clip(lower=1)
        g["field"] = c
        rows.append(g)
    if rows:
        comp = pd.concat(rows)
        viz.show(viz.multi_line(comp, "period", "pct", "field", height=350,
                                title="Field completeness within the applicable cohort",
                                ytitle="% complete"))
        st.caption(
            "Completeness is calculated within the cohort each field applies to. "
            "Reporting door-to-needle as '92% missing' across all admissions is "
            "technically true and completely uninformative — it is missing "
            "because those patients were not thrombolysed.")

    st.divider()
    st.markdown("### Denominator drift")
    st.markdown(
        "The 'needs therapy' denominators are clinical judgements recorded by the "
        "service being measured. A team under pressure can raise its 72-hour "
        "compliance by recording fewer patients as requiring input, and every "
        "compliance indicator will applaud. Watch the two together.")
    dd = (d.groupby("period", observed=True)
          .agg(admissions=("admission_id", "count"),
               pt=("pt_needed", "mean"), ot=("ot_needed", "mean"),
               slt=("slt_needed", "mean")).reset_index())
    long = dd.melt(id_vars="period", value_vars=["pt", "ot", "slt"],
                   var_name="discipline", value_name="share")
    long["share"] *= 100
    long["discipline"] = long["discipline"].str.upper()
    viz.show(viz.multi_line(long, "period", "share", "discipline", height=300,
                            title="Share of admissions recorded as needing each therapy",
                            ytitle="% of admissions"))

# ===========================================================================
with tab_digit:
    st.markdown("### Digit preference")
    st.markdown(
        "Times entered by hand cluster on round numbers. A door-to-needle time "
        "recorded as 45 when the clock said 43, over and over, does not change "
        "the median much — but it does something worse: it compresses the "
        "variance, which narrows the control limits, which manufactures special "
        "cause signals out of nothing.\n\n"
        "The test is simple. Under accurate recording the final digit of a "
        "minute-level timing should be roughly uniform across 0–9, at about 10% "
        "each. Spikes at 0 and 5 are the fingerprint of estimation. This is a "
        "standard forensic check in audit data and it takes about a second to run.")

    tcol = st.selectbox("Timing field",
                        [c for c in ["door_to_needle_min", "door_to_ct_min",
                                     "onset_to_door_min", "door_to_puncture_min"]
                         if c in adm.columns], key="dq_digit")
    vals = pd.to_numeric(adm[tcol], errors="coerce").dropna()
    if len(vals) < 40:
        st.info("Too few values for a meaningful digit-preference check.")
    else:
        last = (vals.astype(int) % 10).value_counts().reindex(range(10), fill_value=0)
        pct = 100 * last / last.sum()
        expected = 10.0
        # Chi-square against uniform. With large n this rejects on trivial
        # departures, so read the effect size (the 0/5 excess) alongside it.
        chi2 = float(((last - last.sum() / 10) ** 2 / (last.sum() / 10)).sum())
        excess = float(pct.loc[0] + pct.loc[5] - 20.0)

        g1, g2 = st.columns([3, 2], gap="large")
        with g1:
            p = viz.palette()
            fig = go.Figure(go.Bar(
                x=[str(i) for i in range(10)], y=pct.to_numpy(),
                marker=dict(color=p["series"][0], line=dict(color=p["surface"], width=2)),
                hovertemplate="<b>%{y:.1f}%</b> of values<extra>final digit %{x}</extra>"))
            fig.add_hline(y=expected, line=dict(color=p["target"], width=1.8, dash="dot"),
                          annotation_text="uniform expectation (10%)",
                          annotation_font=dict(size=10.5, color=p["muted"]))
            lay = viz._layout(p, 320, f"Final digit of {tcol}",
                              ytitle="% of values", xtitle="Final digit")
            lay["showlegend"] = False
            lay["hovermode"] = "closest"
            fig.update_layout(**lay)
            viz.show(fig)
        with g2:
            st.metric("Excess on 0 and 5", f"{excess:+.1f} pp",
                      help="Percentage points above the 20% expected under uniformity.")
            st.metric("χ² against uniform (9 df)", f"{chi2:.1f}")
            st.markdown(
                ("**Consistent with accurate recording.** No material clustering "
                 "on round numbers." if abs(excess) < 6 else
                 "**Rounding is present.** Values cluster on 0 and 5 more than "
                 "chance allows. Treat this field's variance — and therefore any "
                 "control limits built on it — as artificially narrow, and be "
                 "sceptical of signals close to the limit.")
                + "\n\nWith a large sample the χ² will reject uniformity on "
                "departures far too small to matter. Read the excess on 0 and 5 "
                "as the effect size and let the χ² be a footnote.")

# ===========================================================================
with tab_load:
    st.markdown("### Load your own extract")
    st.markdown(
        "The schema below is the contract between this dashboard and your data "
        "warehouse. Everything in the app is written against it, so porting to "
        "real data means producing two files — or two SQL views — with these "
        "columns, not editing the dashboard.")

    st.download_button("Download the CSV template and data dictionary",
                       loaders.template_zip(), file_name="stroke_dashboard_template.zip",
                       mime="application/zip")

    st.markdown(
        "**The one column that decides whether this works: `sessions.applicable`.** "
        "It must be TRUE for every patient-day on which therapy was indicated, "
        "whether or not any therapy happened. It is the denominator. Services "
        "that record only delivered sessions cannot compute percentage-of-days or "
        "45-minute attainment at all, and usually discover this at the moment "
        "someone asks them to prove their reliability.")

    up1, up2, up3 = st.columns(3)
    with up1:
        adm_file = st.file_uploader("admissions.csv", type=["csv"], key="up_adm")
    with up2:
        ses_file = st.file_uploader("sessions.csv", type=["csv"], key="up_ses")
    with up3:
        staff_file = st.file_uploader("staffing.csv (optional)", type=["csv"], key="up_staff")

    if adm_file is not None and ses_file is not None:
        if st.button("Load this data", type="primary"):
            try:
                new = {"admissions": pd.read_csv(adm_file),
                       "sessions": pd.read_csv(ses_file)}
                if staff_file is not None:
                    new["staffing"] = pd.read_csv(staff_file)
                else:
                    new["staffing"] = tables["staffing"].iloc[0:0]
                new["therapists"] = tables["therapists"].iloc[0:0]
                new = loaders.coerce(new)
                found = loaders.validate(new)
                errors = [i for i in found if i.severity == "error"]
                if errors:
                    st.error(f"{len(errors)} blocking problem(s) — see below. "
                             "The data was not loaded.")
                    st.dataframe(pd.DataFrame(
                        [{"Table": i.table, "Column": i.column, "Issue": i.message}
                         for i in errors]), hide_index=True, width="stretch")
                else:
                    for k in list(st.session_state.keys()):
                        if str(k).startswith("pt::"):
                            del st.session_state[k]
                    st.session_state["tables"] = loaders.prepare(new)
                    st.session_state["source"] = f"Uploaded · {adm_file.name}"
                    st.success("Loaded. Every page now runs on your data.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not read the files: {exc}")

    if st.session_state.get("source", "").startswith("Uploaded"):
        if st.button("Return to the demonstration dataset"):
            for k in list(st.session_state.keys()):
                if str(k).startswith("pt::") or k in ("tables", "source"):
                    del st.session_state[k]
            st.rerun()

    st.divider()
    st.markdown("### The schema")
    for name, schema in loaders.SCHEMAS.items():
        with st.expander(f"`{name}` — {len(schema)} columns"):
            st.dataframe(pd.DataFrame(
                [{"Column": c, "Type": k, "Required": "yes" if r else "no",
                  "Description": d} for c, (k, r, d) in schema.items()]),
                hide_index=True, width="stretch")
