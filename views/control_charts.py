"""The control chart workbench: any indicator, any chart, with the maths shown."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import mdc, metrics, spc, ui, viz
from core.standards import DOMAINS, METRICS, metric, metrics_in_domain

ui.setup("Control charts")
f = ui.sidebar()
data = ui.filtered(f)
adm = data["admissions"]

st.title("Control charts")
st.caption("One question, asked properly: is this variation the ordinary noise of a "
           "stable process, or has something genuinely changed?")

tab_main, tab_cusum, tab_funnel, tab_rare, tab_rules = st.tabs(
    ["Shewhart chart", "CUSUM", "Funnel plot", "Rare events", "The rules"])

# ===========================================================================
with tab_main:
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        domain = st.selectbox("Domain", DOMAINS, key="cc_domain")
    with c2:
        keys = metrics_in_domain(domain, f.standard)
        if not keys:
            st.info("No indicators in this domain under the selected standard.")
            st.stop()
        key = ui.metric_picker(keys, f, "Indicator", key="cc_metric")
    with c3:
        split = st.selectbox("Split by", ["All selected sites", "One site"], key="cc_split")

    if split == "One site":
        site = st.selectbox("Site", sorted(adm["site"].unique()), key="cc_site")
        tbl_all = ui.period_table(f, group_col="site")
        tbl = tbl_all[tbl_all["site"] == site].reset_index(drop=True)
        scope = site
    else:
        tbl = ui.period_table(f)
        scope = f"{adm['site'].nunique()} site(s)"

    spec, thr = metric(key, f.standard, f.overrides)

    # -- phase break control -------------------------------------------------
    periods = pd.to_datetime(tbl["period"])
    preview = ui.series(tbl, key, f)
    cp = spc.suggest_phase_break(preview.frame["value"])

    with st.expander("Limit calculation for this chart", expanded=False):
        st.markdown(
            "Limits calculated over the whole series assume the process has one "
            "level throughout. When a real step change sits inside the series the "
            "centre line lands between the two levels, and the run rules then flag "
            "most of the chart. That is correct arithmetic and useless "
            "information — the chart says 'something changed' without saying what "
            "it changed *from*.\n\n"
            "Worth knowing precisely, because it is usually overstated: the limits "
            "themselves are barely affected. Sigma comes from the moving range, a "
            "step contaminates exactly one moving range out of n−1, and the "
            "screening rule discards it — so a 25-sigma jump estimates the same "
            "sigma as a 3-sigma one. Whole-series limits rarely *hide* a change. "
            "What they lose is the reference level.\n\n"
            "Two honest responses: **freeze** the limits on a baseline period and "
            "extend them forward (best when you are testing whether a change "
            "worked), or **split** them at the date the change went live (best "
            "for reporting a settled new normal). Both require you to name the "
            "change. A phase break you cannot explain is curve-fitting.")
        if cp:
            when = periods.iloc[cp["index"]]
            verdict_txt = ("worth considering" if cp["worth_splitting"]
                           else "probably not worth splitting")
            st.markdown(
                f"**Largest candidate step:** {when:%b %Y}, mean "
                f"{cp['mean_before']:.2f} → {cp['mean_after']:.2f} "
                f"({cp['shift']:+.2f}, t = {cp['t']:.1f}) — {verdict_txt}.")
        use_break = st.checkbox("Split the limits at a date", key="cc_break")
        phases = None
        if use_break:
            default_idx = cp["index"] if cp else len(tbl) // 2
            chosen = st.select_slider(
                "Recalculate limits from", options=list(range(1, len(tbl))),
                value=min(default_idx, len(tbl) - 1),
                format_func=lambda i: f"{periods.iloc[i]:%b %Y}", key="cc_break_at")
            phases = [chosen]

    res = metrics.metric_series(tbl, key, standard=f.standard, overrides=f.overrides,
                                rule_set=f.rule_set, laney=f.laney,
                                baseline=f.baseline, phases=phases)
    v = mdc.verdict(res)

    left, right = st.columns([3, 1], gap="large")
    with left:
        viz.show(viz.control_chart(res, title=f"{spec.label} — {scope}", height=420))
        if res.chart_type == "xmr":
            viz.show(viz.mr_chart(res))
            st.caption("The moving-range chart below the individuals chart is not "
                       "decoration. A stable mean with a growing moving range is a "
                       "service becoming unpredictable, and the individuals chart "
                       "alone cannot show it — worse, its widening limits will "
                       "swallow points that should have signalled.")
    with right:
        ui.verdict_banner(v, res)
        if res.meta.get("overdispersed"):
            st.markdown(
                f'<div class="sqi-note"><b>Overdispersed (σz = '
                f'{res.meta["sigma_z"]:.2f}).</b> The scatter between periods is '
                f'{res.meta["sigma_z"]:.1f}× what binomial sampling can produce, so '
                f'the limits have been widened. Uncorrected, this chart would have '
                f'flagged most periods as special cause — a model failure dressed '
                f'up as a detection triumph.</div>', unsafe_allow_html=True)

    ui.methods_block(res, key, f, expanded=False)

    st.markdown("##### Signals")
    ui.signal_table(res)
    ui.table_view(res)

    # -- run chart alternative ----------------------------------------------
    with st.expander("Run chart view (for short series, or sceptical audiences)"):
        rc = spc.run_chart(res.frame["value"])
        p = viz.palette()
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=periods, y=np.full(len(rc), rc["median"].iloc[0]),
                                 mode="lines", name="Median",
                                 line=dict(color=p["ink2"], width=1.6)))
        fig.add_trace(go.Scatter(
            x=periods, y=rc["value"], mode="lines+markers", name="Value",
            line=dict(color=p["series"][0], width=2),
            marker=dict(size=np.where(rc["shift"] | rc["trend"], 11, 7),
                        symbol=np.where(rc["shift"] | rc["trend"], "diamond", "circle"),
                        color=np.where(rc["shift"] | rc["trend"], mdc.WARNING, p["series"][0]),
                        line=dict(color=p["surface"], width=1.8)),
            hovertemplate="%{y:.2f}<extra></extra>"))
        fig.update_layout(**viz._layout(p, 300, "", ytitle=spec.unit))
        viz.show(fig)
        st.markdown(
            f"Runs observed: **{int(rc['n_runs'].iloc[0])}**, expected "
            f"{rc['runs_lower'].iloc[0]:.0f}–{rc['runs_upper'].iloc[0]:.0f}. "
            + ("Too few runs — the series is clumping, which is a shift signal. "
               if rc["too_few_runs"].iloc[0] else
               "Too many runs — the series alternates more than chance allows, "
               "usually a sign of over-adjustment or of two processes alternating. "
               if rc["too_many_runs"].iloc[0] else
               "The number of runs is consistent with a stable process."))
        st.markdown(
            "A run chart needs no distributional assumption and no minimum number "
            "of points, which makes it the right tool below about a dozen periods "
            "and the right tool for an audience that will not accept control "
            "limits. It is strictly less sensitive: it can detect shifts and "
            "trends, but it has no equivalent of 'this single point is beyond "
            "anything the process produces'.")

# ===========================================================================
with tab_cusum:
    st.markdown("### Cumulative sum")
    st.markdown(
        "A Shewhart chart examines one point at a time. That makes it fast at "
        "catching large jumps and slow at catching small sustained drifts — and "
        "small sustained drifts are what service deterioration usually looks "
        "like. The CUSUM accumulates evidence instead:")
    st.latex(r"C^+_i = \max\left(0,\; C^+_{i-1} + (x_i - \mu_0) - k\right)")
    st.markdown(
        r"Subtracting the reference value $k$ is what makes this a detector "
        r"rather than a random walk: with no shift present the increments are "
        r"negative on average, so the statistic rests against its zero barrier. "
        r"Setting $k = \tfrac12\sigma$ tunes it to detect a one-sigma shift "
        r"efficiently, and a decision interval $h = 4\sigma$ or $5\sigma$ gives "
        r"an in-control false-alarm rate comparable to a 3-sigma Shewhart chart."
        "\n\n"
        "The cost is interpretability: a CUSUM tells you when accumulated "
        "evidence crossed a threshold, not what the rate is now, and the crossing "
        "lags the change that caused it. Run it beside the Shewhart chart, never "
        "instead of it.")

    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        cdomain = st.selectbox("Domain", DOMAINS, key="cu_domain")
    with cc2:
        ckeys = metrics_in_domain(cdomain, f.standard)
        ckey = ui.metric_picker(ckeys, f, "Indicator", key="cu_metric") if ckeys else None
    with cc3:
        k_sig = st.slider("k (σ)", 0.25, 1.5, 0.5, 0.25, key="cu_k",
                          help="Half the shift size you want to detect quickly.")
    with cc4:
        h_sig = st.slider("h (σ)", 3.0, 6.0, 4.0, 0.5, key="cu_h",
                          help="Decision interval. Lower is more sensitive and "
                               "raises more false alarms.")

    if ckey:
        ctbl = ui.period_table(f)
        cres = ui.series(ctbl, ckey, f)
        cspec, _ = metric(ckey, f.standard, f.overrides)
        base_n = f.baseline or max(8, len(cres.frame) // 3)
        base = cres.frame["value"].head(base_n)
        mu0 = float(np.nanmean(base))
        sigma = float(np.nanmean(cres.frame["sigma"])) if "sigma" in cres.frame else None
        cs = spc.cusum(cres.frame["value"], pd.to_datetime(ctbl["period"]),
                       target_mean=mu0, sigma=sigma, k_sigma=k_sig, h_sigma=h_sig)

        viz.show(viz.control_chart(cres, title=cspec.label, height=300))
        viz.show(viz.cusum_chart(cs, title="Accumulated departure from the baseline mean"))

        first_shew = cres.frame.index[cres.frame["special"]].min() if cres.signals.shape[0] else None
        first_cusum = cs.index[cs["signal_high"] | cs["signal_low"]].min() \
            if (cs["signal_high"] | cs["signal_low"]).any() else None
        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline mean μ₀", f"{mu0:.2f}")
        m2.metric("First Shewhart signal",
                  "—" if first_shew is None else
                  f"{pd.to_datetime(ctbl['period']).iloc[int(first_shew)]:%b %Y}")
        m3.metric("First CUSUM signal",
                  "—" if first_cusum is None else
                  f"{pd.to_datetime(ctbl['period']).iloc[int(first_cusum)]:%b %Y}")
        st.caption("Where the CUSUM fires earlier than the Shewhart chart, the "
                   "change was a small persistent drift. Where the Shewhart chart "
                   "fires first, it was a jump. That comparison is itself "
                   "diagnostic about what kind of change happened.")

# ===========================================================================
with tab_funnel:
    st.markdown("### Funnel plot")
    fkeys = [k for d in DOMAINS for k in metrics_in_domain(d, f.standard)
             if METRICS[k].chart == "p"]
    g1, g2, g3 = st.columns([2, 1, 1])
    with g1:
        fkey = ui.metric_picker(fkeys, f, "Indicator", key="fn_metric", default="su_4h")
    with g2:
        group = st.selectbox("Compare", ["site", "stroke_type", "sex"], key="fn_group",
                             format_func=lambda s: {"site": "Sites",
                                                    "stroke_type": "Stroke type",
                                                    "sex": "Sex"}[s])
    with g3:
        od = st.checkbox("Adjust for overdispersion", value=True, key="fn_od")

    fn = metrics.funnel_for_metric(adm, fkey, group, f.standard)
    if fn is None:
        st.info("Need at least three groups with data.")
    else:
        if not od:
            fn = spc.funnel_plot(fn["points"]["numerator"], fn["points"]["denominator"],
                                 fn["points"]["label"], overdispersion=False)
        fspec, _ = metric(fkey, f.standard, f.overrides)
        viz.show(viz.funnel_chart(fn, title=fspec.label, height=430))
        st.markdown(
            f"**Dispersion φ = {fn['phi']:.2f}.** " +
            ("Binomial sampling alone explains the spread; the exact binomial "
             "limits are used unadjusted."
             if fn["phi"] < 1.3 else
             f"The groups vary by {fn['phi']:.1f}× more than binomial sampling can "
             f"produce, so limits are inflated by √φ = {fn['inflation']:.2f}. "))
        st.markdown(
            "**Why the adjustment is not a fudge.** φ is estimated as the mean of "
            "the squared z-scores, winsorised at the 10th and 90th centiles so "
            "that genuine outliers cannot inflate the very limits meant to detect "
            "them. A large φ means the units are not running the same process — "
            "different case mix, different context, different pathways. Declaring "
            "nine of twelve units 'outliers' against limits that assume they are "
            "identical is not a finding, it is a rejected model.\n\n"
            "**What the funnel does not do** is adjust for case mix. Two sites can "
            "both sit inside the funnel while one treats a far sicker population. "
            "For mortality and functional outcome, a risk-adjusted funnel — "
            "plotting observed over expected from a model on age, NIHSS, "
            "pre-stroke mRS and stroke type — is the minimum defensible display.")
        ui.df(fn["points"].round(2), hide_index=True)

# ===========================================================================
with tab_rare:
    st.markdown("### Rare events: t- and g-charts")
    st.markdown(
        "When an event is rare, a monthly count chart is mostly zeros. The mean "
        "is tiny, the lower limit is pinned at zero, and the chart can only ever "
        "signal upward — it is structurally incapable of showing improvement.\n\n"
        "The fix is to change what is plotted. A **t-chart** plots the number of "
        "*days* between successive events; a **g-chart** plots the number of "
        "*admissions* between them. Improvement now appears as points drifting "
        "upward, which is both detectable and, in a ward safety huddle, "
        "considerably more motivating than a row of zeros.")

    ev1, ev2 = st.columns([1, 3])
    with ev1:
        event = st.radio("Event", ["Pressure ulcers", "Falls", "Hospital-acquired pneumonia"],
                         key="rare_event")
    col = {"Pressure ulcers": "pressure_ulcers", "Falls": "falls",
           "Hospital-acquired pneumonia": "hap"}[event]

    ev = adm[adm[col].astype(float) > 0].copy()
    if len(ev) < 12:
        st.info(f"Only {len(ev)} events in the current selection — too few for a "
                "rare-event chart. Widen the date range or include more sites.")
    else:
        # Event date: mid-stay is a reasonable stand-in when the incident
        # date is not carried in the extract. Replace with the real
        # incident timestamp when you have one -- interval charts are
        # sensitive to it in a way that count charts are not.
        ev["event_date"] = (ev["arrival_datetime"]
                            + pd.to_timedelta(ev["los_days"] / 2, unit="D"))
        ev = ev.sort_values("event_date")
        gap_days = ev["event_date"].diff().dt.total_seconds() / 86400
        gap_days = gap_days.dropna()

        adm_sorted = adm.sort_values("arrival_datetime").reset_index(drop=True)
        pos = adm_sorted.index[adm_sorted["admission_id"].isin(ev["admission_id"])]
        gap_adm = pd.Series(pos).diff().dropna()

        t_res = spc.t_chart(gap_days, ev["event_date"].iloc[1:])
        g_res = spc.g_chart(gap_adm, ev["event_date"].iloc[1:])

        with ev2:
            m1, m2, m3 = st.columns(3)
            m1.metric("Events", f"{len(ev):,}")
            m2.metric("Median days between", f"{gap_days.median():.0f}")
            m3.metric("Days since the last", f"{(adm['arrival_datetime'].max() - ev['event_date'].max()).days:,}")

        viz.show(viz.control_chart(t_res, title=f"Days between {event.lower()} (t-chart)",
                                   height=320, show_band=True))
        st.caption(
            "Intervals are strongly right-skewed, so 3-sigma limits on the raw "
            "scale would be nonsense. Nelson's transformation y = t^(1/3.6) maps a "
            "Weibull to approximate normality; the XmR chart is fitted on y and "
            "the limits mapped back with y^3.6. One consequence to keep in mind: "
            "the back-transformed centre line is a median-like quantity, not the "
            "arithmetic mean interval.")

        viz.show(viz.control_chart(g_res, title=f"Admissions between {event.lower()} (g-chart)",
                                   height=300))
        st.caption(
            "The g-chart is the better choice when workload varies. Ninety days "
            "between falls means something quite different in a busy month than a "
            "quiet one; counting admissions instead of days removes that "
            "confounding. Its centre line uses the median because the geometric "
            "distribution is skewed enough that the mean sits well above the "
            "typical value.")

# ===========================================================================
with tab_rules:
    st.markdown("### What counts as a signal")
    st.markdown(
        "The 3-sigma limit is not a p-value. Shewhart chose it as an economic "
        "balance between chasing noise and missing real change, not as an "
        "inference threshold — under normality it corresponds to roughly one "
        "false alarm in 370 points, but the limits hold up well beyond the normal "
        "case, which is why they survive the skewed distributions that dominate "
        "clinical timing data.\n\n"
        "Beyond the single-point rule, the run rules detect changes too small to "
        "push any one point outside the limits. Each added rule raises "
        "sensitivity and raises the false-alarm rate: with all eight Nelson "
        "rules active the in-control false-alarm rate is roughly one point in "
        "90, several times the rate of rule 1 alone. **Choose a rule set once and "
        "stay with it.** Switching after seeing the data is how SPC loses its "
        "credibility in a clinical audience.")

    rows = []
    for rid, text in spc.RULE_TEXT.items():
        in_nhs = rid in spc.RULE_SETS["nhs"]["rules"]
        in_nelson = rid in spc.RULE_SETS["nelson"]["rules"]
        rows.append({"Rule": rid, "Detects": text,
                     "NHS Making Data Count": "yes" if in_nhs else "—",
                     "Nelson": "yes" if in_nelson else "—"})
    ui.df(pd.DataFrame(rows), hide_index=True)

    st.markdown(
        "**Rules 6 and 7 are the interesting ones.** Rule 6 fires when fifteen "
        "consecutive points hug the centre line — a process that looks *too* well "
        "behaved. In audit data that almost never means excellence; it means "
        "stratification, a wrongly-defined subgroup, rounding, or data that has "
        "been smoothed somewhere upstream. Rule 7 fires when eight consecutive "
        "points all sit away from the centre with none near it, which is the "
        "signature of two different processes being averaged together — commonly "
        "weekday and weekend, or two sites reported as one.\n\n"
        "Both are data-quality alarms wearing clinical clothing, and both are "
        "worth more attention than the average point-outside-the-limits.")

    st.markdown("**Current settings**")
    cfg = spc.RULE_SETS[f.rule_set]
    st.markdown(
        f"- Rule set: `{f.rule_set}` — rules {cfg['rules']}\n"
        f"- Shift run length: {cfg['run']} consecutive points on one side\n"
        f"- Trend run length: {cfg['trend']} consecutive rising or falling points\n"
        f"- Overdispersion correction: {'on (auto)' if f.laney else 'off'}\n"
        f"- Limits: {'whole series' if f.baseline is None else f'frozen on first {f.baseline} periods'}")
