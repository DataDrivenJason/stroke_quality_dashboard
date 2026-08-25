"""
Shared Streamlit furniture: filters, cards, sparklines, methods panels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from . import loaders, mdc, metrics, spc, viz
from .standards import METRICS, MetricSpec, Threshold, metric

APP_TITLE = "Stroke Quality Intelligence"

CSS = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px;}
  /* Cards flex so the icon row pins to the bottom: titles wrap to one or
     two lines, and without this the icon rows in a row of cards sit at
     different heights and the scorecard stops reading as a grid. */
  .sqi-card {border: 1px solid rgba(11,11,11,0.10); border-radius: 10px;
             padding: 13px 15px 11px 15px; background: var(--background-color);
             display: flex; flex-direction: column; min-height: 226px; height: 100%;}
  .sqi-card .sqi-icons {margin-top: auto;}
  /* Stretch the column wrappers so every card in a row is the same height —
     otherwise a two-line title makes one card taller and the grid breaks. */
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
      display: flex; flex-direction: column;}
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]
      > div[data-testid="stVerticalBlock"] {height: 100%;}
  .sqi-card h4 {font-size: 0.80rem; font-weight: 600; margin: 0 0 6px 0;
                line-height: 1.28; letter-spacing: 0.005em; opacity: 0.88;}
  .sqi-value {font-size: 1.85rem; font-weight: 620; line-height: 1.05;
              letter-spacing: -0.015em;}
  .sqi-unit {font-size: 0.80rem; font-weight: 500; opacity: 0.60; margin-left: 3px;}
  .sqi-sub {font-size: 0.72rem; opacity: 0.62; margin-top: 3px;}
  .sqi-icons {display: flex; gap: 14px; align-items: center; margin-top: 12px;
              padding-top: 9px; border-top: 1px solid rgba(11,11,11,0.07);
              min-height: 44px;}
  .sqi-icon {display: flex; gap: 7px; align-items: center;}
  .sqi-icon span {font-size: 0.70rem; line-height: 1.2; opacity: 0.82;}
  .sqi-spark {margin-top: 8px;}
  .sqi-note {font-size: 0.78rem; opacity: 0.72; border-left: 3px solid rgba(11,11,11,0.16);
             padding: 2px 0 2px 11px; margin: 8px 0;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
"""


# ---------------------------------------------------------------------------
@dataclass
class Filters:
    standard: str = "SSNAP"
    sites: tuple[str, ...] = ()
    date_from: pd.Timestamp | None = None
    date_to: pd.Timestamp | None = None
    freq: str = "MS"
    rule_set: str = "nhs"
    laney: bool | str = "auto"
    baseline: int | None = None
    overrides: dict[str, float] = field(default_factory=dict)

    @property
    def token(self) -> str:
        return "|".join(map(str, [self.standard, self.sites, self.date_from,
                                  self.date_to, self.freq, self.rule_set,
                                  self.laney, self.baseline]))


def setup(page_title: str, icon: str = "◔") -> None:
    """Per-page setup. Page config belongs to the router in app.py; calling
    it again here would raise, so it is only attempted when a view is run
    directly (handy for developing one page in isolation)."""
    try:
        st.set_page_config(page_title=f"{page_title} · {APP_TITLE}",
                           page_icon=icon, layout="wide",
                           initial_sidebar_state="expanded")
    except Exception:
        pass
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
def df(data, **kw):
    """st.dataframe that spans its container on old and new Streamlit alike.

    Streamlit 1.49 replaced ``use_container_width=True`` with
    ``width="stretch"``. Passing the new form to an older build raises
    ``TypeError: 'str' object cannot be interpreted as an integer``, and
    passing the old form to a new build prints a deprecation warning.
    Rather than pinning a high floor -- users on a conda environment they
    did not choose are a real constituency -- try the new API and fall
    back once.
    """
    kw.pop("width", None)
    try:
        return st.dataframe(data, width="stretch", **kw)
    except TypeError:
        return st.dataframe(data, use_container_width=True, **kw)


def get_tables() -> dict[str, pd.DataFrame]:
    if "tables" not in st.session_state:
        with st.spinner("Generating the demonstration dataset…"):
            st.session_state["tables"] = loaders.prepare(loaders.cached_synthetic())
        st.session_state["source"] = "Simulated demonstration data"
    return st.session_state["tables"]


def sidebar() -> Filters:
    tables = get_tables()
    adm = tables["admissions"]
    all_sites = sorted(adm["site"].unique().tolist())
    lo = adm["arrival_datetime"].min().date()
    hi = adm["arrival_datetime"].max().date()

    with st.sidebar:
        st.markdown(f"### {APP_TITLE}")
        st.caption(st.session_state.get("source", "—"))

        standard = st.radio(
            "Audit standard", ["SSNAP", "INAS"], key="f_standard", horizontal=True,
            help=("Switches indicator labels and targets. The underlying "
                  "calculations are identical — only the thresholds and some "
                  "labels change, which is exactly the separation the registry "
                  "is designed to make explicit."))

        st.markdown("**Cohort**")
        sites = st.multiselect("Site", all_sites, default=all_sites, key="f_sites")
        dates = st.date_input("Arrival date range", value=(lo, hi),
                              min_value=lo, max_value=hi, key="f_dates")
        freq_label = st.radio("Reporting period", ["Month", "Quarter"],
                              key="f_freq", horizontal=True)

        with st.expander("Control chart settings"):
            rule_label = st.selectbox(
                "Special-cause rule set",
                ["NHS Making Data Count (4 rules, runs of 7)",
                 "Nelson (8 rules, runs of 9)",
                 "Rule 1 only (points outside the limits)"],
                key="f_rules",
                help=("More rules find more signals and produce more false "
                      "alarms. Choose once and stay with it — switching rule "
                      "sets after seeing the data is how SPC loses credibility."))
            rule_set = {"NHS Making Data Count (4 rules, runs of 7)": "nhs",
                        "Nelson (8 rules, runs of 9)": "nelson",
                        "Rule 1 only (points outside the limits)": "core"}[rule_label]

            laney_on = st.checkbox(
                "Laney overdispersion correction", value=True, key="f_laney",
                help=("Applies the p′ / u′ correction when the scatter between "
                      "periods exceeds what binomial or Poisson sampling can "
                      "produce. Essential for indicators with large "
                      "denominators, such as percentage of days with therapy, "
                      "where thousands of patient-days per month otherwise "
                      "collapse the limits onto the centre line."))

            base_mode = st.selectbox(
                "Limit calculation",
                ["Whole series", "Freeze on first 12 periods",
                 "Freeze on first 18 periods"], key="f_baseline",
                help=("Freezing limits on a baseline and extending them forward "
                      "keeps the reference level: the chart can then say 'this "
                      "is outside what the process used to do'. Whole-series "
                      "limits put the centre line between the old and new "
                      "levels, so the run rules flag everything and the chart "
                      "loses the comparison that made it useful."))
            baseline = {"Whole series": None, "Freeze on first 12 periods": 12,
                        "Freeze on first 18 periods": 18}[base_mode]

        st.divider()
        st.caption("Simulated data for demonstration. Targets are working "
                   "defaults — verify against current audit guidance before "
                   "external use.")

    if isinstance(dates, (list, tuple)) and len(dates) == 2:
        d_from, d_to = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
    else:
        d_from, d_to = pd.Timestamp(lo), pd.Timestamp(hi)

    return Filters(
        standard=standard, sites=tuple(sites or all_sites),
        date_from=d_from, date_to=d_to,
        freq="MS" if freq_label == "Month" else "QS",
        rule_set=rule_set, laney=("auto" if laney_on else False),
        baseline=baseline,
        overrides=st.session_state.get("target_overrides", {}),
    )


def filtered(f: Filters) -> dict[str, pd.DataFrame]:
    tables = get_tables()
    adm = tables["admissions"]
    mask = adm["site"].isin(list(f.sites))
    if f.date_from is not None:
        mask &= adm["arrival_datetime"] >= f.date_from
    if f.date_to is not None:
        mask &= adm["arrival_datetime"] < f.date_to + pd.Timedelta(days=1)
    adm = adm[mask]
    ses = tables["sessions"]
    # Clip sessions to the same window, not just to the selected admissions.
    # Without this, patients admitted at the end of the range contribute
    # therapy days *after* it, producing a partial trailing period whose
    # tiny denominator collapses every session-derived chart to near zero.
    ses = ses[ses["admission_id"].isin(adm["admission_id"])]
    if f.date_from is not None:
        ses = ses[ses["date"] >= f.date_from]
    if f.date_to is not None:
        ses = ses[ses["date"] <= f.date_to]
    staff = tables["staffing"]
    staff = staff[staff["site"].isin(list(f.sites))]
    return {"admissions": adm, "sessions": ses, "staffing": staff,
            "therapists": tables["therapists"][
                tables["therapists"]["site"].isin(list(f.sites))]}


def period_table(f: Filters, group_col: str | None = None) -> pd.DataFrame:
    """Memoised in session state — rebuilding is cheap but not free."""
    cache_key = f"pt::{f.token}::{group_col}"
    if cache_key not in st.session_state:
        data = filtered(f)
        st.session_state[cache_key] = metrics.build_period_table(
            data["admissions"], data["sessions"], freq=f.freq, group_col=group_col)
    return st.session_state[cache_key]


def series(tbl: pd.DataFrame, key: str, f: Filters) -> spc.ChartResult:
    return metrics.metric_series(tbl, key, standard=f.standard,
                                 overrides=f.overrides, rule_set=f.rule_set,
                                 laney=f.laney, baseline=f.baseline)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def fmt(value: float | None, unit: str) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    if unit == "%":
        return f"{value:.1f}"
    if unit in ("minutes", "minutes/day"):
        return f"{value:.0f}"
    if unit == "hours":
        return f"{value:.1f}"
    if unit == "days":
        return f"{value:.1f}"
    return f"{value:,.2f}"


def unit_suffix(unit: str) -> str:
    return {"%": "%", "minutes": "min", "minutes/day": "min/day",
            "hours": "hrs", "days": "days"}.get(unit, unit)


# ---------------------------------------------------------------------------
# Sparkline -- hand-built SVG rather than a chart library, because a
# scorecard renders a dozen of these and a dozen Plotly instances is a
# visible pause on every rerun.
# ---------------------------------------------------------------------------
def sparkline_svg(res: spc.ChartResult, width: int = 260, height: int = 42,
                  n: int = 18) -> str:
    p = viz.palette()
    f = res.frame.tail(n)
    v = f["value"].to_numpy(dtype=float)
    if np.isfinite(v).sum() < 2:
        return ""
    lo_s = np.nanmin([np.nanmin(v), np.nanmin(f["lcl"])])
    hi_s = np.nanmax([np.nanmax(v), np.nanmax(f["ucl"])])
    if res.target is not None and np.isfinite(res.target):
        lo_s, hi_s = min(lo_s, res.target), max(hi_s, res.target)
    span = (hi_s - lo_s) or 1.0
    pad = 5

    def sy(val: float) -> float:
        return height - pad - (val - lo_s) / span * (height - 2 * pad)

    xs = np.linspace(pad, width - pad, len(v))
    parts = []

    # expected-range band
    band = " ".join(
        f"{x:.1f},{sy(u):.1f}" for x, u in zip(xs, f["ucl"]) if np.isfinite(u))
    band_lo = " ".join(
        f"{x:.1f},{sy(l):.1f}" for x, l in zip(xs[::-1], f["lcl"][::-1]) if np.isfinite(l))
    if band and band_lo:
        parts.append(f'<polygon points="{band} {band_lo}" fill="{p["band"]}"/>')

    if res.target is not None and np.isfinite(res.target):
        ty = sy(res.target)
        parts.append(f'<line x1="{pad}" y1="{ty:.1f}" x2="{width-pad}" y2="{ty:.1f}" '
                     f'stroke="{p["target"]}" stroke-width="1.2" stroke-dasharray="3 3"/>')

    pts = [(x, sy(val)) for x, val in zip(xs, v) if np.isfinite(val)]
    if len(pts) >= 2:
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        parts.append(f'<path d="{d}" fill="none" stroke="{p["series"][0]}" '
                     f'stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round"/>')

    hib = res.higher_is_better
    for (x, y), (_, row) in zip(pts, f[np.isfinite(f["value"])].iterrows()):
        if not row["special"]:
            continue
        col = (mdc.NEUTRAL if hib is None
               else (mdc.GOOD if (row["special_dir"] > 0) == bool(hib) else mdc.CRITICAL))
        parts.append(f'<rect x="{x-2.6:.1f}" y="{y-2.6:.1f}" width="5.2" height="5.2" '
                     f'transform="rotate(45 {x:.1f} {y:.1f})" fill="{col}" '
                     f'stroke="{p["surface"]}" stroke-width="1"/>')
    if pts:
        x, y = pts[-1]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{p["series"][0]}" '
                     f'stroke="{p["surface"]}" stroke-width="1.4"/>')

    return (f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" '
            f'preserveAspectRatio="none" role="img" aria-label="Trend sparkline">'
            + "".join(parts) + "</svg>")


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------
def metric_card(res: spc.ChartResult, tbl: pd.DataFrame, key: str,
                f: Filters, spark: bool = True) -> mdc.MDCVerdict:
    spec, thr = metric(key, f.standard, f.overrides)
    v = mdc.verdict(res)
    value = metrics.current_value(tbl, key, periods=3 if f.freq == "MS" else 1)

    target_txt = ("no target" if thr.target is None
                  else f"target {fmt(thr.target, spec.unit)}{unit_suffix(spec.unit)}")
    window = "last 3 months" if f.freq == "MS" else "last quarter"

    html = (
        '<div class="sqi-card">'
        f'<h4>{spec.label}</h4>'
        f'<div class="sqi-value">{fmt(value, spec.unit)}'
        f'<span class="sqi-unit">{unit_suffix(spec.unit)}</span></div>'
        f'<div class="sqi-sub">{window} · {target_txt}</div>'
        + (f'<div class="sqi-spark">{sparkline_svg(res)}</div>' if spark else "")
        + '<div class="sqi-icons">'
        f'<div class="sqi-icon">{mdc.variation_icon(v.variation, 26)}'
        f'<span>{v.variation_label}</span></div>'
        f'<div class="sqi-icon">{mdc.assurance_icon(v.assurance, 26)}'
        f'<span>{v.assurance_label}</span></div>'
        '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)
    return v


def verdict_banner(v: mdc.MDCVerdict, res: spc.ChartResult | None = None) -> None:
    st.markdown(mdc.icon_pair_html(v, 30), unsafe_allow_html=True)
    st.markdown(f'<div class="sqi-note"><b>{v.variation_label}.</b> {v.variation_detail}<br>'
                f'<b>{v.assurance_label}.</b> {v.assurance_detail}<br>'
                f'<b>So:</b> {v.action}</div>', unsafe_allow_html=True)

    # The icons describe the LATEST period, which is the NHS convention and
    # the right default for a board report. But a chart can be peppered with
    # historical signals while the current point is unremarkable, and a
    # reader looking at the diamonds will not believe a grey icon unless the
    # discrepancy is named.
    if res is not None:
        n_sig = len(res.signals)
        if n_sig and not v.rules_fired:
            st.caption(
                f"The icons describe the most recent period only. There are "
                f"**{n_sig}** special-cause points earlier in this series — the "
                f"process has changed before, it just is not changing right now. "
                f"See the signals table.")
        if res.meta.get("degenerate"):
            st.warning(res.meta.get("degenerate_note", "Zero variation — limits undefined."))


# ---------------------------------------------------------------------------
def methods_block(res: spc.ChartResult, key: str, f: Filters,
                  expanded: bool = False) -> None:
    """The 'how this is calculated' panel that sits under every chart."""
    spec, thr = metric(key, f.standard, f.overrides)
    m = res.meta
    with st.expander("How this indicator is built", expanded=expanded):
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**Definition.** {spec.definition}")
            st.markdown(f"**Why it matters.** {spec.why}")
            st.markdown(f"**Read it with care.** {spec.caveat}")
        with c2:
            chart_names = {"xmr": "XmR (individuals and moving range)",
                           "p": "p-chart (binomial limits)",
                           "p-prime": "p′-chart (Laney overdispersion correction)",
                           "u": "u-chart (Poisson limits)",
                           "u-prime": "u′-chart (Laney overdispersion correction)",
                           "c": "c-chart", "t": "t-chart", "g": "g-chart"}
            rows = [
                ("Chart", chart_names.get(res.chart_type, res.chart_type)),
                ("Sigma estimated from", m.get("sigma_estimator", "—")),
                ("Rule set", {"nhs": "NHS Making Data Count (4 rules)",
                              "nelson": "Nelson (8 rules)",
                              "core": "Rule 1 only"}[m.get("rule_set", "nhs")]),
                ("Limits", "Whole series" if f.baseline is None
                 else f"Frozen on first {f.baseline} periods, extended forward"),
                ("Periods plotted", f"{res.n_points}"),
                ("Signals", f"{len(res.signals)}"),
            ]
            if "sigma_z" in m:
                sz = m["sigma_z"]
                rows.append(("Dispersion σz", f"{sz:.2f}"
                             + (" — overdispersed, correction applied" if sz > 1.1
                                else " — binomial/Poisson variation sufficient")))
            if thr.target is not None:
                rows.append(("Target", f"{fmt(thr.target, spec.unit)}"
                                       f"{unit_suffix(spec.unit)} "
                                       f"({'higher' if spec.higher_is_better else 'lower'} is better)"))
                if thr.ambition is not None:
                    rows.append(("Stretch", f"{fmt(thr.ambition, spec.unit)}"
                                            f"{unit_suffix(spec.unit)}"))
            df(pd.DataFrame(rows, columns=["", "Value"]),
                         hide_index=True)
            if thr.provenance:
                st.caption(f"Target provenance: {thr.provenance}")

        if res.chart_type == "p":
            st.markdown(
                "**The maths.** The centre line is the pooled proportion "
                r"$\bar p = \sum d_i / \sum n_i$ — periods are weighted by their own "
                "denominators, so a quiet month does not carry the same influence as a busy one. "
                r"Limits are $\bar p \pm 3\sqrt{\bar p(1-\bar p)/n_i}$ and therefore "
                "breathe with the monthly denominator: fewer patients, wider limits.")
        elif res.chart_type in ("p-prime", "u-prime"):
            sz = m.get("sigma_z", 1.0)
            st.markdown(
                "**The maths.** Binomial (or Poisson) limits assume every case in a period is "
                "an independent trial with the same underlying probability. Standardising the "
                r"points, $z_i = (p_i - \bar p)/\sqrt{\bar p(1-\bar p)/n_i}$, that assumption "
                r"predicts short-term variation of exactly 1. Here it measures "
                f"**σz = {sz:.2f}**, estimated Shewhart-style from the moving range of z "
                r"($\overline{mR_z}/1.128$). Limits are widened by that factor. "
                "σz above 1 is itself the finding: the periods are not exchangeable, and the "
                "question to ask is what differs between them — case mix, coding, or a real "
                "difference in the process.")
        elif res.chart_type == "xmr":
            st.markdown(
                "**The maths.** Sigma comes from the average moving range, "
                r"$\hat\sigma = \overline{mR}/1.128$, not from the standard deviation of all "
                "points. This matters: the SD of the whole series is inflated by any shift or "
                "trend inside it, so limits built on it widen just enough to hide the signal "
                "you were looking for. The moving range only sees adjacent points, which a "
                r"step change cannot contaminate. Limits are $\bar x \pm 2.66\,\overline{mR}$.")
        elif res.chart_type == "u":
            st.markdown(
                "**The maths.** Poisson limits, "
                r"$\bar u \pm 3\sqrt{\bar u / n_i}$, where $n_i$ is the exposure "
                "(bed days), not the number of admissions. Using admissions would make a "
                "longer average stay read as a worse safety record.")


def signal_table(res: spc.ChartResult) -> None:
    sig = res.signals
    if sig.empty:
        st.caption("No special-cause signals in the plotted range.")
        return
    out = sig[["x", "value", "cl", "lcl", "ucl", "rule_text"]].copy()
    out["x"] = pd.to_datetime(out["x"]).dt.strftime("%b %Y")
    out.columns = ["Period", "Value", "Process mean", "Lower limit",
                   "Upper limit", "Rule broken"]
    df(out.round(2), hide_index=True)


def table_view(res: spc.ChartResult, label: str = "Table view") -> None:
    """Every chart has a table twin. Colour and position are never the
    only route to a value."""
    with st.expander(label):
        out = res.frame[["x", "value", "cl", "lcl", "ucl", "special", "rule_text"]].copy()
        if "numerator" in res.frame.columns:
            out.insert(1, "denominator", res.frame["denominator"])
            out.insert(1, "numerator", res.frame["numerator"])
        out["x"] = pd.to_datetime(out["x"]).dt.strftime("%Y-%m")
        out = out.rename(columns={"x": "Period", "value": "Value", "cl": "Process mean",
                                  "lcl": "Lower limit", "ucl": "Upper limit",
                                  "special": "Special cause", "rule_text": "Rule",
                                  "numerator": "Numerator", "denominator": "Denominator"})
        df(out.round(2), hide_index=True)
        st.download_button("Download this indicator as CSV",
                           out.to_csv(index=False).encode(),
                           file_name=f"{res.label or 'indicator'}.csv",
                           mime="text/csv", key=f"dl_{res.label}_{id(res)}")


def metric_picker(keys: list[str], f: Filters, label: str = "Indicator",
                  key: str = "picker", default: str | None = None) -> str:
    labels = {k: metric(k, f.standard)[0].label for k in keys}
    options = list(labels.keys())
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index,
                        format_func=lambda k: labels[k], key=key)
