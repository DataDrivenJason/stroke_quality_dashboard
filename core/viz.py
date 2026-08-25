"""
Plotly chart builders and the visual system.
=====================================================================

Design rules applied throughout (and why):

* **Colour never carries meaning alone.** Special-cause points differ
  from common-cause points by *shape* (diamond vs circle) and by size as
  well as colour, and every one is labelled in the tooltip with the rule
  it broke. A reader with deuteranopia, a black-and-white printout and a
  projector with the contrast turned up all still work.

* **Control limits are dashed, the centre line is solid.** This is the
  NHS "Making Data Count" convention and clinicians read it fluently.
  Gridlines are solid hairlines -- dashing them too would make the plot
  ambiguous about which dashes mean something.

* **One y-axis, always.** Two scales on one plot invents a relationship
  that is not in the data. Where two measures must be compared, they get
  two stacked charts sharing an x-axis.

* **The band between the limits is shaded faintly.** The single most
  common misreading of a control chart is treating the limits as a
  target corridor. Shading the region and labelling it "expected range"
  in the tooltip pushes readers toward the right reading: this is what
  the process does, not what we want it to do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .mdc import CRITICAL, GOOD, NEUTRAL, SERIOUS, WARNING
from .spc import RULE_TEXT, ChartResult

# ---------------------------------------------------------------------------
# Palette -- the reference data-viz instance. Swap these values for a
# brand palette and re-run the validator; nothing else changes.
# ---------------------------------------------------------------------------
LIGHT = {
    "surface": "#fcfcfb", "page": "#f9f9f7",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "band": "rgba(42,120,214,0.055)",
    "target": "#4a3aa7",
}
DARK = {
    "surface": "#1a1a19", "page": "#0d0d0d",
    "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835",
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"],
    "band": "rgba(57,135,229,0.09)",
    "target": "#9085e9",
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def fmt_num(v: float) -> str:
    """Compact label for a direct annotation.

    Deliberately not a Plotly d3 format string: annotation text is rendered
    by Python, and d3's tilde-trim syntax (``.3~g``) is a ValueError there.
    Precision scales with magnitude, so 1,240 / 62.4 / 0.83 all read
    cleanly without a format argument per call site.
    """
    if v is None or not np.isfinite(v):
        return "—"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return f"{v:.2f}".rstrip("0").rstrip(".")


def palette() -> dict:
    """Follow the Streamlit theme when the running version exposes it."""
    try:  # Streamlit >= 1.46
        import streamlit as st
        if getattr(st.context, "theme", None) and st.context.theme.type == "dark":
            return DARK
    except Exception:
        pass
    return LIGHT


def _layout(p: dict, height: int, title: str = "", ytitle: str = "",
            xtitle: str = "") -> dict:
    """Shared layout.

    Two things here are load-bearing rather than cosmetic. The title key is
    omitted entirely when empty -- passing ``title=None`` makes Plotly render
    the literal string "undefined". And the top margin has to hold the title
    *and* the legend row; sized for the title alone they overlap, which is
    the most common way a chart in a dashboard ends up unreadable.
    """
    lay = dict(
        height=height,
        margin=dict(l=10, r=16, t=(78 if title else 40), b=34),
        paper_bgcolor=p["surface"], plot_bgcolor=p["surface"],
        font=dict(family=FONT, size=12.5, color=p["ink2"]),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=p["surface"], font=dict(family=FONT, size=12,
                                                       color=p["ink"]),
                        bordercolor=p["axis"]),
        xaxis=dict(title=dict(text=xtitle, font=dict(size=11.5, color=p["muted"])),
                   showgrid=False, zeroline=False,
                   linecolor=p["axis"], linewidth=1, ticks="outside",
                   tickcolor=p["axis"], ticklen=4,
                   tickfont=dict(size=11, color=p["muted"])),
        yaxis=dict(title=dict(text=ytitle, font=dict(size=11.5, color=p["muted"])),
                   gridcolor=p["grid"], gridwidth=1, griddash="solid",
                   zeroline=False, linecolor=p["axis"], linewidth=1,
                   tickfont=dict(size=11, color=p["muted"])),
        legend=dict(orientation="h", yanchor="bottom", y=1.015, xanchor="left", x=0,
                    font=dict(size=11, color=p["ink2"]), bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
    )
    if title:
        lay["title"] = dict(text=title, font=dict(size=15, color=p["ink"]),
                            x=0, xanchor="left", y=0.985, yanchor="top",
                            yref="container")
    return lay


def show(fig, **kw):
    """Render a figure, tolerating the Streamlit width API change."""
    import streamlit as st
    try:
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, **kw)
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False}, **kw)


# ---------------------------------------------------------------------------
# Control chart
# ---------------------------------------------------------------------------
def control_chart(res: ChartResult, *, height: int = 400, title: str = "",
                  show_target: bool = True, show_band: bool = True,
                  annotate_signals: bool = True) -> go.Figure:
    p = palette()
    f = res.frame.copy()
    x = pd.to_datetime(f["x"]) if np.issubdtype(np.asarray(f["x"]).dtype, np.datetime64) else f["x"]

    fig = go.Figure()

    # -- expected range band ------------------------------------------------
    if show_band:
        fig.add_trace(go.Scatter(x=x, y=f["ucl"], mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False, name="ucl-fill"))
        fig.add_trace(go.Scatter(x=x, y=f["lcl"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=p["band"],
                                 hoverinfo="skip", showlegend=False, name="Expected range"))

    # -- limits and centre line --------------------------------------------
    for col, nm in (("ucl", "Upper process limit"), ("lcl", "Lower process limit")):
        fig.add_trace(go.Scatter(
            x=x, y=f[col], mode="lines", name=nm,
            line=dict(color=p["muted"], width=1.4, dash="dash"),
            legendgroup="limits", showlegend=(col == "ucl"),
            hovertemplate="%{y:.4~f}<extra>" + nm + "</extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=f["cl"], mode="lines", name="Process mean",
        line=dict(color=p["ink2"], width=1.6),
        hovertemplate="%{y:.4~f}<extra>Process mean</extra>"))

    # -- target -------------------------------------------------------------
    if show_target and res.target is not None:
        fig.add_trace(go.Scatter(
            x=x, y=np.full(len(f), res.target), mode="lines", name="Target",
            line=dict(color=p["target"], width=1.8, dash="dot"),
            hovertemplate="%{y:.4~f}<extra>Target</extra>"))

    # -- the data -----------------------------------------------------------
    hib = res.higher_is_better
    colours, symbols, sizes, notes = [], [], [], []
    for _, row in f.iterrows():
        if not row["special"]:
            colours.append(p["series"][0]); symbols.append("circle")
            sizes.append(7.5); notes.append("Common cause")
            continue
        direction = row["special_dir"]
        if hib is None:
            col = NEUTRAL
        else:
            col = GOOD if ((direction > 0) == bool(hib)) else CRITICAL
        colours.append(col); symbols.append("diamond"); sizes.append(12)
        notes.append("; ".join(RULE_TEXT[r] for r in row["rules"]))

    custom = np.stack([f["cl"], f["lcl"], f["ucl"], np.array(notes, dtype=object)], axis=-1)
    fig.add_trace(go.Scatter(
        x=x, y=f["value"], mode="lines+markers", name=res.label or "Value",
        line=dict(color=p["series"][0], width=2),
        marker=dict(color=colours, size=sizes, symbol=symbols,
                    line=dict(color=p["surface"], width=2)),
        customdata=custom,
        hovertemplate=("<b>%{y:.4~f}</b> " + (res.unit or "") +
                       "<br>Expected %{customdata[1]:.3~f}–%{customdata[2]:.3~f}"
                       "<br>Mean %{customdata[0]:.3~f}"
                       "<br>%{customdata[3]}<extra></extra>")))

    # -- phase break markers -------------------------------------------------
    if f["phase"].nunique() > 1:
        for ph in sorted(f["phase"].unique())[1:]:
            first = f.index[f["phase"] == ph][0]
            fig.add_vline(x=x.iloc[first], line=dict(color=p["axis"], width=1.2, dash="dot"),
                          annotation_text="limits recalculated",
                          annotation_font=dict(size=10, color=p["muted"]),
                          annotation_position="top left")

    # -- selective direct labels: only the signals ---------------------------
    if annotate_signals:
        sig = f[f["special"]]
        if 0 < len(sig) <= 8:
            # Label selectively. Consecutive signalling points sit close
            # enough that their labels collide into unreadable runs like
            # "53.653.8", which is worse than no label at all. Require a
            # gap of at least two periods since the last label, and stop
            # at four -- the axis and the tooltip carry the rest.
            last_labelled = -99
            labelled = 0
            for idx, row in sig.iterrows():
                if idx - last_labelled < 2 or labelled >= 4:
                    continue
                fig.add_annotation(
                    x=pd.to_datetime(row["x"]) if hasattr(row["x"], "year") else row["x"],
                    y=row["value"], text=fmt_num(row["value"]),
                    showarrow=False, yshift=16 if row["special_dir"] > 0 else -16,
                    font=dict(size=10.5, color=p["ink2"]))
                last_labelled = idx
                labelled += 1

    lay = _layout(p, height, title, ytitle=res.unit)
    fig.update_layout(**lay)
    return fig


def mr_chart(res: ChartResult, *, height: int = 170) -> go.Figure:
    """The moving-range companion.

    Always show it alongside the individuals chart. The X chart tells you
    whether the level changed; the mR chart tells you whether the
    *variability* changed. A stable mean with an exploding moving range
    is a service becoming unpredictable, and the X chart alone will not
    show it -- worse, the widening limits will hide points that would
    otherwise have signalled.
    """
    p = palette()
    f = res.frame
    if "mr" not in f.columns:
        return go.Figure()
    x = pd.to_datetime(f["x"]) if np.issubdtype(np.asarray(f["x"]).dtype, np.datetime64) else f["x"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=f["mr_ucl"], mode="lines", name="Upper limit",
                             line=dict(color=p["muted"], width=1.2, dash="dash"),
                             hovertemplate="%{y:.3~f}<extra>mR upper limit</extra>"))
    fig.add_trace(go.Scatter(x=x, y=f["mr_cl"], mode="lines", name="Mean moving range",
                             line=dict(color=p["ink2"], width=1.4),
                             hovertemplate="%{y:.3~f}<extra>Mean moving range</extra>"))
    breach = f["mr"] > f["mr_ucl"]
    fig.add_trace(go.Scatter(
        x=x, y=f["mr"], mode="lines+markers", name="Moving range",
        line=dict(color=p["series"][2], width=1.6),
        marker=dict(size=np.where(breach, 11, 6),
                    symbol=np.where(breach, "diamond", "circle"),
                    color=np.where(breach, CRITICAL, p["series"][2]),
                    line=dict(color=p["surface"], width=1.5)),
        hovertemplate="%{y:.3~f}<extra>Moving range</extra>"))
    lay = _layout(p, height, "", ytitle="Moving range")
    lay["margin"] = dict(l=10, r=16, t=6, b=26)
    lay["showlegend"] = False
    fig.update_layout(**lay)
    return fig


def cusum_chart(cs: pd.DataFrame, *, height: int = 230, title: str = "") -> go.Figure:
    p = palette()
    x = pd.to_datetime(cs["x"]) if np.issubdtype(np.asarray(cs["x"]).dtype, np.datetime64) else cs["x"]
    h = float(cs["h"].iloc[0])
    fig = go.Figure()
    for y, nm in ((np.full(len(cs), h), "Decision interval"),
                  (np.full(len(cs), -h), None)):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=nm or "",
                                 line=dict(color=p["muted"], width=1.3, dash="dash"),
                                 showlegend=nm is not None, legendgroup="h",
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x, y=cs["cusum_high"], mode="lines+markers", name="Accumulating high",
        line=dict(color=p["series"][1], width=2),
        marker=dict(size=np.where(cs["signal_high"], 11, 5),
                    symbol=np.where(cs["signal_high"], "diamond", "circle"),
                    color=np.where(cs["signal_high"], CRITICAL, p["series"][1]),
                    line=dict(color=p["surface"], width=1.5)),
        hovertemplate="%{y:.2f}<extra>Accumulating above expectation</extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=cs["cusum_low"], mode="lines+markers", name="Accumulating low",
        line=dict(color=p["series"][0], width=2),
        marker=dict(size=np.where(cs["signal_low"], 11, 5),
                    symbol=np.where(cs["signal_low"], "diamond", "circle"),
                    color=np.where(cs["signal_low"], CRITICAL, p["series"][0]),
                    line=dict(color=p["surface"], width=1.5)),
        hovertemplate="%{y:.2f}<extra>Accumulating below expectation</extra>"))
    fig.update_layout(**_layout(p, height, title, ytitle="Cumulative sum"))
    return fig


# ---------------------------------------------------------------------------
# Funnel plot
# ---------------------------------------------------------------------------
def funnel_chart(fn: dict, *, height: int = 420, title: str = "",
                 unit: str = "%") -> go.Figure:
    p = palette()
    grid = fn["grid"]
    fig = go.Figure()

    for band, dash, nm in (("998", "dash", "99.8% limits"), ("95", "dot", "95% limits")):
        lo, hi = fn["bands"][band]
        fig.add_trace(go.Scatter(x=grid, y=hi, mode="lines", name=nm,
                                 line=dict(color=p["muted"], width=1.3, dash=dash),
                                 legendgroup=band, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=grid, y=lo, mode="lines", name=nm,
                                 line=dict(color=p["muted"], width=1.3, dash=dash),
                                 legendgroup=band, showlegend=False, hoverinfo="skip"))

    fig.add_hline(y=fn["centre"], line=dict(color=p["ink2"], width=1.6),
                  annotation_text=f"pooled {fn['centre']:.1f}{unit}",
                  annotation_font=dict(size=10.5, color=p["muted"]),
                  annotation_position="right")

    pts = fn["points"]
    colour_map = {"within": p["series"][0], "above 95%": WARNING, "below 95%": WARNING,
                  "above 99.8%": CRITICAL, "below 99.8%": CRITICAL}
    colours = [colour_map[s] for s in pts["status"]]
    symbols = ["circle" if s == "within" else "diamond" for s in pts["status"]]
    sizes = [11 if s == "within" else 15 for s in pts["status"]]

    fig.add_trace(go.Scatter(
        x=pts["denominator"], y=pts["value"], mode="markers+text", name="Unit",
        text=pts["label"], textposition="top center",
        textfont=dict(size=10.5, color=p["ink2"]),
        marker=dict(color=colours, size=sizes, symbol=symbols,
                    line=dict(color=p["surface"], width=2)),
        customdata=np.stack([pts["numerator"], pts["denominator"], pts["status"]], axis=-1),
        hovertemplate=("<b>%{text}</b><br>%{y:.1f}" + unit +
                       "<br>%{customdata[0]:.0f} of %{customdata[1]:.0f}"
                       "<br>%{customdata[2]}<extra></extra>")))

    lay = _layout(p, height, title, ytitle=unit, xtitle="Number of patients (denominator)")
    lay["hovermode"] = "closest"
    fig.update_layout(**lay)
    return fig


# ---------------------------------------------------------------------------
# Supporting forms
# ---------------------------------------------------------------------------
def pareto(labels, values, *, height: int = 330, title: str = "",
           ytitle: str = "Sessions") -> go.Figure:
    """Pareto of missed-session reasons.

    Deliberately *not* the textbook dual-axis Pareto. The cumulative line
    on a second 0-100% axis is the classic dual-scale mistake; the same
    information goes in the tooltip and in the annotation on the bar
    where cumulative share crosses 80%.
    """
    p = palette()
    order = np.argsort(values)[::-1]
    labels = np.asarray(labels)[order]
    values = np.asarray(values, dtype=float)[order]
    cum = 100 * np.cumsum(values) / max(values.sum(), 1)
    cross = int(np.argmax(cum >= 80)) if (cum >= 80).any() else len(cum) - 1

    fig = go.Figure(go.Bar(
        x=labels, y=values, name=ytitle,
        marker=dict(color=[p["series"][0] if i <= cross else p["series"][0]
                           for i in range(len(labels))],
                    opacity=[1.0 if i <= cross else 0.42 for i in range(len(labels))],
                    line=dict(color=p["surface"], width=2)),
        customdata=cum,
        hovertemplate="<b>%{y:,.0f}</b> " + ytitle.lower() +
                      "<br>%{customdata:.0f}% cumulative<extra>%{x}</extra>"))
    fig.add_annotation(x=labels[cross], y=values[cross],
                       text=f"{cum[cross]:.0f}% of all missed sessions up to here",
                       showarrow=False, yshift=18,
                       font=dict(size=10.5, color=p["ink2"]))
    lay = _layout(p, height, title, ytitle=ytitle)
    lay["showlegend"] = False
    lay["xaxis"]["tickangle"] = -18
    fig.update_layout(**lay)
    return fig


def stacked_shift(counts: pd.DataFrame, group_col: str, *, height: int = 300,
                  title: str = "") -> go.Figure:
    """mRS 0-6 shift plot ('Grotta bars'), ordinal ramp, one hue."""
    p = palette()
    ramp = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"] \
        if p is LIGHT else ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
    labels = ["0 No symptoms", "1 No significant disability", "2 Slight disability",
              "3 Moderate disability", "4 Moderately severe", "5 Severe", "6 Died"]
    fig = go.Figure()
    for m in range(7):
        sub = counts[counts["mrs_discharge"] == m]
        fig.add_trace(go.Bar(
            y=sub[group_col], x=sub["pct"], orientation="h", name=labels[m],
            marker=dict(color=ramp[m], line=dict(color=p["surface"], width=2)),
            customdata=sub["n"],
            hovertemplate="<b>" + labels[m] + "</b><br>%{x:.1f}% "
                          "(%{customdata:,.0f} patients)<extra>%{y}</extra>"))
    lay = _layout(p, height, title, xtitle="% of patients")
    lay["barmode"] = "stack"
    lay["yaxis"]["gridcolor"] = "rgba(0,0,0,0)"
    lay["xaxis"]["showgrid"] = True
    lay["xaxis"]["gridcolor"] = p["grid"]
    lay["hovermode"] = "closest"
    lay["legend"]["traceorder"] = "normal"
    fig.update_layout(**lay)
    return fig


def bar(labels, values, *, height: int = 320, title: str = "", ytitle: str = "",
        horizontal: bool = False, target: float | None = None,
        highlight: str | None = None, value_fmt: str = ",.1f") -> go.Figure:
    """One series, one colour. Emphasis by opacity, never by a value ramp."""
    p = palette()
    labels = list(labels)
    opac = [1.0 if (highlight is None or l == highlight) else 0.45 for l in labels]
    common = dict(marker=dict(color=p["series"][0], opacity=opac,
                              line=dict(color=p["surface"], width=2)))
    if horizontal:
        fig = go.Figure(go.Bar(y=labels, x=values, orientation="h", **common,
                               hovertemplate="<b>%{x:" + value_fmt + "}</b><extra>%{y}</extra>"))
    else:
        fig = go.Figure(go.Bar(x=labels, y=values, **common,
                               hovertemplate="<b>%{y:" + value_fmt + "}</b><extra>%{x}</extra>"))
    if target is not None:
        # add_vline takes x only and add_hline takes y only -- passing the
        # unused one as None is a ValueError, not a no-op.
        line = dict(color=p["target"], width=1.8, dash="dot")
        note = dict(annotation_text="target",
                    annotation_font=dict(size=10, color=p["muted"]))
        if horizontal:
            fig.add_vline(x=target, line=line, **note)
        else:
            fig.add_hline(y=target, line=line, **note)
    lay = _layout(p, height, title, ytitle=("" if horizontal else ytitle),
                  xtitle=(ytitle if horizontal else ""))
    lay["showlegend"] = False
    lay["hovermode"] = "closest"
    if horizontal:
        lay["yaxis"]["gridcolor"] = "rgba(0,0,0,0)"
        lay["xaxis"]["showgrid"] = True
        lay["xaxis"]["gridcolor"] = p["grid"]
    fig.update_layout(**lay)
    return fig


def multi_line(df: pd.DataFrame, x: str, y: str, colour: str, *,
               height: int = 340, title: str = "", ytitle: str = "",
               target: float | None = None, order: list | None = None) -> go.Figure:
    """Several entities over time. Colour follows the entity, not its rank,
    so filtering never repaints the survivors."""
    p = palette()
    cats = order or sorted(df[colour].dropna().unique().tolist())
    fig = go.Figure()
    for i, c in enumerate(cats[:8]):
        sub = df[df[colour] == c].sort_values(x)
        fig.add_trace(go.Scatter(
            x=sub[x], y=sub[y], mode="lines+markers", name=str(c),
            line=dict(color=p["series"][i % 8], width=2),
            marker=dict(size=6, line=dict(color=p["surface"], width=1.5)),
            hovertemplate="%{y:,.1f}<extra>" + str(c) + "</extra>"))
    if target is not None:
        fig.add_hline(y=target, line=dict(color=p["target"], width=1.8, dash="dot"),
                      annotation_text="target",
                      annotation_font=dict(size=10, color=p["muted"]))
    fig.update_layout(**_layout(p, height, title, ytitle=ytitle))
    return fig


def heatmap(matrix: pd.DataFrame, *, height: int = 300, title: str = "",
            unit: str = "", zmid: float | None = None) -> go.Figure:
    """Single-hue sequential ramp, light -> dark. Never a rainbow."""
    p = palette()
    scale = [[0.0, "#cde2fb"], [0.25, "#9ec5f4"], [0.5, "#5598e7"],
             [0.75, "#256abf"], [1.0, "#0d366b"]]
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=list(matrix.columns), y=list(matrix.index),
        colorscale=scale, xgap=2, ygap=2, zmid=zmid,
        colorbar=dict(title=dict(text=unit, font=dict(size=11, color=p["muted"])),
                      tickfont=dict(size=10.5, color=p["muted"]),
                      outlinewidth=0, thickness=12, len=0.85),
        hovertemplate="<b>%{z:,.1f}</b> " + unit + "<extra>%{y} · %{x}</extra>"))
    lay = _layout(p, height, title)
    lay["showlegend"] = False
    lay["hovermode"] = "closest"
    lay["yaxis"]["gridcolor"] = "rgba(0,0,0,0)"
    lay["xaxis"]["showgrid"] = False
    fig.update_layout(**lay)
    return fig


def demand_capacity(df: pd.DataFrame, *, height: int = 360, title: str = "") -> go.Figure:
    """Required vs available therapy minutes on one axis.

    Both series are minutes, so they belong on the same scale -- this is
    exactly the case where a dual axis would be tempting and wrong. The
    shortfall is drawn as the shaded area between them, which is the
    number the service actually needs to argue its case.
    """
    p = palette()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["period"], y=df["required_minutes"], mode="lines",
                             name="Minutes required at guideline dose",
                             line=dict(color=p["series"][1], width=2),
                             hovertemplate="%{y:,.0f} min<extra>Required</extra>"))
    fig.add_trace(go.Scatter(x=df["period"], y=df["available_minutes"], mode="lines",
                             name="Clinical minutes available",
                             line=dict(color=p["series"][0], width=2),
                             fill="tonexty", fillcolor="rgba(235,104,52,0.13)",
                             hovertemplate="%{y:,.0f} min<extra>Available</extra>"))
    fig.add_trace(go.Scatter(x=df["period"], y=df["delivered_minutes"], mode="lines",
                             name="Minutes actually delivered",
                             line=dict(color=p["series"][2], width=2, dash="dash"),
                             hovertemplate="%{y:,.0f} min<extra>Delivered</extra>"))
    fig.update_layout(**_layout(p, height, title, ytitle="Therapy minutes per month"))
    return fig
