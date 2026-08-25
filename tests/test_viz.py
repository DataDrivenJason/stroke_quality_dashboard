"""
Rendering smoke tests.
=====================================================================

These exist because of a bug that the statistical tests could not have
caught. ``control_chart`` labels its signals with a Plotly d3 format
string (``.3~g``) evaluated by Python, which is a ValueError -- but the
annotation branch only runs when a chart has between one and six
signals. Charts with none, or with many, rendered fine. Two pages
crashed in the browser while all 48 statistics tests passed.

The lesson generalises: a figure that is built but never rendered is
not tested. So every builder is exercised here across the real data,
every registered metric, and the degenerate inputs (empty, constant,
all-NaN, single point) that a filtered dashboard produces in practice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import loaders, metrics, spc, viz  # noqa: E402
from core.standards import METRICS, THRESHOLDS  # noqa: E402


def _render(fig: go.Figure) -> str:
    """Force full serialisation. Building a Figure is lazy about some
    fields; converting to JSON is what actually exercises them."""
    assert isinstance(fig, go.Figure)
    return fig.to_json()


@pytest.fixture(scope="module")
def tables():
    return loaders.prepare(loaders.cached_synthetic())


@pytest.fixture(scope="module")
def tbl(tables):
    return metrics.build_period_table(tables["admissions"], tables["sessions"])


# ---------------------------------------------------------------------------
def test_fmt_num_handles_every_magnitude():
    assert viz.fmt_num(1240.4) == "1,240"
    assert viz.fmt_num(624.0) == "624"
    assert viz.fmt_num(62.45) == "62.5"
    assert viz.fmt_num(8.20) == "8.2"
    assert viz.fmt_num(0.834) == "0.83"
    assert viz.fmt_num(-12.0) == "-12"
    assert viz.fmt_num(float("nan")) == "—"
    assert viz.fmt_num(None) == "—"


def test_signal_labels_never_collide():
    """Consecutive signalling points would otherwise stack their labels into
    unreadable runs like '53.653.8'. Labels must be at least two periods
    apart and capped at four."""
    v = [50.0] * 40
    for i in range(10, 16):            # six consecutive signals
        v[i] = 90.0
    res = spc.xmr(v, x=pd.date_range("2024-01-01", periods=40, freq="MS"),
                  screen_mr=False)
    fig = viz.control_chart(res)
    notes = [a for a in fig.layout.annotations if a.text]
    assert len(notes) <= 4
    xs = sorted(pd.to_datetime([a.x for a in notes]))
    gaps = [(b - a).days for a, b in zip(xs, xs[1:])]
    assert all(g >= 55 for g in gaps), gaps      # >= 2 monthly periods apart
    _render(fig)


@pytest.mark.parametrize("n_signals", [0, 1, 3, 6, 7, 20])
def test_control_chart_renders_at_every_signal_count(n_signals):
    """The regression test for the d3-format bug: the direct-label branch
    only runs for 1-6 signals, so a chart with 0 or 20 rendered fine while
    a chart with 3 raised."""
    rng = np.random.default_rng(0)
    v = list(rng.normal(50, 2, 40))
    for i in range(n_signals):
        v[10 + i] = 90 + i          # force points outside the limits
    res = spc.xmr(v, x=pd.date_range("2024-01-01", periods=40, freq="MS"),
                  target=55, higher_is_better=False, unit="minutes",
                  label="test", screen_mr=False)
    assert len(res.signals) >= min(n_signals, 1) or n_signals == 0
    _render(viz.control_chart(res))
    _render(viz.mr_chart(res))


def test_every_registered_metric_renders(tbl):
    for standard in ("SSNAP", "INAS"):
        for key in THRESHOLDS[standard]:
            res = metrics.metric_series(tbl, key, standard=standard)
            _render(viz.control_chart(res, title=METRICS[key].label))
            if res.chart_type == "xmr":
                _render(viz.mr_chart(res))


def test_funnel_cusum_and_rare_event_charts_render(tables, tbl):
    fn = metrics.funnel_for_metric(tables["admissions"], "su_4h", "site")
    _render(viz.funnel_chart(fn, title="funnel"))

    res = metrics.metric_series(tbl, "dtn_median")
    cs = spc.cusum(res.frame["value"], pd.to_datetime(tbl["period"]))
    _render(viz.cusum_chart(cs, title="cusum"))

    _render(viz.control_chart(spc.t_chart([12, 45, 30, 88, 21, 64, 39, 17,
                                           120, 55, 41, 73, 29, 51])))
    _render(viz.control_chart(spc.g_chart([10, 25, 8, 40, 15, 30, 12, 22, 19, 33])))


def test_supporting_forms_render(tables):
    adm = tables["admissions"]
    _render(viz.pareto(["a", "b", "c", "d"], [40, 25, 20, 15]))
    _render(viz.stacked_shift(metrics.mrs_distribution(adm, "site"), "site"))
    _render(viz.bar(["x", "y", "z"], [1.0, 2.0, 3.0], target=2.5))
    _render(viz.bar(["x", "y", "z"], [1.0, 2.0, 3.0], horizontal=True, target=2.5))

    ses = tables["sessions"]
    piv = ses.pivot_table(index="discipline", columns="weekend", values="minutes",
                          aggfunc="mean", observed=True)
    _render(viz.heatmap(piv.round(1), unit="min"))

    m = ses.copy()
    m["period"] = m["date"].dt.to_period("M").dt.to_timestamp()
    g = m.groupby("period", observed=True).agg(
        applicable=("applicable", "sum"), delivered_minutes=("minutes", "sum")).reset_index()
    g["required_minutes"] = g["applicable"] * 45
    g["available_minutes"] = g["required_minutes"] * 0.9
    g = g.rename(columns={"delivered_minutes": "delivered_minutes"})
    _render(viz.demand_capacity(g))


def test_multi_line_renders_for_every_site(tables):
    adm = tables["admissions"]
    tbl_site = metrics.build_period_table(adm, tables["sessions"], group_col="site")
    plot = tbl_site.assign(value=tbl_site["los_days"])
    _render(viz.multi_line(plot, "period", "value", "site", target=12.0))


# ---------------------------------------------------------------------------
# Degenerate inputs: what a heavily filtered dashboard actually produces
# ---------------------------------------------------------------------------
def test_charts_render_on_a_constant_series():
    res = spc.xmr([10.0] * 20, x=pd.date_range("2024-01-01", periods=20, freq="MS"))
    assert res.meta["degenerate"]
    _render(viz.control_chart(res))
    _render(viz.mr_chart(res))


def test_charts_render_with_missing_periods():
    v = [10.0, np.nan, 12.0, 11.0, np.nan, np.nan, 13.0, 12.0, 11.0, 14.0,
         12.0, 13.0, np.nan, 11.0, 12.0]
    res = spc.xmr(v, x=pd.date_range("2024-01-01", periods=len(v), freq="MS"))
    _render(viz.control_chart(res))


def test_charts_render_on_a_single_point():
    res = spc.xmr([10.0], x=pd.date_range("2024-01-01", periods=1, freq="MS"))
    _render(viz.control_chart(res))


def test_p_chart_renders_with_zero_denominator_periods():
    num = [5, 0, 8, 0, 6, 7, 0, 9, 5, 6, 8, 7]
    den = [10, 0, 16, 0, 12, 14, 0, 18, 10, 12, 16, 14]
    res = spc.p_chart(num, den, pd.date_range("2024-01-01", periods=12, freq="MS"),
                      target=50, higher_is_better=True)
    _render(viz.control_chart(res))


def test_sparkline_survives_degenerate_input():
    """The scorecard renders a dozen of these on every rerun; one bad
    series must not take the page down."""
    for v in ([10.0] * 20, [np.nan] * 20, [1.0], list(range(30))):
        res = spc.xmr(v, target=5, higher_is_better=True)
        out = _sparkline(res)
        assert isinstance(out, str)


def _sparkline(res):
    # Imported lazily: core.ui imports streamlit, which is fine under
    # pytest but heavier than the rest of this module needs.
    from core import ui
    return ui.sparkline_svg(res)


def test_phase_breaks_and_targets_render_together(tbl):
    res = metrics.metric_series(tbl, "su_4h", phases=[12, 24], baseline=None)
    assert res.frame["phase"].nunique() == 3
    _render(viz.control_chart(res, show_target=True, show_band=True))
