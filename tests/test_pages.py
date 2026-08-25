"""
Page-level tests using Streamlit's in-process harness.
=====================================================================

``tests/test_spc.py`` proves the statistics are right and
``tests/test_viz.py`` proves the figures serialise. Neither runs a
*page*, so neither catches a page that raises on the third radio option
or the fourth tab -- and those are exactly the paths a developer never
clicks and a user finds immediately.

``AppTest`` executes a page script in-process with a real session state,
so widget values can be set and the script re-run. It is far more robust
than driving a browser: no selectors, no timing, and any uncaught
exception is available as ``at.exception``.

The pattern throughout: run the page, assert no exception, then walk
every option of every control that changes what is computed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAGES = {
    "overview": ROOT / "views" / "overview.py",
    "hyperacute": ROOT / "views" / "hyperacute.py",
    "control_charts": ROOT / "views" / "control_charts.py",
    "therapy": ROOT / "views" / "therapy.py",
    "caseload": ROOT / "views" / "caseload.py",
    "explorer": ROOT / "views" / "explorer.py",
    "data_quality": ROOT / "views" / "data_quality.py",
}

TIMEOUT = 180


def run(page: str, **widgets) -> AppTest:
    at = AppTest.from_file(str(PAGES[page]), default_timeout=TIMEOUT)
    for k, v in widgets.items():
        at.session_state[k] = v
    at.run()
    return at


def assert_clean(at: AppTest, context: str = "") -> None:
    if at.exception:
        detail = "\n".join(str(e.value) for e in at.exception)
        raise AssertionError(f"{context}: page raised\n{detail}")


# ---------------------------------------------------------------------------
# Every page loads
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("page", list(PAGES))
def test_page_loads(page):
    assert_clean(run(page), page)


# ---------------------------------------------------------------------------
# Sidebar controls affect every page, so sweep them on the busiest ones
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("standard", ["SSNAP", "INAS"])
@pytest.mark.parametrize("page", ["overview", "hyperacute", "therapy"])
def test_both_standards(page, standard):
    assert_clean(run(page, f_standard=standard), f"{page}/{standard}")


@pytest.mark.parametrize("freq", ["Month", "Quarter"])
def test_reporting_periods(freq):
    assert_clean(run("overview", f_freq=freq), freq)
    assert_clean(run("control_charts", f_freq=freq), freq)


@pytest.mark.parametrize("rules", [
    "NHS Making Data Count (4 rules, runs of 7)",
    "Nelson (8 rules, runs of 9)",
    "Rule 1 only (points outside the limits)",
])
def test_every_rule_set(rules):
    assert_clean(run("control_charts", f_rules=rules), rules)


@pytest.mark.parametrize("baseline", [
    "Whole series", "Freeze on first 12 periods", "Freeze on first 18 periods"])
def test_every_limit_calculation(baseline):
    assert_clean(run("control_charts", f_baseline=baseline), baseline)


@pytest.mark.parametrize("laney", [True, False])
def test_laney_toggle(laney):
    assert_clean(run("therapy", f_laney=laney), f"laney={laney}")


def test_single_site_selection():
    """The most common way a dashboard breaks: filter it down until the
    denominators are tiny."""
    at = run("overview", f_sites=["Lakeview General"])
    assert_clean(at, "single site")
    at2 = run("control_charts", f_sites=["Lakeview General"], cc_split="One site")
    assert_clean(at2, "single site, split")


def test_narrow_date_window():
    """Three months of data: too few points for stable limits. The app must
    degrade to 'not enough data', not raise."""
    import datetime as dt
    at = run("overview",
             f_dates=(dt.date(2026, 1, 1), dt.date(2026, 3, 31)))
    assert_clean(at, "narrow window")


# ---------------------------------------------------------------------------
# Control charts: every domain, and the tabs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("domain", [
    "Hyperacute pathway", "Stroke unit & ward care", "Therapy & rehabilitation",
    "Safety", "Secondary prevention & discharge", "Outcomes"])
def test_every_domain_on_the_workbench(domain):
    assert_clean(run("control_charts", cc_domain=domain, cu_domain=domain), domain)


@pytest.mark.parametrize("event", [
    "Pressure ulcers", "Falls", "Hospital-acquired pneumonia"])
def test_every_rare_event_chart(event):
    assert_clean(run("control_charts", rare_event=event), event)


@pytest.mark.parametrize("group", ["site", "stroke_type", "sex"])
def test_funnel_grouping(group):
    assert_clean(run("control_charts", fn_group=group), group)


def test_funnel_without_overdispersion_adjustment():
    assert_clean(run("control_charts", fn_od=False), "funnel raw")


def test_phase_break_enabled():
    assert_clean(run("control_charts", cc_break=True), "phase break")


@pytest.mark.parametrize("k,h", [(0.25, 3.0), (0.5, 4.0), (1.5, 6.0)])
def test_cusum_parameter_extremes(k, h):
    assert_clean(run("control_charts", cu_k=k, cu_h=h), f"k={k} h={h}")


# ---------------------------------------------------------------------------
# Therapy and caseload: every discipline and measure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("disc", ["PT", "OT", "SLT"])
@pytest.mark.parametrize("measure", [
    "Days with therapy", "Minutes per day", "Days at 45 minutes",
    "Assessment within 72 hours"])
def test_every_therapy_measure(disc, measure):
    assert_clean(run("therapy", td_disc=disc, td_measure=measure),
                 f"{disc}/{measure}")


@pytest.mark.parametrize("stratify", [True, False])
def test_dose_outcome_stratification(stratify):
    """Unstratified is the path that shows Simpson's paradox, and it takes a
    different code branch from the stratified one."""
    assert_clean(run("therapy", td_strat=stratify), f"stratify={stratify}")


@pytest.mark.parametrize("disc", ["PT", "OT", "SLT"])
@pytest.mark.parametrize("view", [
    "Caseload per available WTE", "Active caseload", "Available vs funded WTE"])
def test_every_caseload_view(disc, view):
    assert_clean(run("caseload", cap_disc=disc, cap_view=view), f"{disc}/{view}")


@pytest.mark.parametrize("ratio,dose", [(0.35, 15), (0.60, 45), (0.90, 90)])
def test_capacity_assumption_extremes(ratio, dose):
    """The conclusion text branches on whether establishment is adequate;
    all three branches must be reachable and none may raise."""
    assert_clean(run("caseload", cap_ratio=ratio, cap_dose=dose),
                 f"ratio={ratio} dose={dose}")


# ---------------------------------------------------------------------------
# Hyperacute and data quality
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("measure", [
    "Door-to-needle", "Door-to-imaging", "Time to stroke unit"])
def test_out_of_hours_measures(measure):
    assert_clean(run("hyperacute", ooh_measure=measure), measure)


@pytest.mark.parametrize("key", [
    "dtn_median", "ct_60", "thrombolysis_rate", "su_4h", "dtn_60"])
def test_hyperacute_indicator_picker(key):
    assert_clean(run("hyperacute", hyper_metric=key), key)


@pytest.mark.parametrize("field", [
    "door_to_needle_min", "door_to_ct_min", "onset_to_door_min",
    "door_to_puncture_min"])
def test_digit_preference_on_every_timing_field(field):
    assert_clean(run("data_quality", dq_digit=field), field)


def test_completeness_with_no_fields_selected():
    """An empty multiselect is the classic unguarded path."""
    assert_clean(run("data_quality", dq_fields=[]), "no fields")


# ---------------------------------------------------------------------------
# Explorer: cohort filters, including ones that select nothing
# ---------------------------------------------------------------------------
def test_explorer_breach_cohorts():
    for breach in ["Imaging beyond 1 hour", "Needle beyond 60 minutes",
                   "Stroke unit beyond 4 hours", "SLT assessment beyond 72 hours"]:
        assert_clean(run("explorer", pe_breach=[breach]), breach)


def test_explorer_complication_cohorts():
    assert_clean(run("explorer", pe_comp=["Pneumonia", "Any fall"]), "complications")


def test_explorer_empty_cohort_does_not_raise():
    """Filters that select zero patients must stop cleanly, not divide by zero."""
    at = run("explorer", pe_type=[], pe_treat=[])
    assert_clean(at, "empty cohort")


def test_explorer_single_patient_cohort():
    at = run("explorer", pe_nihss=(41, 42), pe_mrs=(6, 6))
    assert_clean(at, "tiny cohort")
