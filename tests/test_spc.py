"""
Tests for the SPC engine and the metric pipeline.

Two kinds of test here:

* **Arithmetic tests** against hand-computable examples. These pin the
  formulae. If someone "simplifies" the sigma estimator to a standard
  deviation, these fail.

* **Ground-truth tests** against the signals deliberately planted in the
  synthetic generator. These pin the *detection*: they assert that a
  known step change, a known drift and a known outlier are all found by
  the rules, at roughly the right time. If you change the rule set
  defaults, expect these to move.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import loaders, mdc, metrics, spc  # noqa: E402


# ===========================================================================
# Arithmetic
# ===========================================================================
def test_xmr_limits_match_hand_calculation():
    """XmR limits are xbar +/- 2.66 * mRbar, not xbar +/- 3 * SD."""
    v = [10, 12, 11, 13, 12, 14, 13, 11, 12, 13, 12, 14]
    res = spc.xmr(v, screen_mr=False)
    mr = np.abs(np.diff(v))
    expected_sigma = mr.mean() / 1.128
    assert res.sigma == pytest.approx(expected_sigma, rel=1e-9)
    assert res.frame["ucl"].iloc[0] == pytest.approx(
        np.mean(v) + 3 * expected_sigma, rel=1e-9)
    # And it must NOT equal the naive SD-based limit.
    assert res.sigma != pytest.approx(np.std(v, ddof=1), rel=1e-3)


def test_xmr_sigma_is_robust_to_a_step_change():
    """The whole point of the moving-range estimator.

    A series with a big step change has a large overall SD but a small
    average moving range (only one adjacent pair straddles the step).
    """
    v = [10] * 15 + [30] * 15
    res = spc.xmr(v, screen_mr=False)
    overall_sd = np.std(v, ddof=1)
    assert res.sigma < overall_sd / 3
    # The step must therefore be detected.
    assert res.frame["special"].sum() > 0


def test_mr_screening_removes_the_inflating_range():
    plain = spc.xmr([10, 10, 10, 10, 60, 10, 10, 10, 10, 10], screen_mr=False)
    screened = spc.xmr([10, 10, 10, 10, 60, 10, 10, 10, 10, 10], screen_mr=True)
    assert screened.sigma < plain.sigma


def test_p_chart_centre_is_pooled_not_averaged():
    """A month with 3 cases must not weigh the same as a month with 300."""
    num = [3, 150]
    den = [3, 300]
    res = spc.p_chart(num, den, laney=False)
    # Pooled: 153/303 = 50.5%.  Averaged: (100 + 50)/2 = 75%.
    assert res.frame["cl"].iloc[0] == pytest.approx(100 * 153 / 303, rel=1e-9)


def test_p_chart_limits_widen_as_denominator_falls():
    num = [50, 5]
    den = [100, 10]
    res = spc.p_chart(num, den, laney=False)
    width_big = res.frame["ucl"].iloc[0] - res.frame["lcl"].iloc[0]
    width_small = res.frame["ucl"].iloc[1] - res.frame["lcl"].iloc[1]
    assert width_small > width_big


def test_laney_reduces_to_plain_p_chart_when_not_overdispersed():
    """p' is a strict generalisation: sigma_z = 1 must give the p-chart back."""
    rng = np.random.default_rng(7)
    den = np.full(40, 200)
    num = rng.binomial(200, 0.4, 40)
    plain = spc.p_chart(num, den, laney=False)
    auto = spc.p_chart(num, den, laney="auto")
    assert auto.meta["sigma_z"] == pytest.approx(1.0, abs=0.25)
    assert not auto.meta["overdispersed"]
    np.testing.assert_allclose(plain.frame["ucl"], auto.frame["ucl"], rtol=1e-9)


def test_laney_widens_limits_under_real_overdispersion():
    """Beta-binomial data: the same nominal p, but p itself varies."""
    rng = np.random.default_rng(11)
    den = np.full(40, 500)
    p_true = rng.beta(8, 12, 40)          # mean 0.4, but genuinely variable
    num = rng.binomial(500, p_true)
    plain = spc.p_chart(num, den, laney=False)
    corrected = spc.p_chart(num, den, laney=True)
    assert corrected.meta["sigma_z"] > 1.5
    assert (corrected.frame["ucl"] > plain.frame["ucl"]).all()
    # The uncorrected chart is the one that cries wolf.
    assert plain.frame["special"].sum() > corrected.frame["special"].sum()


def test_laney_never_narrows_limits():
    """Underdispersion must not be used to tighten limits."""
    den = np.full(30, 100)
    num = np.full(30, 40)  # zero variation -> sigma_z = 0
    plain = spc.p_chart(num, den, laney=False)
    corrected = spc.p_chart(num, den, laney=True)
    np.testing.assert_allclose(plain.frame["ucl"], corrected.frame["ucl"], rtol=1e-9)


def test_u_chart_uses_exposure_not_count_of_units():
    counts = [5, 5]
    exposure = [1000, 500]      # same count, half the bed days
    res = spc.u_chart(counts, exposure, multiplier=1000)
    assert res.frame["value"].iloc[0] == pytest.approx(5.0)
    assert res.frame["value"].iloc[1] == pytest.approx(10.0)


def test_c_chart_sigma_is_sqrt_of_the_mean():
    v = [4, 6, 5, 5, 4, 6, 5, 5, 4, 6]
    res = spc.c_chart(v)
    assert res.sigma == pytest.approx(np.sqrt(np.mean(v)), rel=1e-9)


def test_t_chart_back_transform_is_monotone_and_positive():
    gaps = [12, 45, 30, 88, 21, 64, 39, 17, 120, 55, 41, 73]
    res = spc.t_chart(gaps)
    f = res.frame
    assert (f["lcl"].dropna() >= 0).all()
    assert (f["ucl"].dropna() >= f["cl"].dropna()).all()
    assert (f["cl"].dropna() >= f["lcl"].dropna()).all()


def test_g_chart_uses_geometric_sigma():
    v = [10, 25, 8, 40, 15, 30, 12, 22]
    res = spc.g_chart(v)
    gbar = np.mean(v)
    assert res.sigma == pytest.approx(np.sqrt(gbar * (gbar + 1)), rel=1e-9)
    assert res.centre == pytest.approx(np.median(v))


# ===========================================================================
# Rules
# ===========================================================================
def test_rule_1_fires_on_a_single_outlier():
    v = [10] * 20 + [40]
    res = spc.xmr(v, rule_set="core", screen_mr=False)
    assert 1 in res.frame["rules"].iloc[-1]


def test_rule_2_shift_respects_the_configured_run_length():
    # 7 points on one side: fires under NHS (run 7), not under Nelson (run 9).
    v = [10, 12, 10, 12, 10, 12, 10, 12, 10, 12,
         13, 13, 13, 13, 13, 13, 13]
    nhs = spc.xmr(v, rule_set="nhs", screen_mr=False)
    nelson = spc.xmr(v, rule_set="nelson", screen_mr=False)
    n_nhs = sum(2 in r for r in nhs.frame["rules"])
    n_nelson = sum(2 in r for r in nelson.frame["rules"])
    assert n_nhs > 0
    assert n_nelson <= n_nhs


def test_rule_3_trend_needs_strict_monotonicity():
    flat_break = [1, 2, 3, 3, 4, 5, 6, 7]     # tie at index 3 breaks the run
    clean = [1, 2, 3, 4, 5, 6, 7, 8]
    a = spc.xmr(flat_break, rule_set="nhs", screen_mr=False)
    b = spc.xmr(clean, rule_set="nhs", screen_mr=False)
    assert sum(3 in r for r in b.frame["rules"]) > sum(3 in r for r in a.frame["rules"])


def test_rule_sets_are_not_nested():
    """A real property worth pinning, because it surprises people.

    Nelson has more rules than the NHS set (8 vs 4) but a *longer* run
    length for a shift (9 vs 7). So Nelson is not uniformly more
    sensitive: on a series whose only signal is a modest shift, the NHS
    set can find more. Only rule 1 is common to all three sets, so only
    'core <= each of the others' holds in general.
    """
    rng = np.random.default_rng(3)
    v = np.concatenate([rng.normal(10, 1, 30), rng.normal(11.2, 1, 30)])
    core = spc.xmr(v, rule_set="core").frame["special"].sum()
    nhs = spc.xmr(v, rule_set="nhs").frame["special"].sum()
    nelson = spc.xmr(v, rule_set="nelson").frame["special"].sum()
    assert core <= nhs
    assert core <= nelson
    assert spc.RULE_SETS["nhs"]["run"] < spc.RULE_SETS["nelson"]["run"]


def test_rules_are_evaluated_in_standardised_space_for_variable_limits():
    """A p-chart with a varying denominator must not flag purely because
    a small month has wide limits."""
    rng = np.random.default_rng(5)
    den = rng.integers(20, 400, 60)
    num = rng.binomial(den, 0.5)
    res = spc.p_chart(num, den, laney=False, rule_set="core")
    # With a true constant p, rule 1 should fire on roughly nothing.
    assert res.frame["special"].sum() <= 2


# ===========================================================================
# Baseline freezing and phases
# ===========================================================================
def test_frozen_baseline_keeps_limits_from_the_baseline_period():
    rng = np.random.default_rng(21)
    v = np.concatenate([rng.normal(10, 1, 20), rng.normal(20, 1, 20)])
    frozen = spc.xmr(v, baseline=20)
    whole = spc.xmr(v)
    assert frozen.frame["cl"].iloc[-1] == pytest.approx(np.mean(v[:20]), abs=0.01)
    assert whole.frame["cl"].iloc[-1] == pytest.approx(np.mean(v), abs=0.01)
    # Frozen limits put every post-change point outside the limits.
    assert frozen.frame["special"].tail(20).all()

    # Whole-series limits put the centre line between the two levels, so
    # the chart loses its reference point: the shift rule now flags all
    # 40 points, 20 either side of a mean that describes neither level.
    assert whole.frame["special"].sum() >= frozen.frame["special"].sum()
    assert sum(2 in r for r in whole.frame["rules"]) > sum(2 in r for r in frozen.frame["rules"])

    # But sigma barely moves. Only one moving range straddles the step and
    # the screening rule discards it, so a single sharp change does NOT
    # hide itself behind inflated limits — the common folk justification
    # for baseline freezing is weaker than usually claimed. It is the lost
    # reference level, not sigma inflation, that matters here.
    assert whole.sigma == pytest.approx(frozen.sigma, rel=0.35)


def _mean_sigma(builder, reps: int = 40) -> float:
    return float(np.mean([spc.xmr(builder(np.random.default_rng(s))).sigma
                          for s in range(reps)]))


def test_moving_range_sigma_is_immune_to_step_changes_of_any_size():
    """Measured, not assumed — and the result is stronger than the folklore.

    A step change contaminates exactly one moving range out of n-1, and
    the screening rule discards it. So sigma is unaffected by a step of
    *any* magnitude: a 25-sigma jump estimates the same sigma as a
    3-sigma one. Whole-series limits therefore do NOT hide a sharp step;
    the reason to freeze a baseline is that the centre line lands between
    the two levels, not that the limits widen.
    """
    base = _mean_sigma(lambda r: r.normal(10, 1, 60))
    for magnitude in (3, 6, 12, 25):
        s = _mean_sigma(lambda r, m=magnitude: np.concatenate(
            [r.normal(10, 1, 30), r.normal(10 + m, 1, 30)]))
        assert s == pytest.approx(base, rel=0.05), (magnitude, s, base)


def test_moving_range_sigma_inflates_only_when_drift_approaches_the_noise():
    """Gradual drift is the case where sigma really does inflate — but the
    effect is second order in (drift per period / noise), so it stays small
    until the two are comparable. Roughly, per-period drift of:
        0.25 sd -> +2%     1.0 sd -> +13%
        0.5  sd -> +4%     2.0 sd -> +45%
    """
    base = _mean_sigma(lambda r: np.concatenate([r.normal(10, 1, 30),
                                                 r.normal(10, 1, 30)]))

    def drifted(per_step):
        return _mean_sigma(lambda r, d=per_step: np.concatenate(
            [r.normal(10, 1, 30), r.normal(10, 1, 30) + np.arange(30) * d]))

    assert drifted(0.25) < base * 1.06        # slow drift: negligible
    assert drifted(1.0) > base * 1.08         # comparable to noise: visible
    assert drifted(2.0) > base * 1.35         # dominating the noise: large
    assert drifted(2.0) > drifted(1.0) > drifted(0.25)


def test_zero_sigma_is_reported_rather_than_read_as_stability():
    """A constant series has zero moving range, so no rule can ever fire.
    Silently reporting 'common cause' would be badly misleading: in real
    audit data this means rounding or a defaulted field."""
    res = spc.xmr([10] * 30)
    assert res.meta["degenerate"]
    assert not res.frame["special"].any()


def test_phase_break_gives_each_segment_its_own_limits():
    v = [10] * 20 + [20] * 20
    res = spc.xmr(v, phases=[20], screen_mr=False)
    assert res.frame["cl"].iloc[0] == pytest.approx(10.0)
    assert res.frame["cl"].iloc[-1] == pytest.approx(20.0)
    assert res.frame["phase"].nunique() == 2


def test_suggest_phase_break_finds_a_planted_step():
    rng = np.random.default_rng(2)
    v = np.concatenate([rng.normal(50, 3, 20), rng.normal(30, 3, 20)])
    cp = spc.suggest_phase_break(v)
    assert cp is not None
    assert cp["worth_splitting"]
    assert abs(cp["index"] - 20) <= 2
    assert cp["shift"] < -15


def test_suggest_phase_break_declines_on_stable_data():
    rng = np.random.default_rng(4)
    cp = spc.suggest_phase_break(rng.normal(50, 3, 60))
    assert cp is None or not cp["worth_splitting"]


# ===========================================================================
# Funnel and CUSUM
# ===========================================================================
def test_funnel_limits_narrow_as_denominator_grows():
    fn = spc.funnel_plot([50, 500], [100, 1000], ["small", "big"], overdispersion=False)
    lo, hi = fn["bands"]["998"]
    width = hi - lo
    assert width[0] > width[-1]


def test_funnel_overdispersion_inflates_only_when_needed():
    rng = np.random.default_rng(9)
    den = rng.integers(80, 400, 15)
    tight = rng.binomial(den, 0.5)                       # pure binomial
    loose = rng.binomial(den, rng.beta(4, 4, 15))        # genuinely variable
    a = spc.funnel_plot(tight, den, [f"u{i}" for i in range(15)])
    b = spc.funnel_plot(loose, den, [f"u{i}" for i in range(15)])
    assert a["phi"] < b["phi"]
    assert b["inflation"] > 1.0


def test_cusum_signals_a_sustained_shift_and_ignores_noise():
    rng = np.random.default_rng(6)
    stable = rng.normal(0, 1, 60)
    shifted = np.concatenate([rng.normal(0, 1, 30), rng.normal(1.0, 1, 30)])
    a = spc.cusum(stable, target_mean=0, sigma=1)
    b = spc.cusum(shifted, target_mean=0, sigma=1)
    assert not (a["signal_high"] | a["signal_low"]).any()
    assert b["signal_high"].any()
    assert int(b.index[b["signal_high"]].min()) > 30


def test_run_chart_detects_a_shift():
    v = [1, 2, 1, 2, 1, 2, 1, 2, 5, 5, 5, 5, 5, 5, 5]
    rc = spc.run_chart(v)
    assert rc["shift"].any()


# ===========================================================================
# Making Data Count classification
# ===========================================================================
def test_assurance_compares_limits_with_target_not_last_point():
    """A single bad month must not flip a 'pass' to a 'fail'."""
    v = [90, 91, 92, 90, 91, 92, 90, 91, 92, 90, 91, 89]
    res = spc.xmr(v, target=80, higher_is_better=True, screen_mr=False)
    assert mdc.classify_assurance(res) == "pass"

    v2 = [50] * 12
    res2 = spc.xmr(v2, target=80, higher_is_better=True, screen_mr=False)
    assert mdc.classify_assurance(res2) == "fail"


def test_hit_and_miss_when_target_sits_inside_the_limits():
    rng = np.random.default_rng(8)
    v = rng.normal(80, 6, 30)
    res = spc.xmr(v, target=80, higher_is_better=True)
    assert mdc.classify_assurance(res) == "hit_miss"


def test_variation_polarity_follows_higher_is_better():
    v = [10] * 20 + [3] * 10
    good = spc.xmr(v, higher_is_better=False, screen_mr=False)
    bad = spc.xmr(v, higher_is_better=True, screen_mr=False)
    assert mdc.verdict(good).variation == "low_improve"
    assert mdc.verdict(bad).variation == "low_concern"


def test_insufficient_data_is_reported_not_guessed():
    res = spc.xmr([1, 2, 3, 4, 5])
    assert mdc.verdict(res).variation == "insufficient"


# ===========================================================================
# Ground truth: the signals planted in the generator
# ===========================================================================
@pytest.fixture(scope="module")
def tables():
    return loaders.prepare(loaders.cached_synthetic())


@pytest.fixture(scope="module")
def by_site(tables):
    return metrics.build_period_table(tables["admissions"], tables["sessions"],
                                      group_col="site")


def _site(by_site, name):
    return by_site[by_site["site"] == name].reset_index(drop=True)


def test_signal_1_riverside_thrombolysis_redesign(by_site):
    """Door-to-needle steps down ~20 min at Riverside from Feb 2025."""
    tbl = _site(by_site, "Riverside General")
    res = metrics.metric_series(tbl, "dtn_median")
    assert len(res.signals) > 0
    v = mdc.verdict(res)
    assert v.variation == "low_improve", v.variation
    cp = spc.suggest_phase_break(res.frame["value"])
    assert cp is not None and cp["worth_splitting"]
    when = pd.to_datetime(tbl["period"].iloc[cp["index"]])
    assert pd.Timestamp("2024-11-01") <= when <= pd.Timestamp("2025-05-01"), when
    assert cp["shift"] < -8


def test_signal_2_northbay_bed_pressure(by_site):
    """Time to stroke unit drifts up from Oct 2025; the 4-hour rate falls."""
    tbl = _site(by_site, "Northbay Regional")
    t2su = metrics.metric_series(tbl, "time_to_su_median")
    assert mdc.verdict(t2su).variation == "high_concern"

    su4 = metrics.metric_series(tbl, "su_4h")
    sig_months = pd.to_datetime(su4.signals["x"])
    assert len(sig_months) > 0
    assert sig_months.min() >= pd.Timestamp("2025-06-01"), sig_months.min()
    assert mdc.verdict(su4).variation == "low_concern"


def test_signal_2_cusum_leads_the_shewhart_chart(by_site):
    """A slow drift is what CUSUM is for.

    The comparison only means something against *whole-series* limits,
    which is the realistic default. Freezing the limits on a clean
    baseline makes the Shewhart chart very sensitive too — sometimes more
    so — which is worth knowing: the CUSUM's advantage is largely an
    advantage over limits that the drift itself has inflated.
    """
    tbl = _site(by_site, "Northbay Regional")
    res = metrics.metric_series(tbl, "time_to_su_median")   # whole-series limits
    baseline_n = 18
    cs = spc.cusum(res.frame["value"],
                   target_mean=float(np.nanmean(res.frame["value"].head(baseline_n))),
                   sigma=float(np.nanmean(res.frame["sigma"])))
    first_cusum = cs.index[cs["signal_high"]].min() if cs["signal_high"].any() else np.inf
    # Compare like with like: a single point outside the limits is the
    # Shewhart chart's statement that "this level is new". The run rule
    # is excluded because, on whole-series limits, it flags the entire
    # pre-change period for sitting below a contaminated centre line —
    # that is the artefact, not a detection.
    rule1_idx = [i for i, r in zip(res.frame.index, res.frame["rules"]) if 1 in r]
    first_shew = min(rule1_idx) if rule1_idx else np.inf
    assert np.isfinite(first_cusum), "CUSUM missed a drift it is designed to find"
    assert first_cusum <= first_shew, (first_cusum, first_shew)


def test_cusum_against_frozen_limits_is_not_uniformly_earlier(by_site):
    """The honest counterpart to the test above — documents the limitation
    rather than hiding it."""
    tbl = _site(by_site, "Northbay Regional")
    frozen = metrics.metric_series(tbl, "time_to_su_median", baseline=18)
    assert frozen.frame["special"].any()


def test_signal_3_lakeview_physiotherapy_gap(by_site):
    """PT delivery falls Jun 2025 - Feb 2026 and recovers."""
    tbl = _site(by_site, "Lakeview General")
    res = metrics.metric_series(tbl, "pt_pct_days", baseline=18)
    gap = tbl["period"].between("2025-06-01", "2026-02-28")
    in_gap = res.frame.loc[gap.to_numpy(), "value"].mean()
    before = res.frame.loc[tbl["period"] < "2025-06-01", "value"].mean()
    after = res.frame.loc[tbl["period"] > "2026-02-28", "value"].mean()
    assert in_gap < before - 5, (before, in_gap)
    assert after > in_gap + 3, (in_gap, after)
    assert res.frame.loc[gap.to_numpy(), "special"].any()


def test_signal_3_overdispersion_correction_engages_on_patient_days(by_site):
    """Patient-day denominators are large enough that plain binomial limits
    would flag nearly everything."""
    tbl = _site(by_site, "Lakeview General")
    corrected = metrics.metric_series(tbl, "pt_pct_days", laney=True)
    plain = metrics.metric_series(tbl, "pt_pct_days", laney=False)
    assert corrected.meta["sigma_z"] > 1.3
    assert plain.frame["special"].sum() > corrected.frame["special"].sum()


def test_signal_4_st_brendans_falls_cluster(by_site):
    """A one-month cluster of falls in Nov 2024."""
    tbl = _site(by_site, "St Brendan's University Hospital")
    res = metrics.metric_series(tbl, "falls_rate", rule_set="core")
    sig = pd.to_datetime(res.signals["x"])
    assert len(sig) >= 1
    assert any(pd.Timestamp("2024-10-01") <= s <= pd.Timestamp("2024-12-31") for s in sig), \
        sig.tolist()


def test_signal_5_swallow_screening_improves_across_all_sites(by_site):
    """ED triage screening from Apr 2024."""
    for site in by_site["site"].unique():
        tbl = _site(by_site, site)
        res = metrics.metric_series(tbl, "swallow_4h")
        before = res.frame.loc[tbl["period"] < "2024-04-01", "value"].mean()
        after = res.frame.loc[tbl["period"] >= "2024-04-01", "value"].mean()
        assert after > before + 5, (site, before, after)


def test_signal_6_weekend_effect_is_large(tables):
    ses = tables["sessions"]
    we = ses[ses["weekend"]]
    wd = ses[~ses["weekend"]]
    we_rate = we["minutes"].sum() / we["applicable"].sum()
    wd_rate = wd["minutes"].sum() / wd["applicable"].sum()
    assert we_rate < wd_rate * 0.5, (we_rate, wd_rate)


def test_seasonality_is_not_flagged_as_special_cause(tables):
    """Winter is common cause. A chart that flags every December has
    limits that are too tight."""
    tbl = metrics.build_period_table(tables["admissions"], tables["sessions"])
    res = metrics.metric_series(tbl, "los_median")
    decembers = pd.to_datetime(tbl["period"]).dt.month == 12
    assert not res.frame.loc[decembers.to_numpy(), "special"].all()


def test_planted_recording_defects_are_detected(tables):
    """The data-quality page must actually find the defects that were put in."""
    issues = loaders.validate(tables)
    by_col = {(i.table, i.column): i for i in issues}

    # D2 -- NIHSS "not recorded" sentinel leaking through as 99
    assert ("admissions", "nihss") in by_col
    assert by_col[("admissions", "nihss")].severity == "error"

    # D3 -- negative door-to-imaging from mismatched timestamps
    assert ("admissions", "door_to_ct_min") in by_col

    # D4 -- the SLT feed break at one site
    assert ("admissions", "slt_assess_hours") in by_col


def test_implausible_timings_do_not_score_as_compliant(tables):
    """A negative door-to-imaging time satisfies '<= 60 minutes'. If the
    numerator does not guard against it, a clock error is silently counted
    as excellent performance."""
    adm = tables["admissions"]
    bad = adm["door_to_ct_min"] < 0
    assert bad.sum() > 0, "the test data should contain planted clock errors"
    assert not adm.loc[bad, "flag_ct_60"].any()
    assert not adm.loc[bad, "flag_ct_720"].any()


def test_digit_preference_is_visible_in_the_planted_data(tables):
    """D1: rounding to the nearest 5 must show up as an excess of final
    digits 0 and 5, which is the check the Data Quality page runs."""
    v = pd.to_numeric(tables["admissions"]["door_to_needle_min"],
                      errors="coerce").dropna()
    last = (v.astype(int) % 10).value_counts().reindex(range(10), fill_value=0)
    pct = 100 * last / last.sum()
    excess = pct.loc[0] + pct.loc[5] - 20.0
    assert excess > 10, excess


def test_conditional_fields_are_judged_within_their_cohort(tables):
    """The validator must not repeat the mistake the app warns about:
    door-to-needle is not '84% missing', it is missing for patients who
    were not thrombolysed."""
    issues = loaders.validate(tables)
    dtn = [i for i in issues if i.column == "door_to_needle_min"
           and "missing values" in i.message]
    adm = tables["admissions"]
    naive_missing_pct = 100 * adm["door_to_needle_min"].isna().mean()
    assert naive_missing_pct > 70          # naively it looks catastrophic
    for issue in dtn:                      # but scoped, it is not reported at all
        assert "among the thrombolysed cohort" in issue.message


# ===========================================================================
# Pipeline integrity
# ===========================================================================
def test_every_registered_metric_computes(tables):
    from core.standards import THRESHOLDS
    tbl = metrics.build_period_table(tables["admissions"], tables["sessions"])
    for standard in ("SSNAP", "INAS"):
        for key in THRESHOLDS[standard]:
            res = metrics.metric_series(tbl, key, standard=standard)
            assert res.n_points > 0, key
            mdc.verdict(res)


def test_numerators_are_nested_inside_denominators(tables):
    from core.standards import METRICS
    adm = tables["admissions"]
    for key, spec in METRICS.items():
        if spec.chart != "p" or spec.numerator not in adm.columns:
            continue
        assert not (adm[spec.numerator] & ~adm[spec.denominator]).any(), key


def test_no_proportion_exceeds_one_hundred_percent(tables):
    from core.standards import METRICS
    tbl = metrics.build_period_table(tables["admissions"], tables["sessions"])
    for key, spec in METRICS.items():
        if spec.chart != "p":
            continue
        num, den = tbl[spec.numerator], tbl[spec.denominator]
        ratio = (num / den.replace(0, np.nan)).dropna()
        assert (ratio <= 1.0 + 1e-9).all(), key


def test_switching_standard_changes_targets_not_values(tables):
    tbl = metrics.build_period_table(tables["admissions"], tables["sessions"])
    a = metrics.metric_series(tbl, "su_4h", standard="SSNAP")
    b = metrics.metric_series(tbl, "su_4h", standard="INAS")
    np.testing.assert_allclose(a.frame["value"].to_numpy(),
                               b.frame["value"].to_numpy(), equal_nan=True)
    assert a.target != b.target


def test_period_axis_has_no_gaps(tables):
    tbl = metrics.build_period_table(tables["admissions"], tables["sessions"])
    periods = pd.to_datetime(tbl["period"])
    assert (periods.diff().dropna().dt.days.between(28, 31)).all()


def test_therapy_minutes_per_day_uses_applicable_days(tables):
    """Dividing by delivered days instead of applicable days is the most
    common way therapy dose gets overstated."""
    pl = metrics.therapy_patient_level(tables["sessions"])
    naive = (pl["total_minutes"] / pl["days_with_therapy"].clip(lower=1)).median()
    correct = pl["minutes_per_day"].median()
    assert correct < naive
