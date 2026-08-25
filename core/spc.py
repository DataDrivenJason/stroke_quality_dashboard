"""
Statistical process control engine.
=====================================================================

Everything in this module answers one question: *is the variation I am
looking at the ordinary noise of a stable process, or is something
genuinely different going on?*

Why not just use the standard deviation
---------------------------------------
The naive approach — plot the mean and mean +/- 3*SD(all points) — is
wrong for process data, and wrong in a specific, dangerous direction.
SD(all points) is computed around the grand mean, so any special cause
in the series (a step change, a trend, an outlier) inflates the very
estimate you are using to detect it. The limits widen, and the signal
hides inside them.

Shewhart's answer is to estimate sigma from *short-term, within-subgroup*
variation only — variation that a shift or trend cannot contaminate,
because it is measured between adjacent points. For individuals data
that estimator is the average moving range:

    sigma_hat = mean(|x_i - x_{i-1}|) / d2,   d2 = 1.128 for n = 2

and therefore the limits are mean +/- 3 * mRbar / 1.128 = mean +/- 2.660 * mRbar.

The 1.128 is E[range of 2 draws from a standard normal]. The 3 is not a
p-value: Shewhart chose it as an economic balance between false alarms
(chasing noise) and missed signals, not as an inference threshold. Under
normality it corresponds to roughly 1 false alarm in 370 points, but the
limits are robust well beyond the normal case, which is why they survive
the skewed distributions that dominate clinical timing data.

Attribute charts (p, u, c) do not estimate sigma from the data at all.
The distributional model supplies it: binomial for proportions, Poisson
for counts. That is elegant when the model holds and disastrous when it
does not -- see the Laney correction below.

Chart selection
---------------
    Data type                                    Chart
    ---------------------------------------------------------------
    Continuous, one value per period             XmR (individuals)
    Proportion, variable denominator             p  (or p' if overdispersed)
    Rate of events per unit exposure             u  (or u')
    Count, constant area of opportunity          c
    Time between rare events                     t
    Opportunities between rare events            g
    Cross-sectional comparison of units          funnel plot
    Small persistent shift detection             CUSUM

Author's note on rule sets: this module implements the full Nelson (1984)
set of 8 plus the reduced NHS "Making Data Count" set. They disagree on
run length (Nelson uses 9 for a shift, NHS uses 7). Both are defensible;
what is not defensible is switching between them after seeing the data.
Pick one in the config and leave it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Shewhart constants for subgroups of size n = 2 (consecutive individuals)
# ---------------------------------------------------------------------------
D2_N2 = 1.128   # E[range] of 2 standard normal draws
D4_N2 = 3.267   # upper control limit factor for the moving-range chart
D3_N2 = 0.0     # lower MR limit is 0 for n = 2
SIGMA_MULTIPLIER = 3.0 / D2_N2  # 2.6595...

RuleSet = Literal["nelson", "nhs", "core"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ChartResult:
    """The output of any control chart calculation.

    ``frame`` always carries these columns, whatever the chart type:
        x, value, cl, ucl, lcl, sigma, phase, special, special_dir, rules
    plus ``ucl_2s``/``lcl_2s``/``ucl_1s``/``lcl_1s`` sigma zone boundaries
    where the chart type supports them (needed for the zone-based rules).
    """

    frame: pd.DataFrame
    chart_type: str
    centre: float
    sigma: float | None = None
    target: float | None = None
    higher_is_better: bool | None = None
    unit: str = ""
    label: str = ""
    meta: dict = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    @property
    def n_points(self) -> int:
        return int(self.frame["value"].notna().sum())

    @property
    def signals(self) -> pd.DataFrame:
        """Only the points that broke at least one rule."""
        return self.frame[self.frame["special"]]

    def last_valid(self) -> pd.Series | None:
        valid = self.frame[self.frame["value"].notna()]
        return None if valid.empty else valid.iloc[-1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_array(x: Iterable) -> np.ndarray:
    return np.asarray(pd.Series(list(x)).astype(float))


def _baseline_mask(n: int, baseline: slice | Sequence[bool] | int | None) -> np.ndarray:
    """Which points are used to *calculate* the limits.

    Freezing limits on a baseline and extending them forward is the
    standard improvement-science display, for three reasons — and it is
    worth being precise about which of them actually bite, because the
    usual folk justification ("otherwise the improvement hides itself")
    is only partly true:

    1. **The reference level is preserved.** This is the real reason.
       Whole-series limits put the centre line *between* the old and new
       levels, so the chart can only say "half the points are below the
       mean and half above". A frozen baseline lets it say the useful
       thing: these points are outside what the process used to do.

    2. **The run rules stop flooding.** With a centre line describing
       neither level, the shift rule flags essentially every point on
       both sides. Technically correct, practically useless.

    3. **Sigma inflation — much rarer than people claim.** Measured (see
       tests/test_spc.py), a step change contaminates exactly one moving
       range out of n-1 and the screening rule discards it, so sigma is
       unaffected by a step of *any* magnitude: a 25-sigma jump estimates
       the same sigma as a 3-sigma one. Gradual drift does inflate it,
       but the effect is second order in (drift per period / noise) and
       stays under about 5% until the per-period change reaches roughly
       half the period-to-period noise.

    So the honest summary: whole-series limits rarely hide a change. What
    they lose is the ability to say *what it changed from*.
    """
    mask = np.zeros(n, dtype=bool)
    if baseline is None:
        mask[:] = True
    elif isinstance(baseline, int):
        mask[: max(baseline, 2)] = True
    elif isinstance(baseline, slice):
        mask[baseline] = True
    else:
        supplied = np.asarray(list(baseline), dtype=bool)
        mask[: len(supplied)] = supplied
    if mask.sum() < 2:  # degenerate baseline: fall back to everything
        mask[:] = True
    return mask


def _safe_nanmean(a: np.ndarray) -> float:
    """nanmean without the 'Mean of empty slice' warning.

    An all-NaN column is normal here, not exceptional: filter a dashboard
    down to one small site and an indicator can legitimately have no
    computable periods. Returning NaN quietly is the right behaviour;
    a RuntimeWarning on every rerun is not.
    """
    a = np.asarray(a, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def _moving_range(values: np.ndarray) -> np.ndarray:
    """|x_i - x_{i-1}|, NaN-safe, first element NaN."""
    mr = np.full(values.shape, np.nan)
    mr[1:] = np.abs(np.diff(values))
    return mr


def _screened_mr_mean(mr: np.ndarray, screen: bool = True) -> float:
    """Mean moving range, optionally screening out inflated ranges.

    A single wild outlier contributes to *two* moving ranges and can
    noticeably inflate sigma_hat, widening the limits so the outlier that
    caused it no longer signals. The conventional guard (Nelson) is to
    drop moving ranges above 3.27 * mRbar and recompute once.
    """
    valid = mr[np.isfinite(mr)]
    if valid.size == 0:
        return float("nan")
    mrbar = float(np.mean(valid))
    if screen and mrbar > 0:
        keep = valid[valid <= D4_N2 * mrbar]
        if keep.size >= 1:
            mrbar = float(np.mean(keep))
    return mrbar


# ---------------------------------------------------------------------------
# Special-cause rule detection
# ---------------------------------------------------------------------------
RULE_TEXT = {
    1: "Single point beyond a process limit",
    2: "Run of consecutive points on one side of the centre line (shift)",
    3: "Consecutive points all rising or all falling (trend)",
    4: "Two out of three consecutive points in the outer third (zone A)",
    5: "Four out of five consecutive points beyond one sigma (zone B or beyond)",
    6: "Fifteen consecutive points hugging the centre line (zone C)",
    7: "Eight consecutive points with none near the centre line (mixture)",
    8: "Fourteen consecutive points alternating up and down (over-control)",
}

RULE_SETS: dict[str, dict] = {
    # Nelson (1984), Journal of Quality Technology 16(4)
    "nelson": {"rules": [1, 2, 3, 4, 5, 6, 7, 8], "run": 9, "trend": 6},
    # NHS "Making Data Count" reduced set -- fewer rules, shorter runs
    "nhs": {"rules": [1, 2, 3, 4], "run": 7, "trend": 7},
    # Rule 1 only -- for audiences who will not tolerate false alarms
    "core": {"rules": [1], "run": 8, "trend": 6},
}


def apply_rules(
    values: np.ndarray,
    cl: np.ndarray,
    sigma_series: np.ndarray,
    rule_set: RuleSet = "nhs",
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Flag special-cause points.

    Returns (special, direction, rules_per_point) where direction is
    +1 above the centre line, -1 below, 0 for none.

    ``sigma_series`` is the *point-specific* sigma. For an XmR chart it is
    constant; for a p-chart it varies with the denominator, which is why
    the zone rules have to be evaluated in standardised (z) space rather
    than against fixed horizontal bands.
    """
    n = len(values)
    cfg = RULE_SETS[rule_set]
    active = set(cfg["rules"])
    run_len, trend_len = cfg["run"], cfg["trend"]

    hits: list[list[int]] = [[] for _ in range(n)]
    finite = np.isfinite(values) & np.isfinite(cl) & np.isfinite(sigma_series) & (sigma_series > 0)

    # Standardised deviation from the centre line -- makes every rule
    # comparable across charts with variable limits.
    z = np.full(n, np.nan)
    z[finite] = (values[finite] - cl[finite]) / sigma_series[finite]
    side = np.sign(z)

    def _mark(i: int, rule: int) -> None:
        if rule in active and rule not in hits[i]:
            hits[i].append(rule)

    # Rule 1 -- beyond 3 sigma
    for i in np.where(finite & (np.abs(z) > 3))[0]:
        _mark(int(i), 1)

    # Rule 2 -- shift: run_len consecutive points on the same side.
    # Points exactly on the centre line are conventionally ignored (they
    # neither continue nor break a run).
    run_start, run_sign = 0, 0
    for i in range(n):
        s = side[i] if finite[i] else 0
        if s == 0:
            continue
        if s != run_sign:
            run_sign, run_start = s, i
        if i - run_start + 1 >= run_len:
            for j in range(run_start, i + 1):
                _mark(j, 2)

    # Rule 3 -- trend: trend_len consecutive strictly monotone points.
    # Ties break the run: a flat step is not evidence of a trend.
    if n >= trend_len:
        d = np.diff(values)
        direction = np.sign(d)
        start = 0
        for i in range(len(direction)):
            if i > 0 and (direction[i] != direction[i - 1] or direction[i] == 0):
                start = i
            if direction[i] != 0 and (i - start + 2) >= trend_len:
                for j in range(start, i + 2):
                    _mark(j, 3)

    # Rule 4 -- 2 of 3 consecutive in zone A (beyond 2 sigma), same side
    for i in range(2, n):
        w = z[i - 2 : i + 1]
        if not np.isfinite(w).all():
            continue
        for sgn in (1, -1):
            if np.sum(sgn * w > 2) >= 2 and sgn * z[i] > 2:
                for j in range(i - 2, i + 1):
                    if sgn * z[j] > 2:
                        _mark(j, 4)

    # Rule 5 -- 4 of 5 consecutive beyond 1 sigma, same side
    for i in range(4, n):
        w = z[i - 4 : i + 1]
        if not np.isfinite(w).all():
            continue
        for sgn in (1, -1):
            if np.sum(sgn * w > 1) >= 4 and sgn * z[i] > 1:
                for j in range(i - 4, i + 1):
                    if sgn * z[j] > 1:
                        _mark(j, 5)

    # Rule 6 -- 15 consecutive within 1 sigma (suspiciously well behaved:
    # usually stratification, a wrongly-sized subgroup, or fabricated data)
    for i in range(14, n):
        w = z[i - 14 : i + 1]
        if np.isfinite(w).all() and np.all(np.abs(w) < 1):
            for j in range(i - 14, i + 1):
                _mark(j, 6)

    # Rule 7 -- 8 consecutive with none within 1 sigma (two processes mixed)
    for i in range(7, n):
        w = z[i - 7 : i + 1]
        if np.isfinite(w).all() and np.all(np.abs(w) > 1):
            for j in range(i - 7, i + 1):
                _mark(j, 7)

    # Rule 8 -- 14 consecutive alternating (classic over-adjustment)
    if n >= 14:
        d = np.diff(values)
        for i in range(13, n):
            seg = d[i - 13 : i]
            if not np.isfinite(seg).all() or np.any(seg == 0):
                continue
            if np.all(np.sign(seg[1:]) == -np.sign(seg[:-1])):
                for j in range(i - 13, i + 1):
                    _mark(j, 8)

    special = np.array([len(h) > 0 for h in hits])
    direction = np.where(special & np.isfinite(side), side, 0).astype(int)
    return special, direction, hits


def _finalise(
    frame: pd.DataFrame,
    sigma_series: np.ndarray,
    rule_set: RuleSet,
) -> pd.DataFrame:
    values = frame["value"].to_numpy(dtype=float)
    cl = frame["cl"].to_numpy(dtype=float)
    special, direction, hits = apply_rules(values, cl, sigma_series, rule_set)
    frame = frame.copy()
    frame["sigma"] = sigma_series
    frame["ucl_2s"] = cl + 2 * sigma_series
    frame["lcl_2s"] = cl - 2 * sigma_series
    frame["ucl_1s"] = cl + 1 * sigma_series
    frame["lcl_1s"] = cl - 1 * sigma_series
    frame["special"] = special
    frame["special_dir"] = direction
    frame["rules"] = [sorted(h) for h in hits]
    frame["rule_text"] = [
        "; ".join(RULE_TEXT[r] for r in sorted(h)) if h else "" for h in hits
    ]
    return frame


# ---------------------------------------------------------------------------
# XmR / individuals chart
# ---------------------------------------------------------------------------
def xmr(
    values: Sequence[float],
    x: Sequence | None = None,
    *,
    target: float | None = None,
    higher_is_better: bool | None = None,
    baseline: slice | int | None = None,
    phases: Sequence[int] | None = None,
    rule_set: RuleSet = "nhs",
    screen_mr: bool = True,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    unit: str = "",
    label: str = "",
) -> ChartResult:
    """Individuals-and-moving-range chart for continuous measures.

    ``phases`` gives the index positions at which limits are recalculated
    (e.g. the month a new pathway went live). Each phase gets its own
    centre line and sigma. Use this only when you can name the change --
    recalculating limits to chase the data is how SPC gets discredited.

    ``lower_bound`` clamps the lower limit, which matters constantly in
    clinical timing data: a door-to-needle time cannot be negative, and a
    percentage cannot exceed 100. Clamping is cosmetic honesty, not
    statistics -- the underlying sigma is unchanged.
    """
    v = _as_array(values)
    n = len(v)
    idx = list(x) if x is not None else list(range(n))

    cut_points = [0] + sorted({int(p) for p in (phases or []) if 0 < int(p) < n}) + [n]
    cl = np.full(n, np.nan)
    sig = np.full(n, np.nan)
    phase_id = np.zeros(n, dtype=int)

    for pi, (start, stop) in enumerate(zip(cut_points[:-1], cut_points[1:])):
        seg = v[start:stop]
        phase_id[start:stop] = pi
        if np.isfinite(seg).sum() < 2:
            cl[start:stop] = np.nanmean(seg) if np.isfinite(seg).any() else np.nan
            continue
        bmask = _baseline_mask(len(seg), baseline if pi == 0 else None)
        calc = seg[bmask]
        centre = float(np.nanmean(calc))
        mrbar = _screened_mr_mean(_moving_range(calc), screen=screen_mr)
        cl[start:stop] = centre
        sig[start:stop] = mrbar / D2_N2 if np.isfinite(mrbar) else np.nan

    ucl = cl + 3 * sig
    lcl = cl - 3 * sig
    if lower_bound is not None:
        lcl = np.maximum(lcl, lower_bound)
    if upper_bound is not None:
        ucl = np.minimum(ucl, upper_bound)

    # Zero sigma is a real possibility (a constant series, or one so heavily
    # rounded that adjacent values never differ) and it silently disables
    # every rule, because a point cannot be "outside" a zero-width limit.
    # Surface it rather than reporting a serenely stable process.
    degenerate = bool(np.all(~np.isfinite(sig)) or np.nanmax(sig, initial=0.0) <= 0)

    mr = _moving_range(v)
    frame = pd.DataFrame(
        {
            "x": idx,
            "value": v,
            "cl": cl,
            "ucl": ucl,
            "lcl": lcl,
            "phase": phase_id,
            "mr": mr,
            "mr_cl": np.where(np.isfinite(sig), sig * D2_N2, np.nan),
            "mr_ucl": np.where(np.isfinite(sig), sig * D2_N2 * D4_N2, np.nan),
        }
    )
    frame = _finalise(frame, sig, rule_set)
    return ChartResult(
        frame=frame,
        chart_type="xmr",
        centre=_safe_nanmean(cl),
        sigma=_safe_nanmean(sig),
        target=target,
        higher_is_better=higher_is_better,
        unit=unit,
        label=label,
        meta={"rule_set": rule_set, "sigma_estimator": "average moving range / 1.128",
              "degenerate": degenerate,
              "degenerate_note": (
                  "Sigma is zero over the limit-calculation window: adjacent values "
                  "never differ. No rule can fire against a zero-width limit. This is "
                  "almost always rounding, a constant, or a field populated from a "
                  "default rather than a stable process." if degenerate else "")},
    )


# ---------------------------------------------------------------------------
# p-chart and Laney p'
# ---------------------------------------------------------------------------
def p_chart(
    numerator: Sequence[float],
    denominator: Sequence[float],
    x: Sequence | None = None,
    *,
    laney: bool | Literal["auto"] = "auto",
    as_percent: bool = True,
    target: float | None = None,
    higher_is_better: bool | None = None,
    baseline: slice | int | None = None,
    phases: Sequence[int] | None = None,
    rule_set: RuleSet = "nhs",
    unit: str = "%",
    label: str = "",
) -> ChartResult:
    """Proportion chart with optional Laney overdispersion correction.

    The classical p-chart assumes every case in a period is an independent
    Bernoulli trial with the *same* probability p. Sigma is then fixed by
    the binomial model:

        sigma_i = sqrt( pbar (1 - pbar) / n_i )

    In clinical audit that assumption almost never holds. Case mix varies
    between periods, patients cluster by consultant and by day of week,
    and the true probability drifts. The consequence is *overdispersion*:
    the observed scatter of the points exceeds what binomial sampling can
    produce. With large denominators the binomial limits collapse toward
    the centre line and practically every point signals -- which is not a
    detection triumph, it is a model failure.

    Laney (2002) fixes this without abandoning the chart. Standardise:

        z_i = (p_i - pbar) / sqrt( pbar(1-pbar)/n_i )

    If the binomial model were right, the z_i would behave like standard
    normal draws, so the short-term variation of z would be 1. Estimate
    that short-term variation the Shewhart way, from the moving range of z:

        sigma_z = mean(|z_i - z_{i-1}|) / 1.128

    and inflate the limits by that factor:

        limits_i = pbar +/- 3 * sigma_z * sqrt( pbar(1-pbar)/n_i )

    When sigma_z = 1 this is exactly the ordinary p-chart, so p' is a
    strict generalisation, not a different chart. sigma_z substantially
    above 1 is itself a finding: it says the periods are not exchangeable
    and you should be asking what differs between them.

    ``laney="auto"`` applies the correction whenever sigma_z > 1.1.
    ``clamp`` behaviour: sigma_z below 1 (underdispersion) is not used to
    *narrow* limits, because narrowing them manufactures false signals.
    """
    num = _as_array(numerator)
    den = _as_array(denominator)
    n = len(num)
    idx = list(x) if x is not None else list(range(n))
    scale = 100.0 if as_percent else 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(den > 0, num / den, np.nan)

    cut_points = [0] + sorted({int(q) for q in (phases or []) if 0 < int(q) < n}) + [n]
    cl = np.full(n, np.nan)
    sig = np.full(n, np.nan)
    phase_id = np.zeros(n, dtype=int)
    sigma_z_by_phase: dict[int, float] = {}

    for pi, (start, stop) in enumerate(zip(cut_points[:-1], cut_points[1:])):
        phase_id[start:stop] = pi
        seg_num, seg_den = num[start:stop], den[start:stop]
        bmask = _baseline_mask(stop - start, baseline if pi == 0 else None)
        ok = np.isfinite(seg_num) & np.isfinite(seg_den) & (seg_den > 0)
        calc = bmask & ok
        if calc.sum() == 0:
            continue

        # Centre line is the pooled proportion, NOT the mean of the
        # period proportions. Pooling weights each period by its own
        # denominator, which is what "the process rate" means. Averaging
        # the proportions gives a month with 3 cases the same influence
        # as a month with 300.
        pbar = float(np.nansum(seg_num[calc]) / np.nansum(seg_den[calc]))
        pbar = min(max(pbar, 1e-9), 1 - 1e-9)

        base_sigma = np.full(stop - start, np.nan)
        base_sigma[ok] = np.sqrt(pbar * (1 - pbar) / seg_den[ok])

        sigma_z = 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            seg_p = np.where(seg_den > 0, seg_num / seg_den, np.nan)
        z = np.where(ok & (base_sigma > 0), (seg_p - pbar) / base_sigma, np.nan)
        if np.isfinite(z).sum() >= 3:
            mrz = _screened_mr_mean(_moving_range(z), screen=False)
            if np.isfinite(mrz):
                sigma_z = mrz / D2_N2

        use_laney = laney is True or (laney == "auto" and sigma_z > 1.1)
        factor = max(sigma_z, 1.0) if use_laney else 1.0
        sigma_z_by_phase[pi] = sigma_z

        cl[start:stop] = pbar * scale
        sig[start:stop] = base_sigma * factor * scale

    ucl = np.minimum(cl + 3 * sig, 1.0 * scale)
    lcl = np.maximum(cl - 3 * sig, 0.0)

    frame = pd.DataFrame(
        {
            "x": idx,
            "value": p * scale,
            "numerator": num,
            "denominator": den,
            "cl": cl,
            "ucl": ucl,
            "lcl": lcl,
            "phase": phase_id,
        }
    )
    frame = _finalise(frame, sig, rule_set)
    sigma_z_overall = sigma_z_by_phase.get(0, 1.0)
    return ChartResult(
        frame=frame,
        chart_type="p-prime" if sigma_z_overall > 1.1 and laney in (True, "auto") else "p",
        centre=_safe_nanmean(cl),
        sigma=None,
        target=target,
        higher_is_better=higher_is_better,
        unit=unit,
        label=label,
        meta={
            "rule_set": rule_set,
            "sigma_z": sigma_z_overall,
            "sigma_z_by_phase": sigma_z_by_phase,
            "overdispersed": sigma_z_overall > 1.1,
            "sigma_estimator": "binomial, Laney-corrected" if sigma_z_overall > 1.1 else "binomial",
            "mean_denominator": float(np.nanmean(den)),
        },
    )


# ---------------------------------------------------------------------------
# u-chart (rate per unit exposure) and c-chart (count)
# ---------------------------------------------------------------------------
def u_chart(
    counts: Sequence[float],
    exposure: Sequence[float],
    x: Sequence | None = None,
    *,
    multiplier: float = 1000.0,
    laney: bool | Literal["auto"] = "auto",
    target: float | None = None,
    higher_is_better: bool | None = None,
    baseline: slice | int | None = None,
    rule_set: RuleSet = "nhs",
    unit: str = "per 1,000 bed days",
    label: str = "",
) -> ChartResult:
    """Events per unit of exposure -- falls per 1,000 bed days, and similar.

    Poisson model: sigma_i = sqrt(ubar / n_i) where n_i is the exposure.
    The Laney u' correction is the exact analogue of p': standardise,
    take the moving range of z, inflate. Overdispersion is if anything
    *more* common here, because events cluster (one confused patient
    falls five times in a week).

    Exposure must be a real denominator of opportunity. Bed days, not
    admissions: a fall rate per admission conflates 'falls more often'
    with 'stays longer'.
    """
    c = _as_array(counts)
    e = _as_array(exposure)
    n = len(c)
    idx = list(x) if x is not None else list(range(n))

    ok = np.isfinite(c) & np.isfinite(e) & (e > 0)
    bmask = _baseline_mask(n, baseline)
    calc = ok & bmask
    ubar = float(np.nansum(c[calc]) / np.nansum(e[calc])) if calc.any() else np.nan

    base_sigma = np.full(n, np.nan)
    base_sigma[ok] = np.sqrt(ubar / e[ok]) if np.isfinite(ubar) and ubar > 0 else np.nan

    rate = np.where(ok, c / e, np.nan)
    sigma_z = 1.0
    z = np.where(ok & (base_sigma > 0), (rate - ubar) / base_sigma, np.nan)
    if np.isfinite(z).sum() >= 3:
        mrz = _screened_mr_mean(_moving_range(z), screen=False)
        if np.isfinite(mrz):
            sigma_z = mrz / D2_N2
    use_laney = laney is True or (laney == "auto" and sigma_z > 1.1)
    factor = max(sigma_z, 1.0) if use_laney else 1.0

    cl = np.full(n, ubar * multiplier)
    sig = base_sigma * factor * multiplier
    ucl = cl + 3 * sig
    lcl = np.maximum(cl - 3 * sig, 0.0)

    frame = pd.DataFrame(
        {
            "x": idx,
            "value": rate * multiplier,
            "numerator": c,
            "denominator": e,
            "cl": cl,
            "ucl": ucl,
            "lcl": lcl,
            "phase": 0,
        }
    )
    frame = _finalise(frame, sig, rule_set)
    return ChartResult(
        frame=frame,
        chart_type="u-prime" if use_laney else "u",
        centre=float(ubar * multiplier),
        sigma=None,
        target=target,
        higher_is_better=higher_is_better,
        unit=unit,
        label=label,
        meta={
            "rule_set": rule_set,
            "sigma_z": sigma_z,
            "overdispersed": sigma_z > 1.1,
            "multiplier": multiplier,
            "sigma_estimator": "Poisson, Laney-corrected" if use_laney else "Poisson",
        },
    )


def c_chart(
    counts: Sequence[float],
    x: Sequence | None = None,
    *,
    target: float | None = None,
    higher_is_better: bool | None = None,
    baseline: slice | int | None = None,
    rule_set: RuleSet = "nhs",
    unit: str = "count",
    label: str = "",
) -> ChartResult:
    """Counts with a constant area of opportunity. sigma = sqrt(cbar)."""
    c = _as_array(counts)
    n = len(c)
    idx = list(x) if x is not None else list(range(n))
    calc = np.isfinite(c) & _baseline_mask(n, baseline)
    cbar = float(np.nanmean(c[calc])) if calc.any() else np.nan
    sig = np.full(n, np.sqrt(cbar) if np.isfinite(cbar) else np.nan)
    cl = np.full(n, cbar)
    frame = pd.DataFrame(
        {"x": idx, "value": c, "cl": cl, "ucl": cl + 3 * sig,
         "lcl": np.maximum(cl - 3 * sig, 0.0), "phase": 0}
    )
    frame = _finalise(frame, sig, rule_set)
    return ChartResult(frame=frame, chart_type="c", centre=cbar, sigma=float(np.sqrt(cbar)),
                       target=target, higher_is_better=higher_is_better, unit=unit, label=label,
                       meta={"rule_set": rule_set, "sigma_estimator": "Poisson sqrt(cbar)"})


# ---------------------------------------------------------------------------
# Rare-event charts: t and g
# ---------------------------------------------------------------------------
def t_chart(
    days_between: Sequence[float],
    x: Sequence | None = None,
    *,
    baseline: slice | int | None = None,
    rule_set: RuleSet = "core",
    unit: str = "days between events",
    label: str = "",
) -> ChartResult:
    """Time between rare adverse events.

    When an event is rare -- a stroke-unit pressure ulcer, an in-hospital
    fall with harm -- a monthly count chart is mostly zeros. cbar is tiny,
    the lower limit is pinned at zero, and the chart can only ever signal
    upward. It cannot show you improvement at all.

    The t-chart flips the axis: plot the *interval* between successive
    events. Improvement now shows as points drifting up, which is both
    detectable and motivating ("214 days since the last one").

    Intervals are strongly right-skewed (roughly exponential/Weibull), so
    3-sigma limits on the raw scale are nonsense. Nelson's transformation

        y = t ** (1 / 3.6)

    maps a Weibull to approximate normality; fit XmR on y, then map the
    limits back with y ** 3.6. The centre line after back-transformation
    is a median-like quantity, not the arithmetic mean interval -- report
    it as such.
    """
    t = _as_array(days_between)
    t = np.where(t <= 0, np.nan, t)  # a zero interval is a data error
    y = np.power(t, 1 / 3.6)
    inner = xmr(y, x=x, baseline=baseline, rule_set=rule_set, lower_bound=0.0)
    f = inner.frame.copy()
    for col in ("value", "cl", "ucl", "lcl", "ucl_2s", "lcl_2s", "ucl_1s", "lcl_1s"):
        f[col] = np.power(np.maximum(f[col].to_numpy(dtype=float), 0.0), 3.6)
    return ChartResult(
        frame=f, chart_type="t", centre=float(np.nanmean(f["cl"])), sigma=None,
        target=None, higher_is_better=True, unit=unit, label=label,
        meta={"rule_set": rule_set, "transform": "Nelson y = t^(1/3.6)",
              "sigma_estimator": "XmR on transformed scale, back-transformed"},
    )


def g_chart(
    opportunities_between: Sequence[float],
    x: Sequence | None = None,
    *,
    rule_set: RuleSet = "core",
    unit: str = "cases between events",
    label: str = "",
) -> ChartResult:
    """Number of *opportunities* (e.g. admissions) between rare events.

    Geometric distribution with mean gbar, so sigma = sqrt(gbar(gbar+1)).
    Preferred over the t-chart when workload varies a lot: 90 days between
    falls means something different in a busy month than a quiet one.
    The centre line uses the median, because the geometric is so skewed
    that the mean sits well above the typical value.
    """
    g = _as_array(opportunities_between)
    n = len(g)
    idx = list(x) if x is not None else list(range(n))
    gbar = float(np.nanmean(g))
    gmed = float(np.nanmedian(g))
    sigma = float(np.sqrt(gbar * (gbar + 1))) if np.isfinite(gbar) else np.nan
    cl = np.full(n, gmed)
    sig = np.full(n, sigma)
    frame = pd.DataFrame(
        {"x": idx, "value": g, "cl": cl, "ucl": np.full(n, gbar + 3 * sigma),
         "lcl": np.full(n, max(0.0, gbar - 3 * sigma)), "phase": 0}
    )
    frame = _finalise(frame, sig, rule_set)
    return ChartResult(frame=frame, chart_type="g", centre=gmed, sigma=sigma,
                       target=None, higher_is_better=True, unit=unit, label=label,
                       meta={"rule_set": rule_set, "mean": gbar,
                             "sigma_estimator": "geometric sqrt(gbar(gbar+1))",
                             "note": "centre line is the median; limits use the mean"})


# ---------------------------------------------------------------------------
# Funnel plot
# ---------------------------------------------------------------------------
def funnel_plot(
    numerator: Sequence[float],
    denominator: Sequence[float],
    labels: Sequence[str],
    *,
    overdispersion: bool = True,
    exact: bool = True,
    as_percent: bool = True,
    n_grid: int = 220,
) -> dict:
    """Cross-sectional comparison of units (sites, wards, consultants).

    A league table ranks units by a point estimate and hides the fact that
    a small unit's estimate is mostly noise. Someone is always bottom.

    The funnel plot fixes that by plotting the indicator against the
    denominator, with control limits that widen as volume falls -- the
    funnel shape. Units inside the funnel are consistent with the common
    rate; only points outside it warrant explanation. There is no ranking
    and no implied ordering.

    Two refinements matter:

    * **Exact limits.** With small denominators the normal approximation
      is poor and gives limits below 0 or above 1. Using binomial
      quantiles directly is both correct and simple.

    * **Overdispersion.** Real provider data is more variable than
      binomial, because case mix and local context differ. Spiegelhalter's
      multiplicative adjustment estimates a dispersion factor phi as the
      mean of the squared z-scores, winsorised at the 10th/90th centiles
      so that genuine outliers do not inflate the very limits meant to
      detect them, and scales the limits by sqrt(phi). phi close to 1
      means binomial variation is sufficient. phi of 3 or 4 means most of
      the spread between units is systematic, and the honest reading is
      "these units are not doing the same thing", not "these 9 units are
      all outliers".
    """
    num = _as_array(numerator)
    den = _as_array(denominator)
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    scale = 100.0 if as_percent else 1.0

    theta = float(np.sum(num[ok]) / np.sum(den[ok]))
    theta = min(max(theta, 1e-9), 1 - 1e-9)
    prop = np.where(ok, num / den, np.nan)

    z = np.where(ok, (prop - theta) / np.sqrt(theta * (1 - theta) / np.where(ok, den, 1)), np.nan)
    phi = 1.0
    zfin = z[np.isfinite(z)]
    if overdispersion and zfin.size >= 4:
        lo, hi = np.percentile(zfin, [10, 90])
        wz = np.clip(zfin, lo, hi)
        phi = float(np.sum(wz ** 2) / zfin.size)
    inflate = float(np.sqrt(max(phi, 1.0))) if overdispersion else 1.0

    lo_n = max(float(np.nanmin(den[ok])) * 0.6, 1.0)
    hi_n = float(np.nanmax(den[ok])) * 1.15
    grid = np.unique(np.round(np.linspace(lo_n, hi_n, n_grid)).astype(int))
    grid = grid[grid >= 1]

    bands = {}
    for alpha, name in ((0.05, "95"), (0.002, "998")):
        if exact and inflate == 1.0:
            lower = stats.binom.ppf(alpha / 2, grid, theta) / grid
            upper = stats.binom.ppf(1 - alpha / 2, grid, theta) / grid
        else:
            zc = stats.norm.ppf(1 - alpha / 2)
            se = inflate * np.sqrt(theta * (1 - theta) / grid)
            lower = np.clip(theta - zc * se, 0, 1)
            upper = np.clip(theta + zc * se, 0, 1)
        bands[name] = (lower * scale, upper * scale)

    # classify each unit against the 99.8% band
    lo998 = np.interp(den, grid, bands["998"][0] / scale)
    hi998 = np.interp(den, grid, bands["998"][1] / scale)
    lo95 = np.interp(den, grid, bands["95"][0] / scale)
    hi95 = np.interp(den, grid, bands["95"][1] / scale)
    status = np.full(len(den), "within", dtype=object)
    status[prop > hi95] = "above 95%"
    status[prop < lo95] = "below 95%"
    status[prop > hi998] = "above 99.8%"
    status[prop < lo998] = "below 99.8%"

    return {
        "points": pd.DataFrame(
            {"label": list(labels), "denominator": den, "numerator": num,
             "value": prop * scale, "status": status}
        ),
        "grid": grid,
        "bands": bands,
        "centre": theta * scale,
        "phi": phi,
        "inflated": inflate > 1.0,
        "inflation": inflate,
    }


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------
def cusum(
    values: Sequence[float],
    x: Sequence | None = None,
    *,
    target_mean: float | None = None,
    sigma: float | None = None,
    k_sigma: float = 0.5,
    h_sigma: float = 4.0,
    reset_on_signal: bool = True,
) -> pd.DataFrame:
    """Tabular CUSUM for persistent small shifts.

    A Shewhart chart looks at one point at a time, so it is fast at
    catching big jumps and slow at catching small sustained drifts -- and
    small sustained drifts are exactly what service deterioration looks
    like. The CUSUM accumulates evidence instead:

        C+_i = max(0, C+_{i-1} + (x_i - mu0) - k)
        C-_i = max(0, C-_{i-1} - (x_i - mu0) - k)

    with the reference value k set to half the shift you care about
    (k = 0.5 sigma detects a 1-sigma shift efficiently) and the decision
    interval h = 4 or 5 sigma giving an in-control run length comparable
    to a 3-sigma Shewhart chart. Subtracting k is what makes it a
    detector rather than a random walk: under no shift the increments are
    negative on average, so the statistic sits at its zero barrier.

    The price is interpretability. A CUSUM tells you *when* accumulated
    evidence crossed a threshold, not what the rate is now, and the point
    of crossing lags the change. Run it beside the Shewhart chart, never
    instead of it.
    """
    v = _as_array(values)
    n = len(v)
    idx = list(x) if x is not None else list(range(n))
    mu0 = float(np.nanmean(v)) if target_mean is None else float(target_mean)
    if sigma is None:
        sigma = _screened_mr_mean(_moving_range(v)) / D2_N2
    sigma = float(sigma) if np.isfinite(sigma) and sigma > 0 else 1.0

    k, h = k_sigma * sigma, h_sigma * sigma
    cp = np.zeros(n)
    cm = np.zeros(n)
    sig_up = np.zeros(n, dtype=bool)
    sig_dn = np.zeros(n, dtype=bool)
    a = b = 0.0
    for i in range(n):
        xi = v[i]
        if not np.isfinite(xi):
            cp[i], cm[i] = a, b
            continue
        a = max(0.0, a + (xi - mu0) - k)
        b = max(0.0, b - (xi - mu0) - k)
        if a > h:
            sig_up[i] = True
            if reset_on_signal:
                a = 0.0
        if b > h:
            sig_dn[i] = True
            if reset_on_signal:
                b = 0.0
        cp[i], cm[i] = a, b

    return pd.DataFrame(
        {"x": idx, "value": v, "cusum_high": cp, "cusum_low": -cm,
         "h": h, "signal_high": sig_up, "signal_low": sig_dn,
         "mu0": mu0, "sigma": sigma, "k": k}
    )


# ---------------------------------------------------------------------------
# Change-point suggestion
# ---------------------------------------------------------------------------
def suggest_phase_break(values: Sequence[float], min_segment: int = 8) -> dict | None:
    """Locate the most likely single step change in the mean.

    Why this is needed. When a real step change sits inside the series,
    limits calculated over the whole series straddle both levels: the
    centre line lands between them and the run rules then flag *most of
    the series* as special cause. That output is technically correct and
    practically useless -- it says "something changed" without saying
    when, and it invites people to recalculate limits until the chart
    looks calm. (Sigma itself often survives a single sharp step, because
    only one moving range is contaminated and the screening rule discards
    it; a gradual drift is the case where sigma really does inflate.)

    So: find the split point that maximises the standardised difference
    between the two segment means (a Welch t statistic), which is
    equivalent to binary segmentation with a mean-shift cost. Then judge
    whether the split is worth making. The threshold used here (|t| >= 3)
    is deliberately conservative -- it is not a hypothesis test, because
    the split point was chosen by looking at the data, and the null
    distribution of a maximised statistic is not Student's t. Treat the
    output as a prompt to ask "did something happen around then?", never
    as evidence on its own.

    The rule for acting on it: recalculate limits only when you can name
    the change and date it independently of this chart. A phase break you
    cannot explain is curve-fitting.
    """
    v = _as_array(values)
    finite = np.isfinite(v)
    if finite.sum() < 2 * min_segment:
        return None

    best = None
    for cut in range(min_segment, len(v) - min_segment + 1):
        a, b = v[:cut][np.isfinite(v[:cut])], v[cut:][np.isfinite(v[cut:])]
        if len(a) < min_segment or len(b) < min_segment:
            continue
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        se = np.sqrt(va / len(a) + vb / len(b))
        if not np.isfinite(se) or se == 0:
            continue
        t = (np.mean(b) - np.mean(a)) / se
        if best is None or abs(t) > abs(best["t"]):
            best = {"index": cut, "t": float(t),
                    "mean_before": float(np.mean(a)), "mean_after": float(np.mean(b)),
                    "shift": float(np.mean(b) - np.mean(a))}
    if best is None:
        return None
    best["worth_splitting"] = abs(best["t"]) >= 3.0
    return best


# ---------------------------------------------------------------------------
# Run chart (for when limits are not defensible)
# ---------------------------------------------------------------------------
def run_chart(values: Sequence[float], x: Sequence | None = None) -> pd.DataFrame:
    """Median-based run chart with the probability-based run rules.

    Useful when you have fewer than ~12 points, or when the audience will
    not accept control limits. Rules (Perla et al., BMJ Qual Saf 2011):

    * a shift  -- 6+ consecutive points all above or all below the median
    * a trend  -- 5+ consecutive points all going up or all going down
    * too few or too many runs, judged against a lookup based on the
      binomial distribution of run counts
    * an astronomical point -- judged by eye, deliberately not automated

    Points *on* the median are ignored for shift counting and do not
    break a run, which is why the effective n excludes them.
    """
    v = _as_array(values)
    n = len(v)
    idx = list(x) if x is not None else list(range(n))
    med = float(np.nanmedian(v))
    side = np.sign(v - med)
    useful = side[np.isfinite(side) & (side != 0)]
    n_useful = len(useful)
    n_runs = 1 + int(np.sum(useful[1:] != useful[:-1])) if n_useful else 0

    # Normal approximation to the run-count distribution (Swed & Eisenhart)
    mu_runs = n_useful / 2 + 1
    sd_runs = np.sqrt((n_useful - 1) / 4) if n_useful > 1 else np.nan
    lower_runs = mu_runs - 1.96 * sd_runs if np.isfinite(sd_runs) else np.nan
    upper_runs = mu_runs + 1.96 * sd_runs if np.isfinite(sd_runs) else np.nan

    shift = np.zeros(n, dtype=bool)
    start, cur = 0, 0
    for i in range(n):
        s = side[i] if np.isfinite(side[i]) else 0
        if s == 0:
            continue
        if s != cur:
            cur, start = s, i
        if i - start + 1 >= 6:
            shift[start : i + 1] = True

    trend = np.zeros(n, dtype=bool)
    d = np.sign(np.diff(v))
    st = 0
    for i in range(len(d)):
        if i > 0 and (d[i] != d[i - 1] or d[i] == 0):
            st = i
        if d[i] != 0 and (i - st + 2) >= 5:
            trend[st : i + 2] = True

    return pd.DataFrame(
        {"x": idx, "value": v, "median": med, "shift": shift, "trend": trend,
         "n_runs": n_runs, "runs_lower": lower_runs, "runs_upper": upper_runs,
         "too_few_runs": bool(np.isfinite(lower_runs) and n_runs < lower_runs),
         "too_many_runs": bool(np.isfinite(upper_runs) and n_runs > upper_runs)}
    )
