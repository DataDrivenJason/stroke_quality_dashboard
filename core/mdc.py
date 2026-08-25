"""
NHS "Making Data Count" variation and assurance icons.
=====================================================================

The problem this solves
-----------------------
Board and directorate reports are still dominated by a table of numbers
with an arrow against last month and a RAG colour against target. Both
are statistically illiterate. Month-on-month comparison of two points
drawn from a stable process is pure noise -- a stable process goes up
about half the time -- and RAG rating a single point against a target
tells you about that point, not about the process that produced it.

Making Data Count replaces both with two orthogonal questions, each
answered by an icon:

1. **Variation.** Is the process stable, or has something changed?
   Answered by the control chart rules. Nothing to do with the target.

2. **Assurance.** Given how this process actually behaves, can it be
   relied on to meet the target? Answered by comparing the *process
   limits* against the target, not the latest point against the target.

Keeping them separate is the whole point. A process can be perfectly
stable and reliably fail its target (common cause + fail) -- that is a
redesign problem, and no amount of performance management of the current
system will fix it. A process can be improving and still failing
(special cause improvement + fail) -- that is a project working, leave it
alone. A process can be hitting target unpredictably (hit and miss) --
that is a process worth stabilising before anyone claims success.

Assurance logic in full
-----------------------
For a metric where high is good, with process limits [LCL, UCL]:

    LCL >= target   ->  PASS      the process cannot ordinarily fail
    UCL <  target   ->  FAIL      the process cannot ordinarily pass
    otherwise       ->  HIT/MISS  the target sits inside natural variation

Mirror it when low is good. Note what this implies: a "pass" is a
statement about the *system*, and it survives a single bad month. That is
correct behaviour, and it is the main reason clinicians trust these
icons more than RAG.

Reference: NHS England, "Making Data Count" (2019 onward).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .spc import ChartResult

Variation = Literal[
    "common", "high_improve", "high_concern", "high_neutral",
    "low_improve", "low_concern", "low_neutral", "insufficient",
]
Assurance = Literal["pass", "fail", "hit_miss", "none"]

# Status palette (fixed, never themed) -- see the data-viz reference palette.
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"
NEUTRAL = "#898781"
INK = "#52514e"


@dataclass
class MDCVerdict:
    variation: Variation
    assurance: Assurance
    variation_label: str
    variation_detail: str
    assurance_label: str
    assurance_detail: str
    action: str
    last_value: float | None
    rules_fired: list[int]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_variation(result: ChartResult, lookback: int = 1) -> tuple[Variation, list[int], int]:
    """Classify the *recent* behaviour of the chart.

    ``lookback`` controls how many of the most recent points are inspected.
    The NHS convention is to icon the latest point, but a run rule by
    definition flags a block of points ending at or before the latest, so
    a lookback of 1 with run rules already captures "we are currently in
    a shift". Widening the lookback catches a signal that has just ended,
    which is usually not what a board wants to see.
    """
    frame = result.frame
    valid = frame[frame["value"].notna()]
    if len(valid) < 8:
        return "insufficient", [], 0

    tail = valid.tail(max(1, lookback))
    rules: list[int] = sorted({r for rs in tail["rules"] for r in rs})
    if not rules:
        return "common", [], 0

    # Direction: which side of the centre line the signalling points sit.
    signalling = tail[tail["special"]]
    direction = int(np.sign(signalling["special_dir"].sum())) if len(signalling) else 0
    if direction == 0:
        direction = int(np.sign(tail["value"].iloc[-1] - tail["cl"].iloc[-1]))

    hib = result.higher_is_better
    side = "high" if direction > 0 else "low"
    if hib is None:
        return f"{side}_neutral", rules, direction  # type: ignore[return-value]
    good = (direction > 0) == bool(hib)
    return f"{side}_{'improve' if good else 'concern'}", rules, direction  # type: ignore[return-value]


def classify_assurance(result: ChartResult) -> Assurance:
    """Compare the process limits with the target, not the last point."""
    target = result.target
    if target is None:
        return "none"
    valid = result.frame[result.frame["value"].notna()]
    if valid.empty:
        return "none"

    # For variable-width limits (p-charts) use the limits at the most
    # recent point: that is the process as it currently operates, at its
    # current volume. Averaging limits across periods with very different
    # denominators would describe a process that never existed.
    last = valid.iloc[-1]
    lcl, ucl = float(last["lcl"]), float(last["ucl"])
    if not (np.isfinite(lcl) and np.isfinite(ucl)):
        return "none"

    hib = result.higher_is_better
    if hib is None:
        return "none"
    if hib:
        if lcl >= target:
            return "pass"
        if ucl < target:
            return "fail"
    else:
        if ucl <= target:
            return "pass"
        if lcl > target:
            return "fail"
    return "hit_miss"


VARIATION_TEXT: dict[str, tuple[str, str]] = {
    "common": ("Common cause", "Natural variation. No evidence anything has changed."),
    "high_improve": ("Special cause — improving (high)", "Values significantly higher than the process baseline, and higher is better here."),
    "high_concern": ("Special cause — concern (high)", "Values significantly higher than the process baseline, and higher is worse here."),
    "high_neutral": ("Special cause (high)", "Values significantly higher than the process baseline. No better/worse direction is defined."),
    "low_improve": ("Special cause — improving (low)", "Values significantly lower than the process baseline, and lower is better here."),
    "low_concern": ("Special cause — concern (low)", "Values significantly lower than the process baseline, and lower is worse here."),
    "low_neutral": ("Special cause (low)", "Values significantly lower than the process baseline. No better/worse direction is defined."),
    "insufficient": ("Not enough data", "Fewer than 8 usable points. Limits are unstable; read as a run chart."),
}

ASSURANCE_TEXT: dict[str, tuple[str, str]] = {
    "pass": ("Consistently meets target", "The whole range of natural variation sits on the target side. Barring a change to the system, this target will keep being met."),
    "fail": ("Consistently misses target", "The whole range of natural variation sits on the wrong side. This system cannot deliver the target as designed — performance management will not close the gap."),
    "hit_miss": ("Hit and miss", "The target lies inside the range of natural variation, so whether it is met in any given period is largely chance."),
    "none": ("No target set", "Assurance cannot be judged without an agreed target and direction."),
}

ACTION_TEXT: dict[tuple[str, str], str] = {
    ("common", "pass"): "Nothing to do. Keep monitoring; resist the urge to explain individual points.",
    ("common", "fail"): "Redesign required. The system is stable at the wrong level — investigate the process, not the months.",
    ("common", "hit_miss"): "Reduce variation first. Stabilise before chasing the target.",
    ("common", "none"): "Agree a target and a direction so assurance can be judged.",
}


def verdict(result: ChartResult, lookback: int = 1) -> MDCVerdict:
    var, rules, _dir = classify_variation(result, lookback=lookback)
    ass = classify_assurance(result)
    vl, vd = VARIATION_TEXT[var]
    al, ad = ASSURANCE_TEXT[ass]

    action = ACTION_TEXT.get((var, ass))
    if action is None:
        if var.endswith("improve"):
            action = "Something is working. Find out what, and hold the gain — do not reset the limits until the change is confirmed."
        elif var.endswith("concern"):
            action = "Investigate now. This is a genuine signal, not month-to-month noise."
        elif var.endswith("neutral"):
            action = "The process has changed. Establish whether the change is wanted before acting."
        else:
            action = "Collect more points before drawing conclusions."

    last = result.last_valid()
    return MDCVerdict(
        variation=var, assurance=ass,
        variation_label=vl, variation_detail=vd,
        assurance_label=al, assurance_detail=ad,
        action=action,
        last_value=None if last is None else float(last["value"]),
        rules_fired=rules,
    )


# ---------------------------------------------------------------------------
# Icons -- inline SVG so they render anywhere, including PDF export
# ---------------------------------------------------------------------------
def _wrap(inner: str, size: int, title: str) -> str:
    return (
        f'<svg role="img" aria-label="{title}" width="{size}" height="{size}" '
        f'viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">'
        f"<title>{title}</title>{inner}</svg>"
    )


def variation_icon(variation: Variation, size: int = 34) -> str:
    """Circle + glyph. Colour never carries the meaning alone: the glyph
    (wave / up-arrow / down-arrow) and the accompanying label do."""
    colour = {
        "common": NEUTRAL,
        "high_improve": GOOD, "low_improve": GOOD,
        "high_concern": CRITICAL, "low_concern": CRITICAL,
        "high_neutral": INK, "low_neutral": INK,
        "insufficient": NEUTRAL,
    }[variation]
    ring = f'<circle cx="20" cy="20" r="18" fill="none" stroke="{colour}" stroke-width="2.5"/>'

    if variation == "common":
        glyph = (f'<path d="M8 24 q4 -8 8 0 t8 0 t8 0" fill="none" stroke="{colour}" '
                 f'stroke-width="2.5" stroke-linecap="round"/>'
                 f'<path d="M8 16 q4 -8 8 0 t8 0 t8 0" fill="none" stroke="{colour}" '
                 f'stroke-width="2.5" stroke-linecap="round" opacity="0.45"/>')
    elif variation == "insufficient":
        glyph = (f'<text x="20" y="27" text-anchor="middle" font-size="19" font-weight="700" '
                 f'fill="{colour}" font-family="system-ui,sans-serif">?</text>')
    else:
        up = variation.startswith("high")
        arrow = ("M20 9 L20 31 M12 17 L20 9 L28 17" if up
                 else "M20 31 L20 9 M12 23 L20 31 L28 23")
        glyph = (f'<path d="{arrow}" fill="none" stroke="{colour}" stroke-width="3" '
                 f'stroke-linecap="round" stroke-linejoin="round"/>')
    return _wrap(ring + glyph, size, VARIATION_TEXT[variation][0])


def assurance_icon(assurance: Assurance, size: int = 34) -> str:
    """Target-shaped icons. Pass/fail/hit-and-miss are distinguished by
    shape (filled bullseye / crossed / half-filled), not by colour."""
    colour = {"pass": GOOD, "fail": CRITICAL, "hit_miss": WARNING, "none": NEUTRAL}[assurance]
    ring = f'<circle cx="20" cy="20" r="18" fill="none" stroke="{colour}" stroke-width="2.5"/>'

    if assurance == "pass":
        glyph = (f'<circle cx="20" cy="20" r="11" fill="none" stroke="{colour}" stroke-width="2.2"/>'
                 f'<circle cx="20" cy="20" r="4.5" fill="{colour}"/>')
    elif assurance == "fail":
        glyph = (f'<circle cx="20" cy="20" r="11" fill="none" stroke="{colour}" stroke-width="2.2" '
                 f'opacity="0.45"/>'
                 f'<path d="M13 13 L27 27 M27 13 L13 27" stroke="{colour}" stroke-width="3" '
                 f'stroke-linecap="round"/>')
    elif assurance == "hit_miss":
        glyph = (f'<circle cx="20" cy="20" r="11" fill="none" stroke="{colour}" stroke-width="2.2"/>'
                 f'<path d="M20 9 A11 11 0 0 1 20 31 Z" fill="{colour}"/>')
    else:
        glyph = (f'<path d="M11 20 L29 20" stroke="{colour}" stroke-width="3" stroke-linecap="round"/>')
    return _wrap(ring + glyph, size, ASSURANCE_TEXT[assurance][0])


def icon_pair_html(v: MDCVerdict, size: int = 30) -> str:
    """Both icons side by side, each with its text label underneath.

    The label is not optional. An icon alone is colour-and-shape encoding;
    the pairing with a text label is what makes it accessible and what
    stops a reader inventing their own meaning for the glyph.
    """
    return (
        '<div style="display:flex;gap:18px;align-items:flex-start;">'
        f'<div style="display:flex;gap:8px;align-items:center;">{variation_icon(v.variation, size)}'
        f'<span style="font-size:0.78rem;line-height:1.25;">{v.variation_label}</span></div>'
        f'<div style="display:flex;gap:8px;align-items:center;">{assurance_icon(v.assurance, size)}'
        f'<span style="font-size:0.78rem;line-height:1.25;">{v.assurance_label}</span></div>'
        "</div>"
    )
