"""
Metric registry with a switchable standards layer.
=====================================================================

Design intent
-------------
The *computation* of an indicator and the *threshold* it is judged
against are different things, owned by different people, and they change
on different clocks. A door-to-needle time is a door-to-needle time in
Dublin and in Leeds; what differs is the denominator convention, the
exclusion rules and the target. So this module separates them:

    MetricSpec   -- what is counted, on what denominator, on which chart
    Threshold    -- the target/ambition a given standard applies to it

Switching standard therefore re-labels and re-targets the whole
dashboard without touching a line of calculation. A metric that a
standard does not recognise simply has no threshold under it and is
hidden by default.

A warning about targets
-----------------------
The threshold values below are *sensible working defaults*, not a
transcription of any audit's current technical guidance. National audit
thresholds move -- SSNAP has re-based its domain scoring more than once,
and Irish national stroke reporting has shifted as thrombectomy access
expanded. Before anything from this dashboard reaches a board paper,
open the current year's technical guidance and confirm every number in
the THRESHOLDS tables. They are deliberately kept in one place, and are
editable at runtime from the sidebar, precisely so this is a five-minute
job rather than a code change.

Clock start
-----------
Every timing indicator here measures from *clock start*, defined as:
    - time of arrival at the first hospital, for patients who arrive by
      any route with symptoms already present;
    - time of symptom onset, for in-hospital strokes.
This matters more than any threshold. A service that measures from
"time first seen by the stroke team" will look excellent and be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

Standard = Literal["SSNAP", "INAS"]

DOMAINS = [
    "Hyperacute pathway",
    "Stroke unit & ward care",
    "Therapy & rehabilitation",
    "Safety",
    "Secondary prevention & discharge",
    "Outcomes",
]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    domain: str
    unit: str
    chart: Literal["p", "xmr", "u"]
    higher_is_better: bool | None
    # For proportion (p) charts: names of boolean columns on the admissions frame
    numerator: str | None = None
    denominator: str | None = None
    # For continuous (XmR) charts
    value_col: str | None = None
    agg: Literal["median", "mean"] = "median"
    # For rate (u) charts
    count_col: str | None = None
    exposure_col: str | None = None
    multiplier: float = 1000.0
    # Documentation shown in the UI
    definition: str = ""
    why: str = ""
    caveat: str = ""
    min_denominator: int = 5


@dataclass(frozen=True)
class Threshold:
    target: float | None
    ambition: float | None = None
    label_override: str | None = None
    provenance: str = ""


# ---------------------------------------------------------------------------
# Metric definitions -- standard-independent
# ---------------------------------------------------------------------------
def _m(**kw) -> MetricSpec:
    return MetricSpec(**kw)


METRICS: dict[str, MetricSpec] = {m.key: m for m in [

    # ---------------- Hyperacute pathway ----------------
    _m(key="ct_60", label="Brain imaging within 1 hour of clock start",
       domain="Hyperacute pathway", unit="%", chart="p", higher_is_better=True,
       numerator="flag_ct_60", denominator="den_all_stroke",
       definition="Patients whose first brain scan (CT or MRI) was performed within 60 minutes of clock start, as a proportion of all confirmed strokes.",
       why="Imaging is the gate on every hyperacute decision. Nothing downstream — thrombolysis, thrombectomy transfer, anticoagulation hold — can move until the scan is reported, so scan delay propagates into every other time-based indicator.",
       caveat="A high rate here is achievable by scanning everyone immediately, including those who will not benefit. Read it beside door-to-needle rather than on its own."),

    _m(key="ct_720", label="Brain imaging within 12 hours",
       domain="Hyperacute pathway", unit="%", chart="p", higher_is_better=True,
       numerator="flag_ct_720", denominator="den_all_stroke",
       definition="First brain scan within 12 hours of clock start, over all confirmed strokes.",
       why="This is the floor, not the aim. Points below the limit here usually mean out-of-hours radiology capacity, not clinical decision-making.",
       caveat="Near-ceiling indicators produce very tight p-chart limits; check the Laney correction is on before reacting to a signal."),

    _m(key="thrombolysis_rate", label="Thrombolysis rate (ischaemic strokes)",
       domain="Hyperacute pathway", unit="%", chart="p", higher_is_better=True,
       numerator="flag_thrombolysed", denominator="den_ischaemic",
       definition="Ischaemic stroke patients given intravenous thrombolysis, over all ischaemic strokes.",
       why="The most-quoted single number in stroke services and a reasonable proxy for how fast the front door works, because eligibility is dominated by time.",
       caveat="This is a crude rate, not a rate among the eligible. Case mix — age, onset-unknown proportion, referral catchment — drives most of the between-site spread. Use the funnel plot, never a league table."),

    _m(key="thrombectomy_rate", label="Mechanical thrombectomy rate",
       domain="Hyperacute pathway", unit="%", chart="p", higher_is_better=True,
       numerator="flag_thrombectomy", denominator="den_ischaemic",
       definition="Ischaemic stroke patients treated with mechanical thrombectomy, over all ischaemic strokes.",
       why="Access to thrombectomy is the largest single determinant of disability-free survival in large-vessel occlusion, and is dominated by geography and transfer logistics rather than local clinical decisions.",
       caveat="For a non-neurointerventional site this is really a transfer-pathway indicator. Split by 'treated here' vs 'transferred out' before interpreting."),

    _m(key="dtn_60", label="Door-to-needle within 60 minutes",
       domain="Hyperacute pathway", unit="%", chart="p", higher_is_better=True,
       numerator="flag_dtn_60", denominator="den_thrombolysed",
       min_denominator=3,
       definition="Thrombolysed patients treated within 60 minutes of clock start, over all thrombolysed patients.",
       why="Benefit from thrombolysis decays steeply and continuously with time; there is no plateau to relax into. Every 15 minutes saved translates into measurable disability-free days.",
       caveat="Small monthly denominators. With 6–10 thrombolyses a month the p-chart limits are very wide and only large changes will signal — which is correct, not a defect."),

    _m(key="dtn_median", label="Door-to-needle time (median)",
       domain="Hyperacute pathway", unit="minutes", chart="xmr", higher_is_better=False,
       value_col="door_to_needle_min", agg="median",
       definition="Median minutes from clock start to thrombolysis bolus, among thrombolysed patients each period.",
       why="The proportion-within-60 indicator is blind to whether your failures miss by 5 minutes or 50. The median moves when the whole distribution moves, which is what pathway redesign actually does.",
       caveat="Median of a small monthly sample is itself noisy. The XmR limits account for that; a single fast month is not an improvement."),

    _m(key="onset_to_door_median", label="Onset-to-door time (median)",
       domain="Hyperacute pathway", unit="minutes", chart="xmr", higher_is_better=False,
       value_col="onset_to_door_min", agg="median",
       definition="Median minutes from symptom onset to hospital arrival, where onset time is known.",
       why="Mostly outside hospital control — it measures public awareness and ambulance response. Included because it explains treatment-rate variation that would otherwise be blamed on the unit.",
       caveat="Excludes wake-up and unwitnessed strokes entirely, so it describes a selected, faster-presenting group."),

    _m(key="nihss_recorded", label="NIHSS recorded on arrival",
       domain="Hyperacute pathway", unit="%", chart="p", higher_is_better=True,
       numerator="flag_nihss_recorded", denominator="den_all_stroke",
       definition="Patients with a National Institutes of Health Stroke Scale score documented at first assessment.",
       why="Severity is the dominant confounder in every outcome comparison. Without NIHSS no risk adjustment is possible, so a gap here silently caps the quality of everything else in this dashboard.",
       caveat="This is a data-completeness indicator masquerading as a clinical one. Treat it as the entry ticket to the Outcomes domain."),

    # ---------------- Stroke unit & ward care ----------------
    _m(key="su_4h", label="Admitted to a stroke unit within 4 hours",
       domain="Stroke unit & ward care", unit="%", chart="p", higher_is_better=True,
       numerator="flag_su_4h", denominator="den_all_stroke",
       definition="Patients whose first ward was an acute stroke unit, arriving there within 4 hours of clock start.",
       why="Organised stroke unit care is the single largest evidence-based effect in stroke medicine — larger in absolute terms than thrombolysis, because it applies to everyone rather than a treated minority.",
       caveat="Strongly driven by bed availability rather than stroke-team behaviour. A sustained fall here is usually a flow problem originating elsewhere in the hospital."),

    _m(key="su_90pct", label="90% or more of stay on a stroke unit",
       domain="Stroke unit & ward care", unit="%", chart="p", higher_is_better=True,
       numerator="flag_su_90pct_stay", denominator="den_all_stroke",
       definition="Patients who spent at least 90% of their inpatient stay on a stroke unit.",
       why="Guards against the 'touch the stroke unit then outlie' pattern that a 4-hour access indicator alone would reward.",
       caveat="Patients who die early or are discharged within 24 hours distort the ratio; check the exclusion handling before comparing sites."),

    _m(key="time_to_su_median", label="Time to stroke unit (median)",
       domain="Stroke unit & ward care", unit="hours", chart="xmr", higher_is_better=False,
       value_col="time_to_su_hours", agg="median",
       definition="Median hours from clock start to arrival on a stroke unit.",
       why="The continuous partner to the 4-hour indicator; it shows drift long before the proportion crosses a threshold.",
       caveat="Right-skewed. The median is the right summary; do not switch to the mean because it looks worse."),

    _m(key="swallow_4h", label="Swallow screen within 4 hours",
       domain="Stroke unit & ward care", unit="%", chart="p", higher_is_better=True,
       numerator="flag_swallow_4h", denominator="den_all_stroke",
       definition="Patients receiving a formal swallow screen within 4 hours of clock start.",
       why="Dysphagia affects roughly half of acute strokes and unscreened oral intake is the main modifiable driver of aspiration pneumonia — which in turn drives length of stay and mortality.",
       caveat="Screen and assessment are different acts. A service can score well on screening while specialist SLT assessment lags; the SLT indicators below are the check on that."),

    _m(key="mdt_goals_5d", label="MDT goals agreed within 5 days",
       domain="Stroke unit & ward care", unit="%", chart="p", higher_is_better=True,
       numerator="flag_mdt_goals_5d", denominator="den_inpatient_rehab",
       definition="Patients with documented multidisciplinary rehabilitation goals agreed within 5 days of admission.",
       why="Goal setting is the mechanism by which therapy input becomes a plan rather than a series of visits. It is also the best single predictor of whether discharge planning starts early enough.",
       caveat="Highly susceptible to documentation practice. A step change here usually means a form changed, not that care changed."),

    _m(key="vte_ipc_24h", label="IPC started within 24 hours where indicated",
       domain="Stroke unit & ward care", unit="%", chart="p", higher_is_better=True,
       numerator="flag_vte_ipc", denominator="den_immobile",
       definition="Immobile patients started on intermittent pneumatic compression within 24 hours.",
       why="One of very few stroke interventions with a clean randomised effect on a hard outcome in immobile patients, and cheap to deliver reliably.",
       caveat="Denominator depends on how 'immobile' is coded locally; check before cross-site comparison."),

    # ---------------- Therapy & rehabilitation ----------------
    _m(key="pt_72h", label="Physiotherapy assessment within 72 hours",
       domain="Therapy & rehabilitation", unit="%", chart="p", higher_is_better=True,
       numerator="flag_pt_72h", denominator="den_pt_needed",
       definition="Patients assessed by physiotherapy within 72 hours of clock start, among those recorded as requiring physiotherapy.",
       why="Early mobilisation and positioning decisions are made in the first days; a late first contact cannot be made up later.",
       caveat="The denominator is a clinical judgement ('needs PT'), so a service can improve this indicator by narrowing who it says needs input. Watch the denominator trend alongside."),

    _m(key="ot_72h", label="Occupational therapy assessment within 72 hours",
       domain="Therapy & rehabilitation", unit="%", chart="p", higher_is_better=True,
       numerator="flag_ot_72h", denominator="den_ot_needed",
       definition="Patients assessed by occupational therapy within 72 hours of clock start, among those recorded as requiring OT.",
       why="Early OT assessment drives cognitive and perceptual screening, seating and the first realistic view of the discharge environment — all of which set the length of stay.",
       caveat="As with PT, a self-defined denominator. Track alongside the proportion of admissions deemed to need OT."),

    _m(key="slt_72h", label="Speech & language therapy assessment within 72 hours",
       domain="Therapy & rehabilitation", unit="%", chart="p", higher_is_better=True,
       numerator="flag_slt_72h", denominator="den_slt_needed",
       definition="Patients assessed by SLT within 72 hours of clock start, among those recorded as requiring SLT.",
       why="Covers both dysphagia management beyond the initial screen and communication assessment. Delay leaves patients either nil-by-mouth longer than necessary or eating unsafely, and leaves aphasic patients unable to participate in their own care decisions.",
       caveat="Combines two distinct caseloads with very different urgency. Split dysphagia from communication before acting on a signal."),

    _m(key="pt_min_per_day", label="Physiotherapy minutes per day (median)",
       domain="Therapy & rehabilitation", unit="minutes/day", chart="xmr", higher_is_better=True,
       value_col="pt_min_per_day", agg="median",
       definition="Median minutes of physiotherapy per patient per day on which therapy was applicable.",
       why="Rehabilitation is dose-dependent. Assessment-within-72-hours tells you contact was made; only minutes-per-day tells you whether a therapeutic dose followed.",
       caveat="Minutes are an input, not an outcome, and are easily inflated by counting non-therapeutic contact. Read with the percentage-of-days indicator: 90 minutes on two days a week is not the same dose as 45 minutes on five."),

    _m(key="ot_min_per_day", label="Occupational therapy minutes per day (median)",
       domain="Therapy & rehabilitation", unit="minutes/day", chart="xmr", higher_is_better=True,
       value_col="ot_min_per_day", agg="median",
       definition="Median minutes of OT per patient per applicable day.",
       why="Same logic as physiotherapy: dose, not contact.",
       caveat="OT sessions are often longer and less frequent than PT; the same target applied to both disciplines can mislead."),

    _m(key="slt_min_per_day", label="SLT minutes per day (median)",
       domain="Therapy & rehabilitation", unit="minutes/day", chart="xmr", higher_is_better=True,
       value_col="slt_min_per_day", agg="median",
       definition="Median minutes of SLT per patient per applicable day.",
       why="SLT dose is the most consistently under-delivered of the three disciplines in national audit, largely because establishment is set against dysphagia demand rather than aphasia demand.",
       caveat="Dysphagia review is short and frequent; aphasia therapy is long and less frequent. An unsplit median is an average of two different services."),

    _m(key="pt_pct_days", label="Days on which physiotherapy was received",
       domain="Therapy & rehabilitation", unit="%", chart="p", higher_is_better=True,
       numerator="pt_days_with_therapy", denominator="pt_days_applicable",
       definition="Patient-days with at least one attended PT session, over all patient-days on which PT was applicable.",
       why="Frequency is what separates a rehabilitation service from an assessment service, and it is where seven-day working shows up.",
       caveat="Very large denominators — thousands of patient-days a month — so the ordinary binomial p-chart limits will be far too tight. This indicator is the reason the Laney p' correction exists in this app."),

    _m(key="ot_pct_days", label="Days on which occupational therapy was received",
       domain="Therapy & rehabilitation", unit="%", chart="p", higher_is_better=True,
       numerator="ot_days_with_therapy", denominator="ot_days_applicable",
       definition="Patient-days with at least one attended OT session, over applicable patient-days.",
       why="OT reliability decides whether the discharge plan is built from a real assessment of function or from an assumption. Days lost here do not just reduce dose; they push the home visit, the equipment order and the family conversation later, and those sit on the critical path out of hospital.",
       caveat="OT input is often front-loaded (assessment, seating, cognition) and then tapers by design. A falling rate late in a long stay can be correct clinical practice rather than a gap — check it against day of stay before acting."),

    _m(key="slt_pct_days", label="Days on which SLT was received",
       domain="Therapy & rehabilitation", unit="%", chart="p", higher_is_better=True,
       numerator="slt_days_with_therapy", denominator="slt_days_applicable",
       definition="Patient-days with at least one attended SLT session, over applicable patient-days.",
       why="This is the indicator where the two SLT caseloads pull hardest against each other. Dysphagia review is urgent and short; aphasia therapy is neither. When establishment is set against swallowing demand, communication work is what silently absorbs the shortfall — and it does so without ever showing up as a missed safety standard.",
       caveat="An unsplit rate averages two services with different urgency and different session lengths. Split dysphagia from communication before drawing any conclusion, or the number describes neither."),

    _m(key="pt_45min", label="Days meeting the 45-minute PT standard",
       domain="Therapy & rehabilitation", unit="%", chart="p", higher_is_better=True,
       numerator="pt_days_45min", denominator="pt_days_applicable",
       definition="Patient-days with 45 or more attended PT minutes, over applicable patient-days.",
       why="The 45-minute guideline figure is a dose threshold, not an average. Measuring attainment day by day is the only way to see whether the standard is met or merely averaged into.",
       caveat="A day meeting 45 minutes through three brief contacts is not clinically equivalent to one sustained session. Session-level data is available on the Therapy page."),

    _m(key="ot_45min", label="Days meeting the 45-minute OT standard",
       domain="Therapy & rehabilitation", unit="%", chart="p", higher_is_better=True,
       numerator="ot_days_45min", denominator="ot_days_applicable",
       definition="Patient-days with 45 or more attended OT minutes, over applicable patient-days.",
       why="OT sessions tend to be longer and less frequent than physiotherapy, so a service can look strong on minutes-per-day and still miss the daily threshold often. Attainment measured day by day is what separates the two patterns.",
       caveat="Applying an identical 45-minute threshold to all three disciplines is a simplification the guideline invites but the clinical reality does not always support. Agree locally whether it is the right bar for OT before performance-managing against it."),

    _m(key="slt_45min", label="Days meeting the 45-minute SLT standard",
       domain="Therapy & rehabilitation", unit="%", chart="p", higher_is_better=True,
       numerator="slt_days_45min", denominator="slt_days_applicable",
       definition="Patient-days with 45 or more attended SLT minutes, over applicable patient-days.",
       why="The hardest of the three to achieve and the most revealing. Reaching 45 minutes of SLT in a day generally means aphasia therapy actually happened, because dysphagia review alone rarely fills it. Attainment here is a reasonable proxy for whether communication work is being delivered at all.",
       caveat="Fatigue and attention limits mean a shorter, well-timed session can be worth more than a long one for an aphasic patient. Read this beside outcome measures rather than as a standard to be hit at any cost."),

    _m(key="weekend_therapy", label="Weekend therapy delivery",
       domain="Therapy & rehabilitation", unit="%", chart="p", higher_is_better=True,
       numerator="weekend_days_with_therapy", denominator="weekend_days_applicable",
       definition="Weekend patient-days with at least one attended therapy session of any discipline, over applicable weekend patient-days.",
       why="Weekend provision is the clearest structural determinant of total rehabilitation dose. A service delivering excellent weekday therapy still loses two sevenths of every patient's stay.",
       caveat="Weekend cover is often deliberately targeted at a subset (new admissions, dysphagia review). A low rate may be a correct prioritisation decision rather than a gap."),

    # ---------------- Safety ----------------
    _m(key="falls_rate", label="Inpatient falls", domain="Safety",
       unit="per 1,000 bed days", chart="u", higher_is_better=False,
       count_col="falls", exposure_col="bed_days",
       definition="All reported inpatient falls per 1,000 stroke bed days.",
       why="Falls are the most common adverse event in stroke rehabilitation and sit directly on the tension between mobilising patients and keeping them safe. A falls rate of zero on a rehabilitation ward means nobody is being mobilised.",
       caveat="Bed days, not admissions, is the correct denominator — otherwise a longer average stay reads as a worse safety record. Falls also cluster within patients, which is why the u' overdispersion correction usually engages here."),

    _m(key="hap_rate", label="Hospital-acquired pneumonia", domain="Safety",
       unit="%", chart="p", higher_is_better=False,
       numerator="flag_hap", denominator="den_all_stroke",
       definition="Patients developing pneumonia after 48 hours from admission, over all stroke admissions.",
       why="The main avoidable driver of stroke mortality and prolonged stay, and the outcome most sensitive to swallow screening and oral care reliability.",
       caveat="Ascertainment varies enormously with local coding practice. Sudden improvement is more often a coding change than a clinical one."),

    _m(key="pressure_ulcer_rate", label="New pressure ulcers", domain="Safety",
       unit="per 1,000 bed days", chart="u", higher_is_better=False,
       count_col="pressure_ulcers", exposure_col="bed_days",
       definition="Newly acquired pressure ulcers (category 2 and above) per 1,000 stroke bed days.",
       why="A direct readout of repositioning, seating and continence care in a population with impaired sensation and mobility.",
       caveat="Rare enough that monthly counts are mostly zero. The rare-event t-chart on the Control Charts page is the more informative view."),

    # ---------------- Secondary prevention & discharge ----------------
    _m(key="af_anticoag", label="Anticoagulation for AF on discharge",
       domain="Secondary prevention & discharge", unit="%", chart="p", higher_is_better=True,
       numerator="flag_af_anticoag", denominator="den_af_survivors",
       definition="Patients with atrial fibrillation discharged on an anticoagulant, over surviving AF patients.",
       why="The largest absolute risk reduction available in secondary prevention. An untreated AF patient carries several times the annual recurrence risk of a treated one.",
       caveat="Legitimate contraindications exist and are rarely coded well, so 100% is not the right aim. Investigate the reasons behind the gap rather than the gap size."),

    _m(key="antiplatelet", label="Antiplatelet on discharge (ischaemic)",
       domain="Secondary prevention & discharge", unit="%", chart="p", higher_is_better=True,
       numerator="flag_antiplatelet", denominator="den_ischaemic_survivors",
       definition="Surviving ischaemic stroke patients discharged on an antiplatelet (or anticoagulant where indicated instead).",
       why="Near-universal indication and near-zero cost; a reliable process indicator for discharge medicines reconciliation as a whole.",
       caveat="Ceiling effect. Use as an assurance indicator, not an improvement target."),

    _m(key="esd_referral", label="Early supported discharge referral",
       domain="Secondary prevention & discharge", unit="%", chart="p", higher_is_better=True,
       numerator="flag_esd", denominator="den_esd_eligible",
       definition="Eligible patients referred to an early supported discharge team, over those meeting local eligibility.",
       why="ESD delivers rehabilitation at the same intensity in the patient's own environment and shortens stay without harming outcome. It is the main lever on inpatient bed pressure that does not involve reducing care.",
       caveat="Eligibility criteria differ between services, so the denominator is not comparable across sites without agreeing definitions first."),

    _m(key="los_median", label="Length of stay (median)",
       domain="Secondary prevention & discharge", unit="days", chart="xmr", higher_is_better=False,
       value_col="los_days", agg="median",
       definition="Median inpatient length of stay in days for discharged stroke patients.",
       why="The summary consequence of everything upstream — flow, therapy dose, complications and discharge planning all land here.",
       caveat="Not a quality indicator on its own, and dangerous as a target. Falling LOS with rising readmissions or falling home-discharge rates is deterioration, not improvement. Always read the three together."),

    _m(key="discharge_home", label="Discharged to usual place of residence",
       domain="Secondary prevention & discharge", unit="%", chart="p", higher_is_better=True,
       numerator="flag_home", denominator="den_survivors_prehome",
       definition="Surviving patients discharged to their pre-admission residence, over survivors admitted from home.",
       why="The outcome patients themselves rank highest, and a far better companion to length of stay than any process measure.",
       caveat="Heavily case-mix dependent (age, pre-stroke mRS, living alone). Compare via the funnel plot with those factors in mind."),

    _m(key="mood_screen", label="Mood screened before discharge",
       domain="Secondary prevention & discharge", unit="%", chart="p", higher_is_better=True,
       numerator="flag_mood_screen", denominator="den_inpatient_rehab",
       definition="Patients with a documented mood screen before discharge.",
       why="Post-stroke depression affects roughly a third of survivors, suppresses engagement with rehabilitation, and is the most commonly missed treatable problem in the pathway.",
       caveat="Screening without an onward pathway achieves nothing. Track referral-on rates beside this."),

    # ---------------- Outcomes ----------------
    _m(key="inpatient_mortality", label="Inpatient mortality",
       domain="Outcomes", unit="%", chart="p", higher_is_better=False,
       numerator="flag_died", denominator="den_all_stroke",
       definition="Deaths before discharge, over all stroke admissions.",
       why="The hardest outcome available and the one least affected by documentation practice.",
       caveat="Crude mortality is dominated by case mix — age, stroke type, NIHSS, pre-stroke function. Comparing crude rates between sites is close to meaningless; the funnel plot with overdispersion is the minimum defensible display, and formal risk adjustment is the right answer."),

    _m(key="mrs_0_2", label="Independent at discharge (mRS 0–2)",
       domain="Outcomes", unit="%", chart="p", higher_is_better=True,
       numerator="flag_mrs_0_2", denominator="den_all_stroke",
       definition="Patients with a modified Rankin Scale score of 0–2 at discharge, over all admissions.",
       why="The standard functional endpoint of stroke trials, and the closest routine measure to 'the thing rehabilitation is for'.",
       caveat="Discharge mRS is measured too early to reflect rehabilitation potential and is inflated by early discharge of mild strokes. Where 90-day mRS exists, prefer it, and always look at the full mRS distribution rather than the dichotomy."),

    _m(key="readmission_30d", label="Emergency readmission within 30 days",
       domain="Outcomes", unit="%", chart="p", higher_is_better=False,
       numerator="flag_readmit_30d", denominator="den_survivors",
       definition="Surviving patients with an emergency readmission within 30 days of discharge.",
       why="The check on length-of-stay reduction. A service that discharges faster and readmits more has moved cost, not improved care.",
       caveat="Captures readmissions to the same organisation only unless linked data is supplied; expect under-count."),
]}


# ---------------------------------------------------------------------------
# Thresholds -- the standard-dependent layer
# ---------------------------------------------------------------------------
_SSNAP_NOTE = "Indicative working default aligned to SSNAP-style reporting. Confirm against the current audit year's technical guidance before external use."
_INAS_NOTE = "Indicative working default aligned to Irish national stroke reporting. Confirm against the current National Stroke Audit specification before external use."

THRESHOLDS: dict[Standard, dict[str, Threshold]] = {
    "SSNAP": {
        "ct_60": Threshold(50.0, 60.0, provenance=_SSNAP_NOTE),
        "ct_720": Threshold(95.0, 98.0, provenance=_SSNAP_NOTE),
        "thrombolysis_rate": Threshold(11.0, 14.0, provenance=_SSNAP_NOTE),
        "thrombectomy_rate": Threshold(5.0, 10.0, provenance=_SSNAP_NOTE),
        "dtn_60": Threshold(65.0, 80.0, provenance=_SSNAP_NOTE),
        "dtn_median": Threshold(45.0, 30.0, provenance=_SSNAP_NOTE),
        "onset_to_door_median": Threshold(None),
        "nihss_recorded": Threshold(95.0, 99.0, provenance=_SSNAP_NOTE),
        "su_4h": Threshold(60.0, 75.0, provenance=_SSNAP_NOTE),
        "su_90pct": Threshold(80.0, 90.0, provenance=_SSNAP_NOTE),
        "time_to_su_median": Threshold(4.0, 2.0, provenance=_SSNAP_NOTE),
        "swallow_4h": Threshold(75.0, 90.0, provenance=_SSNAP_NOTE),
        "mdt_goals_5d": Threshold(85.0, 95.0, provenance=_SSNAP_NOTE),
        "vte_ipc_24h": Threshold(90.0, 95.0, provenance=_SSNAP_NOTE),
        "pt_72h": Threshold(85.0, 95.0, provenance=_SSNAP_NOTE),
        "ot_72h": Threshold(85.0, 95.0, provenance=_SSNAP_NOTE),
        "slt_72h": Threshold(75.0, 90.0, provenance=_SSNAP_NOTE),
        "pt_min_per_day": Threshold(45.0, 60.0, provenance="NICE/RCP guideline dose of 45 minutes per relevant therapy per day."),
        "ot_min_per_day": Threshold(45.0, 60.0, provenance="NICE/RCP guideline dose of 45 minutes per relevant therapy per day."),
        "slt_min_per_day": Threshold(45.0, 60.0, provenance="NICE/RCP guideline dose of 45 minutes per relevant therapy per day."),
        "pt_pct_days": Threshold(80.0, 90.0, provenance=_SSNAP_NOTE),
        "ot_pct_days": Threshold(80.0, 90.0, provenance=_SSNAP_NOTE),
        "slt_pct_days": Threshold(70.0, 85.0, provenance=_SSNAP_NOTE),
        "pt_45min": Threshold(60.0, 80.0, provenance=_SSNAP_NOTE),
        "ot_45min": Threshold(60.0, 80.0, provenance=_SSNAP_NOTE),
        "slt_45min": Threshold(50.0, 70.0, provenance=_SSNAP_NOTE),
        "weekend_therapy": Threshold(60.0, 80.0, provenance="Local working default; no national threshold is universally agreed."),
        "falls_rate": Threshold(5.0, 3.0, provenance="Local working default; falls rates are compared, not targeted, in national audit."),
        "hap_rate": Threshold(8.0, 5.0, provenance="Local working default."),
        "pressure_ulcer_rate": Threshold(0.5, 0.2, provenance="Local working default."),
        "af_anticoag": Threshold(90.0, 95.0, provenance=_SSNAP_NOTE),
        "antiplatelet": Threshold(95.0, 99.0, provenance=_SSNAP_NOTE),
        "esd_referral": Threshold(75.0, 90.0, provenance=_SSNAP_NOTE),
        "los_median": Threshold(12.0, 9.0, provenance="Local working default; length of stay is not a national quality target."),
        "discharge_home": Threshold(65.0, 75.0, provenance="Local working default."),
        "mood_screen": Threshold(85.0, 95.0, provenance=_SSNAP_NOTE),
        "inpatient_mortality": Threshold(12.0, 9.0, provenance="Local working default; mortality should be risk-adjusted, not targeted."),
        "mrs_0_2": Threshold(50.0, 60.0, provenance="Local working default."),
        "readmission_30d": Threshold(10.0, 7.0, provenance="Local working default."),
    },
    "INAS": {
        # Irish national reporting emphasises access and treatment rates,
        # applies ceiling-style targets to screening, and does not score
        # therapy dose in the SSNAP domain style.
        "ct_60": Threshold(60.0, 80.0, label_override="Brain imaging within 1 hour", provenance=_INAS_NOTE),
        "ct_720": Threshold(98.0, 100.0, provenance=_INAS_NOTE),
        "thrombolysis_rate": Threshold(12.0, 15.0, provenance=_INAS_NOTE),
        "thrombectomy_rate": Threshold(8.0, 12.0, provenance=_INAS_NOTE),
        "dtn_60": Threshold(70.0, 85.0, provenance=_INAS_NOTE),
        "dtn_median": Threshold(40.0, 30.0, provenance=_INAS_NOTE),
        "onset_to_door_median": Threshold(None),
        "nihss_recorded": Threshold(95.0, 100.0, provenance=_INAS_NOTE),
        "su_4h": Threshold(70.0, 85.0, label_override="Admitted to a stroke unit within 4 hours", provenance=_INAS_NOTE),
        "su_90pct": Threshold(90.0, 95.0, provenance=_INAS_NOTE),
        "time_to_su_median": Threshold(4.0, 2.0, provenance=_INAS_NOTE),
        "swallow_4h": Threshold(90.0, 100.0, provenance=_INAS_NOTE),
        "mdt_goals_5d": Threshold(80.0, 90.0, provenance=_INAS_NOTE),
        "vte_ipc_24h": Threshold(90.0, 95.0, provenance=_INAS_NOTE),
        "pt_72h": Threshold(90.0, 95.0, label_override="Physiotherapy access within 72 hours", provenance=_INAS_NOTE),
        "ot_72h": Threshold(90.0, 95.0, label_override="Occupational therapy access within 72 hours", provenance=_INAS_NOTE),
        "slt_72h": Threshold(85.0, 95.0, label_override="SLT access within 72 hours", provenance=_INAS_NOTE),
        "pt_min_per_day": Threshold(45.0, 60.0, provenance="ESO/RCP guideline dose; not separately scored in Irish national audit."),
        "ot_min_per_day": Threshold(45.0, 60.0, provenance="ESO/RCP guideline dose; not separately scored in Irish national audit."),
        "slt_min_per_day": Threshold(45.0, 60.0, provenance="ESO/RCP guideline dose; not separately scored in Irish national audit."),
        "pt_pct_days": Threshold(75.0, 90.0, provenance="Local working default."),
        "ot_pct_days": Threshold(75.0, 90.0, provenance="Local working default."),
        "slt_pct_days": Threshold(65.0, 85.0, provenance="Local working default."),
        "pt_45min": Threshold(55.0, 75.0, provenance="Local working default."),
        "ot_45min": Threshold(55.0, 75.0, provenance="Local working default."),
        "slt_45min": Threshold(45.0, 70.0, provenance="Local working default."),
        "weekend_therapy": Threshold(50.0, 75.0, provenance="Local working default."),
        "falls_rate": Threshold(5.0, 3.0, provenance="Local working default."),
        "hap_rate": Threshold(8.0, 5.0, provenance="Local working default."),
        "pressure_ulcer_rate": Threshold(0.5, 0.2, provenance="Local working default."),
        "af_anticoag": Threshold(90.0, 95.0, provenance=_INAS_NOTE),
        "antiplatelet": Threshold(95.0, 99.0, provenance=_INAS_NOTE),
        "esd_referral": Threshold(70.0, 85.0, label_override="Early supported discharge referral", provenance=_INAS_NOTE),
        "los_median": Threshold(11.0, 8.0, provenance="Local working default."),
        "discharge_home": Threshold(65.0, 75.0, provenance="Local working default."),
        "mood_screen": Threshold(80.0, 90.0, provenance=_INAS_NOTE),
        "inpatient_mortality": Threshold(12.0, 9.0, provenance="Local working default; report risk-adjusted."),
        "mrs_0_2": Threshold(50.0, 60.0, provenance="Local working default."),
        "readmission_30d": Threshold(10.0, 7.0, provenance="Local working default."),
    },
}


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------
def metric(key: str, standard: Standard = "SSNAP",
           overrides: dict[str, float] | None = None) -> tuple[MetricSpec, Threshold]:
    """Return the spec and the threshold applying under ``standard``.

    ``overrides`` lets the sidebar re-target a metric at runtime without
    touching this file -- which is how a local service adapts national
    thresholds to a locally agreed trajectory.
    """
    spec = METRICS[key]
    thr = THRESHOLDS[standard].get(key, Threshold(None))
    if overrides and key in overrides:
        thr = replace(thr, target=overrides[key],
                      provenance="Locally overridden in this session.")
    if thr.label_override:
        spec = replace(spec, label=thr.label_override)
    return spec, thr


def metrics_in_domain(domain: str, standard: Standard = "SSNAP") -> list[str]:
    return [k for k, m in METRICS.items()
            if m.domain == domain and k in THRESHOLDS[standard]]


def all_metric_keys(standard: Standard = "SSNAP") -> list[str]:
    return [k for k in METRICS if k in THRESHOLDS[standard]]


# The subset shown on the front-page scorecard: one or two per domain,
# chosen because they are the ones a clinical director is asked about.
HEADLINE_METRICS = [
    "ct_60", "dtn_60", "thrombolysis_rate", "su_4h", "swallow_4h",
    "pt_72h", "ot_72h", "slt_72h", "pt_pct_days", "af_anticoag",
    "esd_referral", "los_median", "hap_rate", "inpatient_mortality",
]
