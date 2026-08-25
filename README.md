# Stroke Quality Intelligence

[![tests](https://github.com/USERNAME/stroke-quality-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/stroke-quality-dashboard/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b)](https://streamlit.io)

A Streamlit dashboard for stroke service quality improvement: audit indicators on a
switchable SSNAP / Irish national standard, a full statistical process control toolkit
with NHS *Making Data Count* variation and assurance icons, and a therapy layer built
for the PT, OT and SLT services rather than bolted onto a medical dashboard.



> All data shipped with this app is **simulated**. It is realistic in structure and
> must never be quoted as evidence about any real service.

![Board scorecard](docs/screenshots/01-service-overview.png)

---

## What it does differently

**1. Separates the indicator from the standard.** How a metric is calculated and what
target it is judged against are different things, owned by different people, changing
on different clocks. Switching between SSNAP-style and Irish national reporting
re-labels and re-targets 39 indicators without touching a line of calculation.

**2. Treats variation properly.** XmR, p and Laney p′, u and u′, c, t and g charts,
funnel plots with overdispersion adjustment, CUSUM, baseline freezing and phase
breaks: with *Making Data Count* variation and assurance icons on every indicator.
No RAG ratings, no month-on-month arrows.

**3. Builds the therapy layer for therapists.** PT, OT and SLT get dose and frequency
measured against a real denominator of opportunity, missed-session reasons split into
capacity and patient causes, a demand-versus-capacity model with its assumptions on
the page rather than buried, and caseload measured per *available* WTE.

---

## Quick start

```bash
git clone https://github.com/USERNAME/stroke-quality-dashboard.git
cd stroke-quality-dashboard
pip install -r requirements.txt
streamlit run app.py
```

First launch generates a simulated three-year, four-site cohort (~6,000 admissions,
~180,000 therapy patient-days) and caches it to `data/` as Parquet. Subsequent
launches are instant.

```bash
pytest tests/ -q      # 154 tests
```

---

## The pages

### Control charts 

Any indicator, any chart type, with baseline freezing, phase breaks, a change-point
suggester, CUSUM, funnel plots, rare-event t- and g-charts, and the rule sets laid out
side by side.

![Control charts](docs/screenshots/02-control-charts.png)

### Therapy dose 

The dose cascade (applicable days → days delivered → days meeting 45 minutes), the
weekend effect quantified in minutes, time to first assessment, and a dose–outcome
view with the confounding made explicit rather than hidden.

![Therapy dose](docs/screenshots/03-therapy-dose.png)

### Caseload and capacity 

Caseload per *available* WTE, demand against capacity in minutes, missed-session
Pareto split by cause, waiting time to first delivered session, and vacancy against
establishment. The screenshot below shows a planted physiotherapy vacancy: caseload
per WTE jumps from about 5 to over 30 while the establishment figure barely moves.

![Caseload and capacity](docs/screenshots/04-caseload-capacity.png)

### Hyperacute pathway

Where the time goes from clock start, the out-of-hours tail with a Hodges–Lehmann
shift estimate, and the full door-to-needle distribution rather than just the
60-minute threshold.

![Hyperacute pathway](docs/screenshots/05-hyperacute-pathway.png)

### Data quality

Validation report, completeness measured *within each field's applicable cohort*,
denominator drift, and a digit-preference check for rounded timings.

![Data quality](docs/screenshots/06-data-quality.png)

### Also

**Service overview**
**Patient explorer**

---

## Deploying it

The app is a standard Streamlit project and deploys as-is on
[Streamlit Community Cloud](https://share.streamlit.io), free:

1. Push this repo to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. **New app** → pick this repo, branch `main`, main file `app.py`.
4. Deploy. It redeploys automatically on every push to `main`.

**If you point it at real data, do not deploy it publicly.** Community Cloud apps have
no authentication. Run it behind your organisation's infrastructure instead; a
`Dockerfile` is a small addition if you need one.

---

## Using your own data

Two files, or two SQL views. Download the template and data dictionary from
**Data quality → Load your own data**, or see `core/loaders.py` for the full schema.

### `admissions` 

Key columns: `admission_id`, `site`, `arrival_datetime` (clock start), `age`,
`stroke_type`, `nihss`, `prestroke_mrs`, `door_to_ct_min`, `thrombolysed`,
`door_to_needle_min`, `time_to_su_hours`, `swallow_screen_hours`,
`pt_needed` / `ot_needed` / `slt_needed`,
`pt_assess_hours` / `ot_assess_hours` / `slt_assess_hours`, `los_days`,
`died_inpatient`, `discharge_destination`, `mrs_discharge`.

### `sessions` 

Not one row per delivered session. This is the single most important modelling choice
in the app.


### `staffing`

`wte_funded`, `wte_vacant`, `wte_absent`, `wte_available`. Without it everything works
except the caseload and capacity page.

### A note on patient data

## Targets

The thresholds in `core/standards.py` are **sensible working defaults, not a
transcription of any audit's current technical guidance**. National thresholds move.
Before anything from this dashboard reaches a board paper, open the current year's
technical guidance and confirm every value in the `THRESHOLDS` tables. They are
deliberately kept in one place, and are overridable at runtime, so this is a
five-minute job rather than a code change.

Each target carries a `provenance` string, shown in the interface, so a reader can
tell a national threshold from a local working assumption.

---

## Project layout

```
app.py                  router
core/
    standards.py        indicator registry + the switchable target layer
    metrics.py          patient records -> numerators and denominators
    spc.py              the control chart engine
    mdc.py              Making Data Count classification and icons
    synth.py            synthetic cohort with planted special causes
    loaders.py          schema contract, validation, caching
    viz.py              Plotly builders and the palette
    ui.py               shared Streamlit furniture
views/                  one module per page
tests/
    test_spc.py         SPC arithmetic + planted-signal ground truth
    test_viz.py         every figure serialises, including degenerate input
    test_pages.py       every page and every control, via Streamlit's AppTest
```

### Extending it

**Adding an indicator.** Add a boolean `flag_*` (and a `den_*` cohort if it needs a
new one) in `core/metrics.derive_flags`, then a `MetricSpec` and its thresholds in
`core/standards.py`. Nothing else changes: it appears on the scorecard, in the
control chart workbench and in the funnel picker automatically.

**Adding a standard.** Add a key to `THRESHOLDS`. The sidebar picks it up.

**Risk adjustment.** The most valuable extension, and deliberately not included
because it is a modelling commitment rather than a feature. Crude mortality and
functional-outcome comparisons between sites are close to meaningless without it.
`METHODS.md` §10 sets out how to do it and what to watch for.

---

## Tested against

| Python | streamlit | numpy | pandas | scipy |
|---|---|---|---|---|
| 3.11 | 1.36.0 | 1.26.4 | 2.2.3 | 1.13.1 |
| 3.12 | 1.36.0 | latest | latest | latest |
| 3.12 | latest | latest | 2.3.x | latest |
| 3.12 / 3.13 | latest | latest | 3.x | latest |



---


