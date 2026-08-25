# Methods

The statistical reasoning behind the dashboard, in the order you meet it.

---

## 1. Why not just plot the number and compare it to last month

A stable process produces a different number every period. Compare two
consecutive draws from it and you will find a "change" about half the
time, in a direction determined by nothing. Month-on-month arrows in
performance reports are therefore not weak evidence — they are a
random number generator with a narrative attached.

RAG rating against a target is a different failure. It tells you about
one point, when what you need to know is about the process that produced
it. A service whose true rate is 78% against a 75% target will show red
in roughly a third of months. Managing those red months is managing
noise, and it reliably makes things worse: Deming's funnel experiment is
the demonstration that adjusting a stable process in response to its own
variation increases that variation.

Control charts answer the only two questions worth asking:

- **Has the process changed?** (variation)
- **Can it be relied on to meet the target?** (assurance)

They are orthogonal, and keeping them apart is the whole point. A
process can be perfectly stable and reliably fail — which is a redesign
problem that no amount of performance management will touch.

---

## 2. Estimating sigma: why the moving range

The naive control chart plots mean ± 3 × SD(all points). It is wrong,
and wrong in a specific, dangerous direction.

SD(all points) is computed around the grand mean, so any special cause
in the series — a step, a trend, an outlier — inflates the very estimate
you are using to detect it. The limits widen and the signal hides inside
them.

Shewhart's estimator uses only *short-term* variation, measured between
adjacent points, which a shift cannot contaminate:

$$\hat\sigma = \frac{\overline{mR}}{d_2}, \qquad
  \overline{mR} = \frac{1}{n-1}\sum_{i=2}^{n}|x_i - x_{i-1}|, \qquad
  d_2 = 1.128$$

so the limits are $\bar x \pm 2.660\,\overline{mR}$.

The 1.128 is $E[\text{range of 2 draws from } N(0,1)]$. The **3** is not
a p-value: Shewhart chose it as an economic balance between chasing
noise and missing real change. Under normality it corresponds to roughly
one false alarm in 370 points, but the limits are robust well beyond the
normal case — which is why they survive the right-skewed distributions
that dominate clinical timing data.

### How robust, exactly

This is usually asserted and rarely measured. `tests/test_spc.py`
measures it, and the answer is stronger than the folklore:

| Change in the series | Effect on $\hat\sigma$ |
|---|---|
| Step change, 3σ | none |
| Step change, 25σ | none |
| Gradual drift, 0.25σ per period | +2% |
| Gradual drift, 0.5σ per period | +4% |
| Gradual drift, 1.0σ per period | +13% |
| Gradual drift, 2.0σ per period | +45% |

A step change contaminates exactly one moving range out of $n-1$, and
the screening rule (drop moving ranges above $3.27\,\overline{mR}$ and
recompute once) discards it. So sigma is unaffected by a step of *any*
magnitude. Drift does inflate it, but the effect is second order in
(drift per period / noise) and stays negligible until the two are
comparable.

**This changes the usual argument for baseline freezing.** The folk
justification — "otherwise the improvement absorbs itself into the
limits and the chart shows nothing" — is mostly false. What whole-series
limits actually lose is the *reference level*: the centre line lands
between the old and new levels, so the chart can only say "half the
points are above the mean and half below", and the run rule flags the
entire series. Freeze a baseline to keep the comparison, not to keep the
limits tight.

---

## 3. Attribute charts and the overdispersion problem

Proportions and rates do not estimate sigma from the data. The
distributional model supplies it:

$$\text{p-chart:}\quad \sigma_i = \sqrt{\frac{\bar p(1-\bar p)}{n_i}}
\qquad
\text{u-chart:}\quad \sigma_i = \sqrt{\frac{\bar u}{n_i}}$$

Elegant when the model holds. Disastrous when it does not — and in
clinical audit it usually does not.

The binomial model assumes every case in a period is an independent
Bernoulli trial with the *same* probability. In reality case mix varies
between periods, patients cluster by consultant and by day of week, and
the true probability drifts. The result is **overdispersion**: observed
scatter exceeds what binomial sampling can produce. With large
denominators the limits collapse toward the centre line and practically
every point signals — which is not a detection triumph, it is a rejected
model.

The percentage-of-days-with-therapy indicator is the extreme case. Its
denominator is patient-*days* — thousands per month — so binomial limits
sit within a fraction of a percentage point of the centre line.

### The Laney correction

Laney (2002) fixes this without abandoning the chart. Standardise:

$$z_i = \frac{p_i - \bar p}{\sqrt{\bar p(1-\bar p)/n_i}}$$

If the binomial model were right, the $z_i$ would behave like standard
normal draws, so their short-term variation would be exactly 1. Estimate
it the Shewhart way, from the moving range of $z$:

$$\sigma_z = \frac{\overline{mR_z}}{1.128}$$

and widen the limits by that factor:

$$\text{limits}_i = \bar p \pm 3\,\sigma_z\sqrt{\frac{\bar p(1-\bar p)}{n_i}}$$

When $\sigma_z = 1$ this is *exactly* the ordinary p-chart, so p′ is a
strict generalisation rather than a different chart. (Pinned by a test.)

**$\sigma_z$ is itself a finding.** A value substantially above 1 says
the periods are not exchangeable, and the useful question becomes what
differs between them — case mix, coding, or a real difference in the
process. It is reported on every chart's methods panel.

$\sigma_z$ below 1 (underdispersion) is never used to *narrow* limits.
Narrowing manufactures false signals, and underdispersion in audit data
usually means rounding or smoothing upstream.

### Pooled, not averaged

The centre line is $\bar p = \sum d_i / \sum n_i$, weighting each period
by its own denominator. Averaging the period proportions instead would
give a month with 3 cases the same influence as a month with 300.

---

## 4. The rule sets

| Rule | Detects | NHS *Making Data Count* | Nelson |
|---|---|---|---|
| 1 | Point beyond 3σ | ✓ | ✓ |
| 2 | Run on one side of the centre line | ✓ (7) | ✓ (9) |
| 3 | Consecutive points rising or falling | ✓ (7) | ✓ (6) |
| 4 | 2 of 3 beyond 2σ, same side | ✓ | ✓ |
| 5 | 4 of 5 beyond 1σ, same side | — | ✓ |
| 6 | 15 consecutive within 1σ | — | ✓ |
| 7 | 8 consecutive with none within 1σ | — | ✓ |
| 8 | 14 consecutive alternating | — | ✓ |

**The sets are not nested.** Nelson has more rules but a longer run
length for a shift (9 vs 7), so on a series whose only signal is a
modest shift the NHS set can find *more*. Only rule 1 is common to all.

Each added rule raises sensitivity and raises the false-alarm rate: with
all eight Nelson rules active the in-control false-alarm rate is roughly
one point in 90, several times that of rule 1 alone. **Choose a set once
and stay with it.** Switching after seeing the data is how SPC loses
credibility with a clinical audience.

### Rules 6 and 7 are data-quality alarms

Rule 6 fires when points hug the centre line — a process that looks
*too* well behaved. In audit data this almost never means excellence; it
means stratification, a wrongly-defined subgroup, heavy rounding, or a
figure smoothed upstream. Rule 7 fires when points sit away from the
centre with none near it — the signature of two processes averaged
together (weekday and weekend, two sites reported as one, two coders
applying a definition differently).

Both deserve more attention than the average point outside the limits.

### Variable limits and standardised space

All rules are evaluated on $z = (x_i - \text{CL})/\sigma_i$ rather than
against fixed horizontal bands. Necessary for p-charts, where $\sigma_i$
moves with the denominator: without it, a quiet month with wide limits
would be treated as though it had the same zones as a busy one.

---

## 5. Making Data Count

**Variation** comes from the rules. Nothing to do with the target.

**Assurance** compares the *process limits* with the target — not the
latest point with the target. For a metric where high is good:

| | |
|---|---|
| $\text{LCL} \ge \text{target}$ | **Pass** — the process cannot ordinarily fail |
| $\text{UCL} < \text{target}$ | **Fail** — the process cannot ordinarily pass |
| otherwise | **Hit and miss** — the target sits inside natural variation |

A "pass" is a statement about the *system* and survives a single bad
month. That is correct behaviour, and it is the main reason clinicians
trust these icons more than RAG.

For variable-width limits the most recent period's limits are used: that
is the process as it currently operates, at its current volume.
Averaging limits across periods with very different denominators would
describe a process that never existed.

The icons describe the latest period, per the NHS convention. Where a
series carries historical signals but the current point is unremarkable,
the interface says so explicitly — otherwise a reader looking at the
diamonds will not believe a grey icon.

---

## 6. Funnel plots

A league table ranks units by a point estimate and hides the fact that a
small unit's estimate is mostly noise. Someone is always bottom. Rank a
set of *identical* units on 30 patients each and you still get a
convincing-looking spread.

The funnel plots the indicator against its denominator, with limits that
widen as volume falls. Units inside are consistent with the common rate.
No ranking, no implied ordering.

**Exact limits.** With small denominators the normal approximation gives
limits below 0 or above 1. Binomial quantiles are used directly.

**Overdispersion.** Provider data is more variable than binomial because
case mix and context differ. Spiegelhalter's multiplicative adjustment
estimates a dispersion factor as the mean of the squared z-scores,
**winsorised at the 10th and 90th centiles** so genuine outliers cannot
inflate the limits meant to detect them, and scales the limits by
$\sqrt\phi$.

$\phi \approx 1$: binomial variation suffices. $\phi = 3$ or $4$: most
of the between-unit spread is systematic, and the honest reading is
"these units are not running the same process", not "nine of twelve are
outliers".

**What the funnel does not do is adjust for case mix.** Two sites can
both sit inside while one treats a far sicker population. For mortality
and functional outcome this is the minimum defensible display, not a
sufficient one — see §10.

---

## 7. CUSUM

A Shewhart chart examines one point at a time: fast on large jumps, slow
on small sustained drifts. Service deterioration is usually the latter.

$$C^+_i = \max\!\left(0,\; C^+_{i-1} + (x_i - \mu_0) - k\right), \qquad
  C^-_i = \max\!\left(0,\; C^-_{i-1} - (x_i - \mu_0) - k\right)$$

Subtracting $k$ is what makes this a detector rather than a random walk:
with no shift the increments are negative on average, so the statistic
rests against its zero barrier. $k = \tfrac12\sigma$ tunes it for a
one-sigma shift; $h = 4\sigma$ or $5\sigma$ gives an in-control run
length comparable to a 3-sigma Shewhart chart.

**The cost is interpretability.** A CUSUM says when accumulated evidence
crossed a threshold, not what the rate is now, and the crossing lags the
change. Run it beside the Shewhart chart, never instead of it.

**And the advantage is conditional.** Against whole-series limits — the
realistic default — the CUSUM leads on gradual drift. Against limits
frozen on a clean baseline, the Shewhart chart is very sensitive too and
sometimes fires first. Both behaviours are pinned by tests, including
the one that documents the limitation.

---

## 8. Rare events

When an event is rare, a monthly count chart is mostly zeros: the mean
is tiny, the lower limit pins at zero, and the chart is *structurally
incapable* of showing improvement.

**t-chart** — days between successive events. Improvement appears as
points drifting upward, which is both detectable and, in a ward safety
huddle, considerably more motivating than a row of zeros. Intervals are
strongly right-skewed, so Nelson's transformation $y = t^{1/3.6}$ maps a
Weibull to approximate normality; the XmR is fitted on $y$ and mapped
back with $y^{3.6}$. The back-transformed centre line is a median-like
quantity, not the arithmetic mean interval.

**g-chart** — opportunities (admissions) between events. Geometric, so
$\sigma = \sqrt{\bar g(\bar g + 1)}$, with the centre line at the median
because the distribution is skewed enough that the mean sits well above
the typical value. Preferred when workload varies: 90 days between falls
means something different in a busy month than a quiet one.

---

## 9. Therapy dose

### The denominator decides everything

`sessions` is one row per **applicable patient-day per discipline** —
not per delivered session. A table of attendances cannot answer "on what
proportion of days did this patient receive therapy?", because the
denominator is not in it.

Minutes per day divides by *applicable* days, not delivered days. A
patient receiving 60 minutes on two days out of ten has had 12 minutes a
day of rehabilitation, not 60. Dividing by delivered days is the most
common way therapy dose gets overstated, and the difference is pinned by
a test.

### Frequency and intensity are different problems

The cascade separates them:

```
applicable patient-days
    └─> days with any therapy          the gap here is FREQUENCY
            └─> days meeting 45 min    the gap here is INTENSITY
```

Different causes, different fixes. A single "minutes per day" average
hides both.

### Watch the denominator, not just the rate

"Needs PT" is a clinical judgement recorded by the service being
measured. A team under pressure can improve its 72-hour compliance by
recording fewer patients as needing input, and the indicator will
applaud. The Data Quality page tracks the share of admissions deemed to
need each therapy alongside the compliance rate.

### Capacity arithmetic

$$\text{available minutes} = \text{WTE}_{\text{available}} \times
  \text{hours/week} \times 60 \times \text{contact ratio}$$

Two choices carry the argument:

**Available, not funded WTE.** A service is not short against its
establishment; it is short against the establishment it can actually
field this week, after vacancy, leave and sickness.

**The clinical contact ratio** — the share of contracted time that
becomes face-to-face therapy, after documentation, MDT, handover,
travel, supervision and training. Published inpatient figures typically
fall between 0.50 and 0.65. It is the number a finance director will
challenge first, so measure it locally from a diary study rather than
adopting a published value. It sits on the page as a slider precisely so
its influence on the conclusion is visible.

---

## 10. What this dashboard deliberately does not do

### Risk adjustment

Crude mortality and functional-outcome comparisons between sites are
close to meaningless. Case mix — age, NIHSS, pre-stroke mRS, stroke type
— dominates. The funnel with overdispersion is the minimum defensible
display, not a sufficient one.

Doing it properly: fit a logistic model on those covariates, compute
expected events per site, plot O/E on the funnel. Then be careful about
what you have built. Risk adjustment on variables partly determined by
the care being assessed (a complication, a length of stay) adjusts away
the very signal you are looking for. And a model fitted on the same data
it evaluates is optimistic — cross-validate, or fit on a prior period.

### Causal claims about therapy dose

Sicker patients receive more therapy *and* have worse outcomes, so the
pooled association between dose and disability comes out backwards:
apparently, more therapy makes people worse. Severity drives both
variables. Stratifying by NIHSS band pulls them apart, and the fact that
pooled and within-band associations can point in opposite directions is
Simpson's paradox — visible on the Therapy page, with the stratification
toggleable so you can watch it happen.

Even stratified, this is observational. Therapy allocation responds to
how the patient is progressing, and a patient recovering well is
discharged sooner and receives less total therapy. Time-varying
confounding of that shape needs marginal structural models or a
target-trial emulation, not a cross-sectional regression with severity
thrown in as a covariate.

### Cohort comparisons from the Patient Explorer

Selecting on a pathway breach and comparing outcomes with everyone else
is a case-control design assembled after the fact. It will confidently
reproduce whatever confounding drove the breach — out-of-hours arrivals
breach more and are sicker. Use it to find records worth reading, not to
estimate the effect of the breach.

---

## 11. Data quality checks worth running

**Impossible values beat missing values.** A missing NIHSS means a field
was not collected. An NIHSS of 47 means the field is being populated
from the wrong source — a bigger problem and an easier fix.

**Internal contradictions.** A needle time on a patient not flagged as
thrombolysed. Attended therapy with zero minutes. These reveal joins and
mappings that are wrong in ways no null count will show.

**Completeness within the applicable cohort.** Reporting door-to-needle
as "92% missing" across all admissions is true and useless — it is
missing because those patients were not thrombolysed.

**Digit preference.** Under accurate recording, the final digit of a
minute-level timing should be roughly uniform across 0–9. Spikes at 0
and 5 are the fingerprint of estimation. This matters more than it
sounds: rounding compresses variance, which narrows control limits,
which manufactures special-cause signals out of nothing. Read the excess
on 0 and 5 as the effect size; with a large sample the χ² will reject
uniformity on departures far too small to matter.

**A step change in completeness is a step change in every indicator
built on that field**, and it will present itself as a clinical
improvement. This is the most common way an audit dashboard misleads a
board.

---

## References

- Shewhart, W.A. (1931) *Economic Control of Quality of Manufactured Product.*
- Wheeler, D.J. & Chambers, D.S. (1992) *Understanding Statistical Process Control.*
- Nelson, L.S. (1984) "The Shewhart control chart — tests for special causes", *J. Quality Technology* 16(4).
- Laney, D.B. (2002) "Improved control charts for attributes", *Quality Engineering* 14(4).
- Spiegelhalter, D.J. (2005) "Funnel plots for comparing institutional performance", *Statistics in Medicine* 24(8).
- Perla, R.J. *et al.* (2011) "The run chart: a simple analytical tool for learning from variation in healthcare processes", *BMJ Quality & Safety* 20(1).
- Benneyan, J.C. (2001) "Number-between g-type statistical quality control charts for monitoring adverse events", *Health Care Management Science* 4(4).
- NHS England, *Making Data Count* (2019 onward).
