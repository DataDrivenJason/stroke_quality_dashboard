"""
Data loading, caching, schema contract and validation.
=====================================================================

The schema below is the contract between this dashboard and your data
warehouse. Everything else in the app is written against it, so porting
to real data is a matter of producing two files (or two SQL views) with
these columns -- not of editing the dashboard.

Two design decisions worth flagging:

* **The session table is one row per applicable patient-day, not per
  delivered session.** This is the single most important modelling
  choice in the whole app. A table of delivered sessions cannot answer
  "on what proportion of days did this patient receive therapy?",
  because the denominator -- days on which therapy *should* have
  happened -- is not in it. Services that record only attendances can
  never measure their own reliability, and usually discover this at the
  point they are asked to prove it.

* **Validation is advisory, not blocking.** Real extracts are always
  imperfect, and an app that refuses to open until the data is perfect
  gets abandoned. Instead every problem is reported, quantified, and the
  affected indicator is annotated on the Data Quality page.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import metrics, synth

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------
ADMISSION_SCHEMA: dict[str, tuple[str, bool, str]] = {
    # column: (dtype family, required, description)
    "admission_id": ("string", True, "Unique key for the admission."),
    "site": ("string", True, "Hospital or unit name."),
    "arrival_datetime": ("datetime", True, "Clock start: arrival at first hospital, or onset for in-hospital stroke."),
    "discharge_datetime": ("datetime", False, "Discharge or death timestamp."),
    "age": ("numeric", True, "Age in whole years at admission."),
    "sex": ("string", False, "Recorded sex."),
    "stroke_type": ("string", True, "'Ischaemic' or 'Intracerebral haemorrhage'."),
    "nihss": ("numeric", True, "NIHSS at first assessment (0-42)."),
    "prestroke_mrs": ("numeric", True, "Modified Rankin Scale before this stroke (0-5)."),
    "af": ("bool", True, "Atrial fibrillation known or newly detected."),
    "lives_alone": ("bool", False, "Lived alone before admission."),
    "admitted_from_home": ("bool", True, "Usual residence was own home."),
    "onset_known": ("bool", False, "Time of symptom onset is known."),
    "onset_to_door_min": ("numeric", False, "Minutes from onset to arrival."),
    "door_to_ct_min": ("numeric", True, "Minutes from clock start to first brain imaging."),
    "thrombolysed": ("bool", True, "Received intravenous thrombolysis."),
    "door_to_needle_min": ("numeric", False, "Minutes from clock start to thrombolysis bolus."),
    "thrombectomy": ("bool", True, "Received mechanical thrombectomy."),
    "door_to_puncture_min": ("numeric", False, "Minutes from clock start to arterial puncture."),
    "time_to_su_hours": ("numeric", True, "Hours from clock start to stroke unit arrival."),
    "swallow_screen_hours": ("numeric", True, "Hours from clock start to formal swallow screen."),
    "dysphagia": ("bool", False, "Dysphagia identified."),
    "nihss_recorded": ("bool", False, "NIHSS documented at first assessment."),
    "immobile": ("bool", False, "Immobile, for VTE prophylaxis denominator."),
    "vte_ipc_24h": ("bool", False, "IPC started within 24 hours."),
    "mdt_goals_5d": ("bool", False, "MDT rehabilitation goals agreed within 5 days."),
    "mood_screened": ("bool", False, "Mood screened before discharge."),
    "pt_needed": ("bool", True, "Recorded as requiring physiotherapy."),
    "ot_needed": ("bool", True, "Recorded as requiring occupational therapy."),
    "slt_needed": ("bool", True, "Recorded as requiring speech and language therapy."),
    "pt_assess_hours": ("numeric", True, "Hours from clock start to first PT assessment."),
    "ot_assess_hours": ("numeric", True, "Hours from clock start to first OT assessment."),
    "slt_assess_hours": ("numeric", True, "Hours from clock start to first SLT assessment."),
    "los_days": ("numeric", True, "Inpatient length of stay in whole days."),
    "died_inpatient": ("bool", True, "Died before discharge."),
    "discharge_destination": ("string", True, "Usual residence / Inpatient rehabilitation / Nursing home / Died / Other."),
    "esd_eligible": ("bool", False, "Met local early supported discharge criteria."),
    "esd_referral": ("bool", False, "Referred to early supported discharge."),
    "af_anticoagulated": ("bool", False, "Discharged on an anticoagulant."),
    "antiplatelet": ("bool", False, "Discharged on an antiplatelet."),
    "mrs_discharge": ("numeric", True, "Modified Rankin Scale at discharge (0-6)."),
    "readmitted_30d": ("bool", False, "Emergency readmission within 30 days."),
    "hap": ("bool", False, "Hospital-acquired pneumonia."),
    "falls": ("numeric", False, "Count of inpatient falls this admission."),
    "pressure_ulcers": ("numeric", False, "Count of new pressure ulcers, category 2+."),
}

SESSION_SCHEMA: dict[str, tuple[str, bool, str]] = {
    "admission_id": ("string", True, "Foreign key to the admission."),
    "site": ("string", True, "Hospital or unit name."),
    "discipline": ("string", True, "PT, OT or SLT."),
    "date": ("datetime", True, "Calendar date of the patient-day."),
    "day_of_stay": ("numeric", False, "Days since clock start."),
    "applicable": ("bool", True, "Therapy was indicated on this day. THE DENOMINATOR."),
    "attended": ("bool", True, "At least one session was delivered."),
    "n_sessions": ("numeric", False, "Sessions delivered that day."),
    "minutes": ("numeric", True, "Attended therapy minutes that day (0 if none)."),
    "missed_reason": ("string", False, "Why no therapy happened. Blank when attended."),
}

STAFFING_SCHEMA: dict[str, tuple[str, bool, str]] = {
    "week": ("datetime", True, "Week commencing."),
    "site": ("string", True, "Hospital or unit name."),
    "discipline": ("string", True, "PT, OT or SLT."),
    "wte_funded": ("numeric", True, "Funded establishment in whole-time equivalents."),
    "wte_vacant": ("numeric", False, "Vacant WTE."),
    "wte_absent": ("numeric", False, "WTE lost to leave and sickness."),
    "wte_available": ("numeric", True, "WTE actually available to see patients."),
}

SCHEMAS = {"admissions": ADMISSION_SCHEMA, "sessions": SESSION_SCHEMA,
           "staffing": STAFFING_SCHEMA}


@dataclass
class ValidationIssue:
    table: str
    column: str
    severity: str          # "error" | "warning" | "info"
    message: str
    n_affected: int = 0


# Fields that are *supposed* to be null outside a cohort. Reporting
# "door_to_needle_min is 84% missing" across all admissions is true and
# useless -- it is missing because those patients were not thrombolysed.
# Completeness for these is judged inside the cohort that should have a
# value, which is the same discipline the Data Quality page preaches.
CONDITIONAL_ON: dict[str, str] = {
    "door_to_needle_min": "thrombolysed",
    "door_to_puncture_min": "thrombectomy",
    "onset_to_door_min": "onset_known",
    "pt_assess_hours": "pt_needed",
    "ot_assess_hours": "ot_needed",
    "slt_assess_hours": "slt_needed",
    "af_anticoagulated": "af",
    "vte_ipc_24h": "immobile",
}


# ---------------------------------------------------------------------------
def validate(tables: dict[str, pd.DataFrame]) -> list[ValidationIssue]:
    """Advisory validation. Reports, never blocks."""
    issues: list[ValidationIssue] = []

    for name, schema in SCHEMAS.items():
        df = tables.get(name)
        if df is None:
            issues.append(ValidationIssue(name, "-", "warning", f"Table '{name}' not supplied."))
            continue
        for col, (kind, required, _desc) in schema.items():
            if col not in df.columns:
                issues.append(ValidationIssue(
                    name, col, "error" if required else "info",
                    "Required column missing." if required else "Optional column not supplied."))
                continue
            s = df[col]
            # Restrict the completeness test to the cohort that should
            # carry a value for this field.
            gate = CONDITIONAL_ON.get(col)
            scope = ""
            if name == "admissions" and gate and gate in df.columns:
                applies = df[gate].fillna(False).astype(bool)
                s = s[applies]
                scope = f" among the {gate.replace('_', ' ')} cohort"
            denom = len(s)
            n_null = int(s.isna().sum())
            if n_null and required and denom:
                issues.append(ValidationIssue(
                    name, col, "warning",
                    f"{n_null:,} missing values ({100 * n_null / denom:.1f}%){scope}.",
                    n_null))
            if kind == "numeric" and not pd.api.types.is_numeric_dtype(s):
                issues.append(ValidationIssue(name, col, "error", "Expected numeric."))
            if kind == "datetime" and not pd.api.types.is_datetime64_any_dtype(s):
                issues.append(ValidationIssue(name, col, "error", "Expected a datetime."))

    adm = tables.get("admissions")
    if adm is not None:
        if adm["admission_id"].duplicated().any():
            n = int(adm["admission_id"].duplicated().sum())
            issues.append(ValidationIssue("admissions", "admission_id", "error",
                                          f"{n:,} duplicate admission identifiers.", n))
        # Clinically impossible values are more informative than nulls:
        # they tell you a field is being populated from the wrong source.
        for col, lo, hi in (("nihss", 0, 42), ("prestroke_mrs", 0, 5),
                            ("mrs_discharge", 0, 6), ("age", 0, 120),
                            ("los_days", 0, 400), ("door_to_ct_min", 0, 20000),
                            ("time_to_su_hours", 0, 2000)):
            if col in adm.columns and pd.api.types.is_numeric_dtype(adm[col]):
                bad = int(((adm[col] < lo) | (adm[col] > hi)).sum())
                if bad:
                    issues.append(ValidationIssue(
                        "admissions", col, "error",
                        f"{bad:,} values outside the plausible range {lo}-{hi}.", bad))
        if "door_to_needle_min" in adm.columns and "thrombolysed" in adm.columns:
            orphan = int((adm["door_to_needle_min"].notna() & ~adm["thrombolysed"]).sum())
            if orphan:
                issues.append(ValidationIssue(
                    "admissions", "door_to_needle_min", "warning",
                    f"{orphan:,} records have a needle time but are not flagged as thrombolysed.",
                    orphan))

    ses = tables.get("sessions")
    if ses is not None and adm is not None:
        unknown = int((~ses["admission_id"].isin(adm["admission_id"])).sum())
        if unknown:
            issues.append(ValidationIssue("sessions", "admission_id", "error",
                                          f"{unknown:,} rows reference an unknown admission.", unknown))
        if {"attended", "minutes"} <= set(ses.columns):
            contra = int((ses["attended"] & (ses["minutes"] <= 0)).sum())
            if contra:
                issues.append(ValidationIssue(
                    "sessions", "minutes", "warning",
                    f"{contra:,} attended patient-days record zero minutes.", contra))
            contra2 = int((~ses["attended"] & (ses["minutes"] > 0)).sum())
            if contra2:
                issues.append(ValidationIssue(
                    "sessions", "minutes", "error",
                    f"{contra2:,} non-attended patient-days record therapy minutes.", contra2))
    return issues


# ---------------------------------------------------------------------------
def coerce(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Best-effort type coercion against the schema."""
    out = {}
    for name, df in tables.items():
        schema = SCHEMAS.get(name)
        if schema is None:
            out[name] = df
            continue
        d = df.copy()
        for col, (kind, _req, _desc) in schema.items():
            if col not in d.columns:
                continue
            if kind == "datetime":
                d[col] = pd.to_datetime(d[col], errors="coerce")
            elif kind == "numeric":
                d[col] = pd.to_numeric(d[col], errors="coerce")
            elif kind == "bool":
                if d[col].dtype != bool:
                    d[col] = (d[col].astype("string").str.strip().str.lower()
                              .isin(["true", "1", "yes", "y", "t"]))
        out[name] = d
    return out


def cached_synthetic(seed: int = synth.SEED) -> dict[str, pd.DataFrame]:
    """Generate once, then read from parquet."""
    DATA_DIR.mkdir(exist_ok=True)
    stamp = DATA_DIR / f"synthetic_{seed}"
    names = ["admissions", "sessions", "therapists", "staffing"]
    if all((stamp.parent / f"{stamp.name}_{n}.parquet").exists() for n in names):
        try:
            return {n: pd.read_parquet(stamp.parent / f"{stamp.name}_{n}.parquet")
                    for n in names}
        except Exception:
            pass
    tables = synth.generate(seed)
    for n, df in tables.items():
        try:
            df.to_parquet(stamp.parent / f"{stamp.name}_{n}.parquet", index=False)
        except Exception:
            pass
    return tables


def prepare(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Attach derived flags. Safe to call repeatedly."""
    tables = dict(tables)
    tables["admissions"] = metrics.derive_flags(tables["admissions"])
    return tables


def template_zip() -> bytes:
    """A CSV template bundle: headers, one example row, and a data dictionary."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, schema in SCHEMAS.items():
            header = ",".join(schema.keys())
            z.writestr(f"{name}_template.csv", header + "\n")
            dd = ["column,type,required,description"]
            for col, (kind, req, desc) in schema.items():
                dd.append(f'{col},{kind},{"yes" if req else "no"},"{desc}"')
            z.writestr(f"{name}_data_dictionary.csv", "\n".join(dd) + "\n")
        z.writestr("README.txt",
                   "Stroke quality dashboard - data contract\n"
                   "=======================================\n\n"
                   "Populate admissions_template.csv and sessions_template.csv (staffing is\n"
                   "optional but unlocks the caseload and capacity page).\n\n"
                   "The single most important column is sessions.applicable. It must be TRUE\n"
                   "for every patient-day on which therapy was indicated, whether or not any\n"
                   "therapy happened. Without it the percentage-of-days and 45-minute\n"
                   "indicators have no denominator and cannot be calculated.\n\n"
                   "Booleans accept TRUE/FALSE, 1/0, yes/no.\n"
                   "Datetimes should be ISO 8601 (YYYY-MM-DD HH:MM:SS).\n")
    return buf.getvalue()
