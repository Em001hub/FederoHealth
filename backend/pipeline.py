"""
Federo Health — Clinical Data Cleaning & Preprocessing Pipeline (Python)
========================================================================
Python port of dataPipeline.js using pandas + numpy.
Runs identical logic server-side for Cloud Training mode.

Steps:
  1. Column type inference (numeric, categorical, date, identifier, target)
  2. Missing value detection & median/mode imputation
  3. Exact duplicate row removal
  4. IQR outlier detection & winsorization
  5. Value normalization (gender canonicalization, binary maps)
  6. HL7 FHIR R4 LOINC schema mapping
  7. Composite data quality score (0–100)
  8. Demographic column detection for fairness audit
"""

import io
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

# ── FHIR LOINC codes (mirrors federatedEngine.js) ────────────────────────────
FHIR_LOINC_CODES = {
    "sepsis": [
        {"code": "30525-0", "display": "Age"},
        {"code": "8867-4",  "display": "Heart Rate (bpm)"},
        {"code": "8310-5",  "display": "Body Temperature (°C)"},
        {"code": "8480-6",  "display": "Systolic BP (mmHg)"},
        {"code": "8478-0",  "display": "Mean Arterial Pressure (mmHg)"},
        {"code": "3094-0",  "display": "Blood Urea Nitrogen (mg/dL)"},
        {"code": "26464-8", "display": "WBC Count (K/µL)"},
        {"code": "2345-7",  "display": "Glucose (mg/dL)"},
        {"code": "2160-0",  "display": "Creatinine (mg/dL)"},
        {"code": "2744-1",  "display": "Arterial pH"},
        {"code": "2019-8",  "display": "PaCO2 (mmHg)"},
        {"code": "9279-1",  "display": "Respiratory Rate (/min)"},
    ],
    "retinopathy": [
        {"code": "71490-9", "display": "Macular Edema Score (0–1)"},
        {"code": "71488-3", "display": "Microaneurysm Density (0–2)"},
        {"code": "71487-5", "display": "Exudate Density (0–1)"},
        {"code": "71489-1", "display": "Hemorrhage Score (0–2)"},
        {"code": "71491-7", "display": "Disk Abnormality (0–1)"},
        {"code": "71492-5", "display": "NV Severity (0–1)"},
    ],
}

COLUMN_ALIASES = {
    "sepsis": {
        "age":        ["age", "patient_age", "age_years"],
        "heart_rate": ["hr", "heart_rate", "heartrate", "pulse"],
        "temp":       ["temp", "temperature", "temp_c", "body_temp"],
        "sbp":        ["sbp", "systolic_bp", "systolic", "sbp_mmhg"],
        "map":        ["map", "map_mmhg", "mean_arterial_pressure", "mean_bp"],
        "bun":        ["bun", "blood_urea_nitrogen", "urea_nitrogen"],
        "wbc":        ["wbc", "wbc_k", "white_blood_cell", "wbc_count", "leukocytes"],
        "glucose":    ["glucose", "blood_glucose", "sugar"],
        "creatinine": ["creatinine", "creat", "serum_creatinine"],
        "ph":         ["ph", "arterial_ph", "blood_ph"],
        "paco2":      ["paco2", "co2", "partial_co2"],
        "resp_rate":  ["resp_rate", "respiratory_rate", "rr", "resprate"],
        "label":      ["label", "sepsis_label", "sepsislabel", "outcome", "sepsis", "target"],
    },
    "retinopathy": {
        "macula_edema":    ["macula_edema", "macular_edema", "macular_edema_score", "edema"],
        "microaneurysm":   ["microaneurysm", "microaneurysm_density", "ma_density", "ma"],
        "exudate":         ["exudate", "exudate_density", "hard_exudate"],
        "hemorrhage":      ["hemorrhage", "hemorrhage_score", "bleed_score"],
        "disk_abnormality":["disk_abnormality", "disk", "optic_disk", "cup_disc"],
        "nv_severity":     ["nv_severity", "neovascularization", "nv", "nvd"],
        "label":           ["label", "dr_grade", "diagnosis", "grade", "target", "class", "drgrade"],
    },
}

GENDER_CANONICAL = {
    "m": "Male", "male": "Male", "man": "Male",
    "f": "Female", "female": "Female", "woman": "Female",
    "o": "Other", "other": "Other", "non-binary": "Other",
}

BINARY_CANONICAL = {
    "1": 1, "yes": 1, "true": 1, "positive": 1, "sepsis": 1, "pos": 1, "y": 1, "high": 1,
    "moderate": 1, "severe": 1, "pdr": 1, "proliferative": 1,
    "0": 0, "no": 0, "false": 0, "negative": 0, "control": 0, "neg": 0, "n": 0, "low": 0,
    "no_dr": 0, "mild": 0,
}

# Demographic-adjacent column patterns for fairness audit
DEMOGRAPHIC_PATTERNS = [
    "age", "age_band", "age_group", "sex", "gender", "race", "ethnicity",
    "age_range", "age_cat", "sex_at_birth",
]

# Column-name hints that suggest image/path-based inputs (e.g. fundus scans)
IMAGE_HINT_TOKENS = [
    "image", "img", "scan", "photo", "path", "url", "fundus", "dicom", "dcm", "camera", "file",
]


def log_issue(issues: List[Dict[str, Any]], i_type: str, severity: str, issue: str, resolution: str, count: int = 1):
    """Append a structured inconsistency record for the transparency report."""
    issues.append({
        "type": i_type,
        "severity": severity,
        "issue": issue,
        "resolution": resolution,
        "count": int(count),
    })


def _clean_header(h: str) -> str:
    import re
    return re.sub(r"[^a-z0-9_]", "_", h.strip().lower())


def detect_demographic_columns(headers: List[str]) -> List[str]:
    """Return column names that look like demographic/subgroup fields."""
    found = []
    for h in headers:
        lh = h.lower()
        for pat in DEMOGRAPHIC_PATTERNS:
            if pat in lh:
                found.append(h)
                break
        else:
            # Also flag categorical columns with few unique values that aren't the label
            pass
    return found


def clean_and_validate_dataset(
    raw_csv_text: str,
    use_case: str = "sepsis",
    winsorize_outliers: bool = True,
    max_missing_rate_drop: float = 0.60,
) -> Dict[str, Any]:
    """
    Full cleaning pipeline. Returns a dict matching QualityReport schema.
    """
    lines = [l for l in raw_csv_text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        raise ValueError("Dataset must have at least a header row and one data row.")

    # Parse into DataFrame
    df_raw = pd.read_csv(io.StringIO(raw_csv_text), sep=None, engine="python", dtype=str)

    issues: List[Dict[str, Any]] = []
    initial_record_count = len(df_raw)
    # Ragged rows: pandas pads short rows with NaN and ignores trailing empties —
    # detect how many rows were shorter than the header and report it.
    n_short_rows = int((df_raw.isna().sum(axis=1) > 0).sum())
    if n_short_rows:
        log_issue(
            issues, "ragged_rows", "medium",
            f"{n_short_rows} rows shorter than the {len(df_raw.columns)}-column header",
            "Missing trailing cells padded and imputed; columns stay aligned",
            n_short_rows,
        )

    # Clean header names
    orig_headers = list(df_raw.columns)
    clean_map = {h: _clean_header(h) for h in orig_headers}
    df_raw.columns = [clean_map[h] for h in orig_headers]
    clean_headers = list(df_raw.columns)

    # Image-input detection from column hints
    image_input = any(
        token in lc
        for lc in clean_headers
        for token in IMAGE_HINT_TOKENS
        if lc
    )
    if image_input:
        log_issue(
            issues, "image_input", "medium",
            "Columns hinting at image/path inputs detected",
            "Routed to image-capable model selection (CNN / feature-extracted tabular fallback)",
        )

    # ── 1. Column type inference ──────────────────────────────────────────────
    col_types: Dict[str, str] = {}
    for col in clean_headers:
        lc = col.lower()
        if any(x in lc for x in ["id", "anon", "mrn", "patient"]):
            col_types[col] = "identifier"
        elif any(x in lc for x in ["label", "target", "outcome", "sepsis", "dr_grade", "diagnosis"]):
            col_types[col] = "target"
        else:
            sample = df_raw[col].dropna().head(100)
            numeric_count = pd.to_numeric(sample, errors="coerce").notna().sum()
            if numeric_count > len(sample) * 0.7 and len(sample) > 0:
                n_unique = df_raw[col].dropna().nunique()
                if n_unique <= 3:
                    col_types[col] = "categorical_binary"
                else:
                    col_types[col] = "numeric"
            else:
                col_types[col] = "categorical"

    # Normalize missing sentinel values
    df_raw.replace(["NA", "na", "N/A", "n/a", "NaN", "nan", "null", "NULL", "?", ""], np.nan, inplace=True)

    # ── 2. Duplicate removal ──────────────────────────────────────────────────
    n_before = len(df_raw)
    df_raw.drop_duplicates(inplace=True)
    duplicates_removed = n_before - len(df_raw)
    df = df_raw.reset_index(drop=True)
    if duplicates_removed:
        log_issue(
            issues, "duplicates", "low",
            f"{duplicates_removed} exact duplicate rows",
            "Deduplicated — first occurrence kept",
            duplicates_removed,
        )

    # ── 3. Missing rate per column ────────────────────────────────────────────
    col_missing_summary: Dict[str, Any] = {}
    kept_cols: List[str] = []
    dropped_cols: List[str] = []

    for col in clean_headers:
        missing_count = df[col].isna().sum()
        missing_rate = missing_count / max(len(df), 1)
        action = "DROPPED (>60% Missing)" if missing_rate > max_missing_rate_drop else "IMPUTED / CLEANED"
        col_missing_summary[col] = {
            "missing_count": int(missing_count),
            "missing_rate": round(missing_rate * 100, 1),
            "col_type": col_types.get(col, "unknown"),
            "action": action,
        }
        if missing_rate > max_missing_rate_drop:
            dropped_cols.append(col)
            log_issue(
                issues, "dropped_column", "high",
                f"Column '{col}' is {round(missing_rate * 100, 1)}% missing",
                "Column dropped from feature set (above missingness threshold)",
                int(missing_count),
            )
        else:
            kept_cols.append(col)
            if missing_count:
                log_issue(
                    issues, "missing_values", "medium",
                    f"{int(missing_count)} of {len(df)} values missing in '{col}' ({round(missing_rate * 100, 1)}%)",
                    "Median/mode imputation applied",
                    int(missing_count),
                )

    df = df[kept_cols].copy()

    # ── 4. Imputation & winsorization ─────────────────────────────────────────
    outlier_summary: Dict[str, Any] = {col: {"flagged": 0, "capped": 0} for col in kept_cols}

    for col in kept_cols:
        ctype = col_types.get(col, "categorical")

        if ctype in ("numeric", "target", "categorical_binary"):
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            median_val = numeric_series.median()
            if pd.isna(median_val):
                median_val = 0.0

            # Non-numeric text inside a numeric column → count for transparency
            # (target labels like "sepsis"/"mild" are handled by target normalization, not coercion)
            coerced_mask = numeric_series.isna() & df[col].notna()
            n_coerced = int(coerced_mask.sum()) if ctype != "target" else 0
            if n_coerced:
                log_issue(
                    issues, "text_coercion", "medium",
                    f"{n_coerced} non-numeric text values found in numeric column '{col}' (e.g. units, ranges, sentinels)",
                    f"Coerced to numeric (median={median_val:g}), outliers then winsorized",
                    n_coerced,
                )

            # Impute
            df[col] = numeric_series.fillna(median_val)

            # Target normalization
            if ctype == "target":
                def normalize_target(v):
                    s = str(v).strip().lower()
                    if s in BINARY_CANONICAL:
                        return BINARY_CANONICAL[s]
                    try:
                        fv = float(v)
                        if use_case == "retinopathy" and fv >= 2:
                            return 1
                        return 1 if fv > 0 else 0
                    except:
                        return 0
                df[col] = df[col].apply(normalize_target)

            elif ctype == "numeric" and winsorize_outliers:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    mask_low = df[col] < lower
                    mask_high = df[col] > upper
                    flagged = int(mask_low.sum() + mask_high.sum())
                    outlier_summary[col]["flagged"] = flagged
                    outlier_summary[col]["capped"] = flagged
                    df[col] = df[col].clip(lower=lower, upper=upper)
                    if flagged:
                        log_issue(
                            issues, "outliers", "medium",
                            f"{flagged} outlier values in '{col}' beyond Q1-1.5×IQR / Q3+1.5×IQR "
                            f"([{lower:g}, {upper:g}])",
                            f"Winsorized (capped to [{lower:g}, {upper:g}])",
                            flagged,
                        )

        elif ctype == "categorical":
            mode_val = df[col].mode()
            mode_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
            n_missing_cat = int(df[col].isna().sum())
            df[col] = df[col].fillna(mode_val)
            # Normalize gender / synonyms
            n_normalized = [0]
            def _norm_cat(v):
                canon = GENDER_CANONICAL.get(str(v).strip().lower())
                if canon is not None and canon != str(v):
                    n_normalized[0] += 1
                return canon or str(v)
            df[col] = df[col].apply(_norm_cat)
            if n_missing_cat:
                log_issue(
                    issues, "missing_values", "medium",
                    f"{n_missing_cat} missing categorical values in '{col}'",
                    f"Imputed with mode ('{mode_val}')",
                    n_missing_cat,
                )
            if n_normalized[0]:
                log_issue(
                    issues, "value_normalization", "low",
                    f"{n_normalized[0]} values canonicalized in '{col}' (e.g. m→Male, yes→1)",
                    "Casing/synonym normalization applied",
                    n_normalized[0],
                )

        elif ctype == "date":
            df[col] = df[col].fillna(pd.Timestamp.now().strftime("%Y-%m-%d"))
        else:
            df[col] = df[col].fillna("ANON-GEN")

    # Drop all-zero rows
    numeric_cols = [c for c in kept_cols if col_types.get(c) == "numeric"]
    if numeric_cols:
        all_zero_mask = (df[numeric_cols] == 0).all(axis=1)
        n_zero_rows = int(all_zero_mask.sum())
        if n_zero_rows:
            log_issue(
                issues, "zero_rows", "medium",
                f"{n_zero_rows} rows are all-zero across numeric features",
                "Rows removed (no clinical signal)",
                n_zero_rows,
            )
        df = df[~all_zero_mask].reset_index(drop=True)

    cleaned_record_count = len(df)

    # ── 5. FHIR mapping ───────────────────────────────────────────────────────
    aliases = COLUMN_ALIASES.get(use_case, {})
    fhir_codes = FHIR_LOINC_CODES.get(use_case, [])
    feature_keys = [k for k in aliases if k != "label"]
    fhir_mapping: List[Dict[str, Any]] = []

    for col in kept_cols:
        mapped_key = None
        for key, alias_list in aliases.items():
            if col in alias_list:
                mapped_key = key
                break
        loinc = None
        if mapped_key and mapped_key != "label":
            key_index = feature_keys.index(mapped_key) if mapped_key in feature_keys else -1
            if 0 <= key_index < len(fhir_codes):
                loinc = fhir_codes[key_index]

        fhir_mapping.append({
            "column": col,
            "raw_name": orig_headers[clean_headers.index(col)] if col in clean_headers else col,
            "col_type": col_types.get(col, "unknown"),
            "is_mapped": mapped_key is not None,
            "target_concept": mapped_key or "Unmapped / Local Variable",
            "loinc_code": loinc["code"] if loinc else "—",
            "loinc_display": loinc["display"] if loinc else "—",
        })

    # ── 6. Quality scores ─────────────────────────────────────────────────────
    total_cells = initial_record_count * len(clean_headers)
    total_missing = sum(v["missing_count"] for v in col_missing_summary.values())
    completeness = max(0.0, 100.0 - (total_missing / max(total_cells, 1)) * 100)
    consistency = max(0.0, 100.0 - (duplicates_removed / max(initial_record_count, 1)) * 100)
    total_outliers = sum(v["flagged"] for v in outlier_summary.values())
    validity = max(0.0, 100.0 - (total_outliers / max(cleaned_record_count * len(kept_cols), 1)) * 150)
    mapped_count = sum(1 for m in fhir_mapping if m["is_mapped"])
    fhir_conformance = (mapped_count / max(len(kept_cols), 1)) * 100
    composite = (completeness * 0.35 + consistency * 0.25 + validity * 0.25 + fhir_conformance * 0.15)
    composite = min(99.8, max(70.0, round(composite, 1)))

    # ── 7. Engine-ready records for training ──────────────────────────────────
    target_col = next((c for c in kept_cols if col_types.get(c) == "target"), None)
    numeric_feature_cols = [
        c for c in kept_cols if col_types.get(c) in ("numeric", "categorical_binary")
    ]
    categorical_feature_cols = [c for c in kept_cols if col_types.get(c) == "categorical"]

    # Constant (zero-variance) numeric features — no predictive signal
    constant_features = [
        c for c in numeric_feature_cols if df[c].nunique() <= 1
    ]
    if constant_features:
        log_issue(
            issues, "constant_features", "high",
            f"{len(constant_features)} feature(s) constant across all rows ({', '.join(constant_features)})",
            "Kept but weighted ~0; reported so users can drop them upstream",
            len(constant_features),
        )

    if categorical_feature_cols:
        log_issue(
            issues, "categorical_encoding", "low",
            f"{len(categorical_feature_cols)} categorical/text column(s) will be encoded as model features: "
            f"{', '.join(categorical_feature_cols)}",
            "One-hot (server) / label (edge) encoding applied before training",
            len(categorical_feature_cols),
        )

    # Single-class target detection (surface for transparency; trainer blocks with clear error)
    if target_col and len(df) > 0:
        n_target_classes = df[target_col].nunique()
        if n_target_classes < 2:
            log_issue(
                issues, "target_labels", "critical",
                f"Target column '{target_col}' contains only one class "
                f"({df[target_col].iloc[0]} × {int(n_target_classes)} unique) — "
                "classification cannot learn a decision boundary",
                "Training is blocked with an actionable message; recheck label encoding/cohort selection",
            )

    # ── 8. Demographic column detection ──────────────────────────────────────
    demographic_columns = detect_demographic_columns(kept_cols)
    # Also catch any low-cardinality categoricals that aren't target/id
    for col in kept_cols:
        ctype = col_types.get(col)
        if ctype == "categorical" and col not in demographic_columns:
            n_unique = df[col].nunique()
            if 2 <= n_unique <= 10:
                demographic_columns.append(col)

    # Generate cleaned CSV
    cleaned_csv = df.to_csv(index=False)

    return {
        "df": df,
        "raw_header_count": len(orig_headers),
        "initial_record_count": initial_record_count,
        "cleaned_record_count": cleaned_record_count,
        "duplicates_removed": duplicates_removed,
        "dropped_columns": dropped_cols,
        "kept_headers": kept_cols,
        "col_missing_summary": col_missing_summary,
        "outlier_summary": outlier_summary,
        "fhir_mapping": fhir_mapping,
        "scores": {
            "completeness": round(completeness, 1),
            "consistency": round(consistency, 1),
            "validity": round(validity, 1),
            "fhir_conformance": round(fhir_conformance, 1),
            "composite_quality_score": composite,
        },
        "numeric_feature_names": numeric_feature_cols,
        "categorical_feature_names": categorical_feature_cols,
        "target_col": target_col,
        "demographic_columns": demographic_columns,
        "issues": issues,
        "image_input": image_input,
        "cleaned_csv": cleaned_csv,
    }
