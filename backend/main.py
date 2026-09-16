"""
Federo Health — FastAPI Backend
================================
Endpoints:
  POST /api/upload                  → accepts CSV, returns job_id
  POST /api/preprocess/{job_id}     → runs cleaning pipeline, returns Quality Report
  POST /api/train/{job_id}          → kicks off training (synchronous for simplicity)
  GET  /api/train/{job_id}/status   → returns current training status + epoch history
  GET  /api/model/{model_id}/card   → returns full Model Card JSON
  POST /api/federate/{use_case}     → runs equity-weighted FedAvg round

Run with:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import job_store
from models import (
    FederateResponse, ModelCardResponse, QualityReport,
    TrainStartRequest, TrainingStatusResponse, UploadResponse,
)
from pipeline import clean_and_validate_dataset
from trainer import build_model_card, train_model
from federate import run_federated_round
from security_api import router as security_router

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Federo Health API",
    description=(
        "Federo Health Backend — Clinical AI Federation Platform\n\n"
        "Handles server-side data preprocessing, model training (Cloud mode), "
        "equity-weighted FedAvg aggregation, and fairness auditing."
    ),
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

# In-memory model card store keyed by model_id
_model_cards: Dict[str, Dict[str, Any]] = {}


# ── POST /upload ─────────────────────────────────────────────────────────────
@router.post("/upload", response_model=UploadResponse, tags=["Pipeline"])
async def upload_dataset(
    file: UploadFile = File(...),
    use_case: str = Form("sepsis"),
):
    """
    Accept a CSV dataset upload. Returns a job_id for subsequent calls.
    The raw CSV text is stored in-memory for preprocessing.
    """
    raw_bytes = await file.read()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin-1")

    # Quick row count (don't parse yet — full parsing happens in /preprocess)
    lines = [l for l in raw_text.strip().splitlines() if l.strip()]
    row_count = max(0, len(lines) - 1)  # minus header

    job_id = job_store.create_job()
    job_store.update_job(job_id, raw_csv=raw_text, use_case=use_case, status="uploaded")

    return UploadResponse(
        job_id=job_id,
        file_name=file.filename or "upload.csv",
        row_count=row_count,
        use_case=use_case,
    )


# ── POST /preprocess/{job_id} ────────────────────────────────────────────
@router.post("/preprocess/{job_id}", response_model=QualityReport, tags=["Pipeline"])
async def preprocess_dataset(job_id: str):
    """
    Run the full data cleaning pipeline on the uploaded CSV.
    Returns a comprehensive Data Quality Report.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if not job.get("raw_csv"):
        raise HTTPException(status_code=400, detail="No CSV data found for this job. Call /upload first.")

    try:
        result = clean_and_validate_dataset(
            job["raw_csv"],
            use_case=job.get("use_case", "sepsis"),
            winsorize_outliers=True,
            max_missing_rate_drop=0.60,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {str(e)}")

    # Store cleaned data for training step (store as dict, not DataFrame)
    job_store.update_job(job_id, quality_report=result, status="preprocessed")

    scores = result["scores"]
    return QualityReport(
        job_id=job_id,
        raw_header_count=result["raw_header_count"],
        initial_record_count=result["initial_record_count"],
        cleaned_record_count=result["cleaned_record_count"],
        duplicates_removed=result["duplicates_removed"],
        dropped_columns=result["dropped_columns"],
        kept_headers=result["kept_headers"],
        col_missing_summary={
            k: {
                "missing_count": v["missing_count"],
                "missing_rate": v["missing_rate"],
                "col_type": v["col_type"],
                "action": v["action"],
            }
            for k, v in result["col_missing_summary"].items()
        },
        outlier_summary={
            k: {"flagged": v["flagged"], "capped": v["capped"]}
            for k, v in result["outlier_summary"].items()
        },
        fhir_mapping=[
            {
                "column": m["column"],
                "raw_name": m["raw_name"],
                "col_type": m["col_type"],
                "is_mapped": m["is_mapped"],
                "target_concept": m["target_concept"],
                "loinc_code": m["loinc_code"],
                "loinc_display": m["loinc_display"],
            }
            for m in result["fhir_mapping"]
        ],
        scores=scores,
        numeric_feature_names=result["numeric_feature_names"],
        demographic_columns=result["demographic_columns"],
        cleaned_csv=result["cleaned_csv"],
    )


# ── POST /train/{job_id} ─────────────────────────────────────────────────
@router.post("/train/{job_id}", tags=["Training"])
async def start_training(job_id: str, body: TrainStartRequest):
    """
    Train model for the preprocessed job synchronously.
    Returns complete training results, history, and model card immediately.
    """
    job = job_store.get_job(job_id)
    quality_report = job.get("quality_report") if job else None

    # Stateless fallback for Serverless / Lambda environments if job was not found in-memory
    if not quality_report:
        if body.raw_csv:
            try:
                quality_report = clean_and_validate_dataset(
                    raw_csv_text=body.raw_csv,
                    use_case=body.use_case,
                    winsorize_outliers=True,
                    max_missing_rate_drop=0.60,
                )
                job_store.create_job(
                    job_id=job_id,
                    file_name="uploaded_dataset.csv",
                    raw_csv=body.raw_csv,
                    raw_text=body.raw_csv,
                    use_case=body.use_case,
                    quality_report=quality_report,
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to preprocess dataset: {str(e)}")
        else:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found. Please re-upload or provide raw_csv.")

    df = quality_report["df"]
    feature_names = quality_report["numeric_feature_names"]
    categorical_feature_names = quality_report.get("categorical_feature_names", [])
    target_col = quality_report["target_col"]
    demographic_columns = quality_report.get("demographic_columns", [])
    preprocessing_issues = quality_report.get("issues", [])
    image_input = quality_report.get("image_input", False)

    if not feature_names and not categorical_feature_names:
        raise HTTPException(status_code=422, detail="No usable feature columns (numeric or categorical) found in dataset.")

    hospital_info = {
        "id": body.hospital_id,
        "name": body.hospital_name,
        "tier": body.hospital_tier,
        "pub_key_fingerprint": body.pub_key_fingerprint,
    }

    try:
        train_result = train_model(
            df=df,
            feature_names=feature_names,
            target_col=target_col,
            use_case=body.use_case,
            hospital_info=hospital_info,
            regulatory_tags=body.regulatory_tags or [],
            compute_mode=body.compute_mode,
            on_progress=None,
            demographic_columns=demographic_columns,
            categorical_feature_names=categorical_feature_names,
            preprocessing_issues=preprocessing_issues,
            image_input=image_input,
        )

        model_card = build_model_card(
            train_result=train_result,
            data_quality_report=quality_report,
            hospital_info=hospital_info,
            regulatory_tags=body.regulatory_tags or [],
        )

        # Store model card globally
        _model_cards[train_result["model_id"]] = model_card
        _model_cards[job_id] = model_card

        job_store.update_job(job_id,
            status="complete",
            train_result=train_result,
            model_card=model_card,
            training_status={
                "status": "complete",
                "current_epoch": train_result["training_history"][-1]["epoch"] if train_result["training_history"] else 20,
                "total_epochs": 20,
                "history": train_result["training_history"],
                "model_id": train_result["model_id"],
            },
        )

        return {
            "status": "complete",
            "job_id": job_id,
            "model_id": train_result["model_id"],
            "history": train_result["training_history"],
            "model_card": model_card,
        }
    except ValueError as e:
        job_store.update_job(job_id, status="failed", training_status={
            "status": "failed",
            "current_epoch": 0,
            "total_epochs": 20,
            "history": [],
            "model_id": None,
            "error": str(e),
        })
        raise HTTPException(status_code=422, detail=f"Training failed: {str(e)}")
    except Exception as e:
        job_store.update_job(job_id, status="failed", training_status={
            "status": "failed",
            "current_epoch": 0,
            "total_epochs": 20,
            "history": [],
            "model_id": None,
            "error": str(e),
        })
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# ── GET /train/{job_id}/status ──────────────────────────────────────────
@router.get("/train/{job_id}/status", response_model=TrainingStatusResponse, tags=["Training"])
async def get_training_status(job_id: str):
    """
    Poll this endpoint (every 1–2 seconds) for live training progress.
    Returns current epoch, loss/accuracy history, and completion status.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    ts = job.get("training_status") or {"status": "pending", "current_epoch": 0, "total_epochs": 20, "history": [], "model_id": None}

    return TrainingStatusResponse(
        job_id=job_id,
        status=ts.get("status", "pending"),
        current_epoch=ts.get("current_epoch", 0),
        total_epochs=ts.get("total_epochs", 20),
        history=ts.get("history", []),
        model_id=ts.get("model_id"),
    )


# ── GET /model/{model_id}/card ───────────────────────────────────────────
@router.get("/model/{model_id}/card", tags=["Model Card"])
async def get_model_card(model_id: str):
    """
    Return the full Model Card JSON for a trained model.
    This is the same schema used by the frontend ModelCardModal component.
    """
    # Check in model cards store
    if model_id in _model_cards:
        return _model_cards[model_id]

    # Fallback: search all jobs
    for jid, job in job_store._jobs.items():
        mc = job.get("model_card")
        if mc and mc.get("model_id") == model_id:
            return mc

    raise HTTPException(status_code=404, detail=f"Model card for {model_id} not found.")


# ── GET /model/job/{job_id}/card ─────────────────────────────────────────
@router.get("/model/job/{job_id}/card", tags=["Model Card"])
async def get_model_card_by_job(job_id: str):
    """Get model card by job_id (convenience endpoint)."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    mc = job.get("model_card")
    if not mc:
        raise HTTPException(status_code=404, detail="Training not yet complete for this job.")
    return mc


# ── POST /federate/{use_case} ────────────────────────────────────────────
@router.post("/federate/{use_case}", response_model=FederateResponse, tags=["Federation"])
async def federate_models(use_case: str):
    """
    Run one round of equity-weighted FedAvg aggregation for a given use case.
    Returns updated global model metrics + per-hospital equity weights.
    Aggregation method: 70% volume-weighted + 30% equal-weighted (equity blend).
    """
    if use_case not in ("sepsis", "retinopathy"):
        raise HTTPException(status_code=400, detail="use_case must be 'sepsis' or 'retinopathy'.")

    result = run_federated_round(use_case)
    return FederateResponse(**result)


# ── Health check ──────────────────────────────────────────────────────────────
@router.get("/health", tags=["Meta"])
async def health_check():
    return {
        "status": "healthy",
        "service": "Federo Health API v4.0",
        "docs": "/docs",
    }


# Include router under both root prefix and /api prefix for maximum compatibility
app.include_router(router, prefix="")
app.include_router(router, prefix="/api")
# Security Laboratory endpoints (served only under /api, consumed by apiClient.js)
app.include_router(security_router, prefix="/api")
