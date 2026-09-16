"""
Federo Health — Pydantic Request/Response Models
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


# ── Upload ───────────────────────────────────────────────────────────────────
class UploadResponse(BaseModel):
    job_id: str
    file_name: str
    row_count: int
    use_case: str


# ── Preprocess ───────────────────────────────────────────────────────────────
class ColMissing(BaseModel):
    missing_count: int
    missing_rate: float
    col_type: str
    action: str


class OutlierEntry(BaseModel):
    flagged: int
    capped: int


class FHIRMapEntry(BaseModel):
    column: str
    raw_name: str
    col_type: str
    is_mapped: bool
    target_concept: str
    loinc_code: str
    loinc_display: str


class QualityScores(BaseModel):
    completeness: float
    consistency: float
    validity: float
    fhir_conformance: float
    composite_quality_score: float


class QualityReport(BaseModel):
    job_id: str
    raw_header_count: int
    initial_record_count: int
    cleaned_record_count: int
    duplicates_removed: int
    dropped_columns: List[str]
    kept_headers: List[str]
    col_missing_summary: Dict[str, ColMissing]
    outlier_summary: Dict[str, OutlierEntry]
    fhir_mapping: List[FHIRMapEntry]
    scores: QualityScores
    numeric_feature_names: List[str]
    demographic_columns: List[str]   # detected for fairness audit
    cleaned_csv: str


# ── Training Status ───────────────────────────────────────────────────────────
class EpochRecord(BaseModel):
    epoch: int
    loss: float
    accuracy: float


class TrainingStatusResponse(BaseModel):
    job_id: str
    status: str              # "running" | "complete" | "failed"
    current_epoch: int
    total_epochs: int
    history: List[EpochRecord]
    model_id: Optional[str] = None


# ── Train Start ───────────────────────────────────────────────────────────────
class TrainStartRequest(BaseModel):
    hospital_id: str
    hospital_name: str
    hospital_tier: Optional[str] = "Community"
    pub_key_fingerprint: Optional[str] = "0x000000000000"
    regulatory_tags: Optional[List[str]] = ["IRB Approved", "HIPAA De-Identified Safe Harbor"]
    use_case: str = "sepsis"
    compute_mode: str = "cloud"   # "cloud" | "edge"
    raw_csv: Optional[str] = None


# ── Model Card ────────────────────────────────────────────────────────────────
class ConfusionMatrix(BaseModel):
    tp: int
    fp: int
    tn: int
    fn: int
    total: int


class PerformanceMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auroc: float
    confusion_matrix: ConfusionMatrix


class FeatureImportance(BaseModel):
    feature: str
    importance: float
    normalized_score: float


class SubgroupResult(BaseModel):
    subgroup: str
    column: str
    accuracy: float
    f1_score: float
    count: int
    flagged: bool              # True if >10pp below overall


class FairnessAudit(BaseModel):
    demographic_columns: List[str]
    subgroup_results: List[SubgroupResult]
    overall_accuracy: float
    has_demographic_data: bool


class FederationImpact(BaseModel):
    local_accuracy: float
    federated_accuracy: float
    delta: float


class ModelCardResponse(BaseModel):
    model_id: str
    version: str
    name: str
    use_case: str
    created_at: str
    status: str
    training_mode: str         # "cloud" | "edge"
    provenance: Dict[str, Any]
    compliance: Dict[str, Any]
    dataset_profile: Dict[str, Any]
    model_architecture: Dict[str, Any]
    performance: PerformanceMetrics
    feature_importances: List[FeatureImportance]
    training_curve: List[EpochRecord]
    privacy: Dict[str, Any]
    weights_hash: str
    signature_info: Dict[str, Any]
    federation_impact: Optional[FederationImpact] = None
    fairness_audit: Optional[FairnessAudit] = None
    analysis: Optional[Dict[str, Any]] = None


# ── Federate ─────────────────────────────────────────────────────────────────
class HospitalWeight(BaseModel):
    hospital_id: str
    hospital_name: str
    data_volume: int
    volume_weight: float
    equity_weight: float      # blended 70/30 weight


class FederateResponse(BaseModel):
    use_case: str
    round_number: int
    aggregation_method: str   # "equity_weighted_fedavg"
    global_accuracy: float
    global_loss: float
    global_auroc: float
    hospital_weights: List[HospitalWeight]
    participating_hospitals: int
