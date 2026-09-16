"""
Federo Health — Security Laboratory API
"""
from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from security_lab import (
    run_poisoning_scenario,
    run_privacy_attack,
    run_stress_test,
    run_dataset_shift,
    get_governance,
    run_compression_study,
    run_contribution_valuation,
    run_secure_aggregation,
    run_personalization,
    run_async_study,
)

router = APIRouter(prefix="/security", tags=["Security Lab"])


class PoisoningBody(BaseModel):
    use_case: str = "sepsis"
    rounds: int = 10
    malicious_hospitals: Optional[List[str]] = ["h2"]
    attack_type: str = "label_flip"
    poison_ratio: float = 0.5
    attack_scale: float = 4.0
    dp_noise: bool = False
    seeds: int = 3


class PrivacyBody(BaseModel):
    use_case: str = "sepsis"
    epochs_list: Optional[List[int]] = [1, 5, 25]
    dp_noise: bool = True
    seeds: int = 3


class StressBody(BaseModel):
    use_case: str = "sepsis"
    rounds: int = 8
    malicious_hospitals: Optional[List[str]] = ["h3"]
    attack_type: str = "update_poison"
    poison_ratio: float = 0.5
    attack_scale: float = 1.0
    seeds: int = 3


class ShiftBody(BaseModel):
    use_case: str = "sepsis"
    seeds: int = 3


class CompressionBody(BaseModel):
    use_case: str = "sepsis"
    rounds: int = 8
    top_k_frac: float = 0.25
    n_bits: int = 4
    seeds: int = 3


class ContributionBody(BaseModel):
    use_case: str = "sepsis"
    seeds: int = 3


class SecAggBody(BaseModel):
    use_case: str = "sepsis"
    rounds: int = 8
    agg: str = "equity"
    seeds: int = 3


class PersonalizationBody(BaseModel):
    use_case: str = "sepsis"
    rounds: int = 6
    epochs: int = 2


class AsyncBody(BaseModel):
    use_case: str = "sepsis"
    rounds: int = 10
    agg: str = "equity"
    seeds: int = 3
    straggler_penalty: float = 0.5


@router.post("/poisoning")
def api_poisoning(body: PoisoningBody):
    return run_poisoning_scenario(**body.model_dump())


@router.post("/privacy")
def api_privacy(body: PrivacyBody):
    return run_privacy_attack(**body.model_dump())


@router.post("/stress")
def api_stress(body: StressBody):
    return run_stress_test(**body.model_dump())


@router.post("/dataset-shift")
def api_dataset_shift(body: ShiftBody):
    return run_dataset_shift(**body.model_dump())


@router.post("/compression")
def api_compression(body: CompressionBody):
    return run_compression_study(**body.model_dump())


@router.post("/contribution")
def api_contribution(body: ContributionBody):
    return run_contribution_valuation(**body.model_dump())


@router.post("/secure-agg")
def api_secure_agg(body: SecAggBody):
    return run_secure_aggregation(**body.model_dump())


@router.post("/personalization")
def api_personalization(body: PersonalizationBody):
    return run_personalization(**body.model_dump())


@router.post("/async")
def api_async(body: AsyncBody):
    return run_async_study(**body.model_dump())


@router.get("/governance")
def api_governance(use_case: str = "sepsis"):
    return get_governance(use_case)
