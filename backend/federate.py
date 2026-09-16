"""
Federo Health — Equity-Weighted FedAvg Aggregation
==================================================
Aggregates per-hospital local SGD updates on the REAL clinical cohorts using
the security_lab engine. Every metric returned is measured on held-out data
from those same real records.
"""
import math
from typing import Any, Dict, List

import numpy as np

from security_lab import (
    HOSPITALS,
    HOSPITAL_IDS,
    TOTAL_VOLUME,
    _equity_weight,
    federation_round,
    current_federated_accuracy,
)


def compute_equity_weights(
    hospital_volumes: List[Dict[str, Any]],
    volume_ratio: float = 0.70,
    equal_ratio: float = 0.30,
) -> List[Dict[str, Any]]:
    """
    Compute equity-blended weights for each hospital.
    """
    k = len(hospital_volumes)
    if k == 0:
        return []

    total_n = sum(h["data_volume"] for h in hospital_volumes)
    equal_w = 1.0 / k

    result = []
    for h in hospital_volumes:
        vol_w = h["data_volume"] / max(total_n, 1)
        equity_w = volume_ratio * vol_w + equal_ratio * equal_w
        result.append({
            "hospital_id": h["hospital_id"],
            "hospital_name": h["hospital_name"],
            "data_volume": h["data_volume"],
            "volume_weight": round(vol_w, 4),
            "equity_weight": round(equity_w, 4),
        })
    return result


def run_federated_round(use_case: str) -> Dict[str, Any]:
    """
    Run one real equity-weighted FedAvg round via security_lab.
    Returns metrics measured on held-out data + per-hospital equity weights.
    """
    result = federation_round(use_case)

    volumes = [
        {"hospital_id": h["id"], "hospital_name": h["name"], "data_volume": h["volume"]}
        for h in HOSPITALS
    ]
    hospital_weights = compute_equity_weights(volumes)

    return {
        "use_case": use_case,
        "round_number": result["round_number"],
        "aggregation_method": result["aggregation_method"],
        "global_accuracy": result["global_accuracy"],
        "global_loss": result["global_loss"],
        "global_auroc": result["global_auroc"],
        "hospital_weights": hospital_weights,
        "participating_hospitals": result["participating_hospitals"],
    }


def get_federated_accuracy(use_case: str) -> float:
    """Return the current real global federated accuracy for a use case."""
    return current_federated_accuracy(use_case)
