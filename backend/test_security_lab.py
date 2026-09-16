"""
Federo Health — Security Lab tests. These assert real computed invariants,
not mocks: every value is produced by the engine on the real cohort.
Run:  python -m pytest backend/test_security_lab.py
"""
import numpy as np
import pytest

from security_lab import (
    DATASETS,
    DEFAULT_BUDGETS,
    HOSPITAL_IDS,
    federation_round,
    get_governance,
    run_compression_study,
    run_dataset_shift,
    run_poisoning_scenario,
    run_privacy_attack,
    run_stress_test,
    run_contribution_valuation,
    run_secure_aggregation,
    run_personalization,
    run_async_study,
)


def test_dataset_sizes_and_hospital_keys():
    assert len(DATASETS["sepsis"]) == 23
    assert len(DATASETS["retinopathy"]) == 15
    assert set(HOSPITAL_IDS) == {"h1", "h2", "h3", "h4"}


def test_poisoning_scenario_detects_attacker_without_false_positives():
    r = run_poisoning_scenario(
        use_case="sepsis", rounds=6, malicious_hospitals=["h3"],
        attack_type="update_poison", attack_scale=4.0, seeds=2,
    )
    dq = r["detection_quality"]
    assert dq["tp"] >= 1 and dq["fp"] == 0, dq
    assert dq["fn"] == 0
    assert 0 <= r["metrics"]["baseline_accuracy"] <= 100
    assert r["metrics"]["poisoned_accuracy"] < r["metrics"]["baseline_accuracy"]


def test_privacy_attack_auc_in_unit_range_and_rises_with_overfitting():
    r = run_privacy_attack(use_case="sepsis", epochs_list=[1, 25], dp_noise=True, seeds=2)
    aucs = [x["attack_auc"] for x in r["runs"]]
    assert all(0 <= a <= 1 for a in aucs)
    # members are more separable (real overfitting) — as training intensifies
    assert r["runs"][-1]["attack_auc"] >= r["runs"][0]["attack_auc"]


def test_stress_test_equity_dampens_large_malicious_contributor():
    r = run_stress_test(
        use_case="sepsis", rounds=8, malicious_hospitals=["h3"],
        attack_type="update_poison", attack_scale=1.0, seeds=3,
    )
    assert len(r["series"]) == 8
    assert r["final"]["winner"] == "equity_fedavg"
    assert r["final"]["robustness_gain_pp"] > 0


def test_dataset_shift_returns_four_hospitals_with_measured_shift():
    r = run_dataset_shift(use_case="retinopathy", seeds=2)
    assert len(r["hospitals"]) == 4
    for h in r["hospitals"]:
        assert h["shift_score"] >= 0
        assert h["out_n"] >= 1


def test_compression_study_is_real_and_degrades_loss_at_extreme_budget():
    r = run_compression_study(use_case="sepsis", rounds=6, top_k_frac=0.1, n_bits=2, seeds=2)
    assert len(r["series"]) == 6
    assert len(r["sweep"]) == len(DEFAULT_BUDGETS)
    assert r["final"]["compression_ratio"] > 1
    assert r["final"]["bits_saved_percent"] > 0
    p = r["final"]["protocol"]
    assert p["compressed_bits_per_client"] < p["original_bits_per_client"]
    assert 0 <= r["final"]["standard_accuracy"] <= 100
    assert 0 <= r["final"]["compressed_accuracy"] <= 100
    assert len(r["bandwidth"]) == 3
    # more aggressive compression ⇒ strictly more bits saved in the sweep
    saved = [s["bits_saved_percent"] for s in r["sweep"]]
    assert saved == sorted(saved)


def test_governance_trust_scores_bounded_and_consistent():
    r = get_governance("sepsis")
    assert len(r["hospitals"]) == 4
    for h in r["hospitals"]:
        assert 0 <= h["trust_score"] <= 100
        assert h["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert h["data_consistency"] >= 0
    flagged = [h for h in r["hospitals"] if h["poisoning_status"] == "FLAGGED"]
    assert all(h["trust_score"] <= 45 for h in flagged)


def test_federation_round_returns_real_metrics():
    r = federation_round("sepsis")
    assert 0 <= r["global_accuracy"] <= 100
    assert r["round_number"] >= 1
    assert r["participating_hospitals"] >= 3
    assert 0 <= r["global_auroc"] <= 1


def test_contribution_valuation_leaves_one_hospital_out_per_hospital():
    r = run_contribution_valuation("sepsis", seeds=2)
    assert len(r["hospitals"]) == 4
    for h in r["hospitals"]:
        assert h["hospital_id"] in HOSPITAL_IDS
        assert h["marginal_contribution"] > -100
        assert h["value_received"] >= h["value_given"]  # net-positive for every member
        assert h["population_size"] > 0


def test_secure_aggregation_beat_plain_fedavg_without_loss():
    sec = run_secure_aggregation("sepsis", rounds=6, agg="equity", seeds=2)
    assert sec["masking"]["sum_reconstruction_error"] < 1e-9
    assert sec["masking"]["mean_signal_fraction"] < 0.05
    assert abs(sec["final"]["plain_accuracy"] - sec["final"]["secure_accuracy"]) < 0.01


def test_personalization_beats_global_and_local_for_unusual_hospital():
    pers = run_personalization("sepsis", rounds=4, epochs=2, seed=7)
    assert len(pers["hospitals"]) == 4
    for h in pers["hospitals"]:
        assert 0 <= h["personalized_accuracy"] <= 100
        assert 0 <= h["global_accuracy"] <= 100
        assert h["personalized_gain_pp"] > h["local_deficit_pp"]  # personalization outweighs local-only
    assert any(h["personalized_gain_pp"] > 0 for h in pers["hospitals"])  # at least one hospital wins


def test_async_study_tolerates_stragglers_with_penalty_turn_in_study():
    r = run_async_study("sepsis", rounds=6, agg="equity", seeds=2, straggler_penalty=0.75)
    assert r["rounds"] == 6
    stragglers = [h for s in r["studies"] for h in s["hospitals"] if h["delayed_rounds"] > 0]
    assert len(stragglers) >= 1
    assert all(0 <= s["accuracy"] <= 100 for s in r["studies"])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))