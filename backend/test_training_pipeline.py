"""
Federo Health — Training pipeline tests. Assert real computed invariants on the
cleaning/selection/analysis chain (no mocks).
Run:  python -m pytest backend/test_training_pipeline.py
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import clean_and_validate_dataset
from trainer import build_model_card, train_model

MESSY_CSV = """Patient_ID,Age,HR,Temp_C,Systolic_BP,Gender,Sepsis_Label
P1,68,124,39.2,82,Male,sepsis
P2,42,80,37.0,124,F,0
P2,42,80,37.0,124,F,0
P3,78,135,39.8,75,m,1
P4,55,,38.5,86,Female,Sepsis
P5,29,72,36.6,116,M,0
P6,60,110bpm,38.1,90,Male,1
P7,50,95,37.4,120,F,0
"""


def test_single_class_target_raises_with_clear_message():
    df = pd.DataFrame({"age": [1, 2, 3, 4], "label": [0, 0, 0, 0]})
    with pytest.raises(ValueError, match="only one class"):
        train_model(df, ["age"], "label", "sepsis", {"id": "h1"}, [])


def test_cleanup_issues_and_analysis_round_trip_through_model_card():
    r = clean_and_validate_dataset(MESSY_CSV, use_case="sepsis")
    issue_types = {i["type"] for i in r["issues"]}
    assert {"duplicates", "missing_values", "text_coercion", "categorical_encoding"} <= issue_types

    res = train_model(
        r["df"], r["numeric_feature_names"], r["target_col"], "sepsis", {"id": "h1"},
        [], categorical_feature_names=r["categorical_feature_names"],
        preprocessing_issues=r["issues"], image_input=r["image_input"],
    )
    card = build_model_card(res, r, {"id": "h1", "name": "H1"}, [])

    ms = card["analysis"]["model_selection"]
    pp = card["analysis"]["preprocessing"]
    assert ms["selected_algorithm"].startswith("Clinical Logistic Regression")
    assert len(ms["candidates"]) >= 2 and isinstance(ms["decision_factors"], list)
    assert pp["resolved_issues"] and pp["input_profile"]["records_used"] == 7


def test_selection_scales_to_ensemble_for_large_cohorts():
    df = pd.DataFrame({
        "f1": np.random.rand(3200) + 3,
        "f2": np.random.rand(3200) * 10,
        "label": np.random.binomial(1, 0.4, 3200),
    })
    res = train_model(df, ["f1", "f2"], "label", "sepsis", {"id": "h1"}, [])
    assert res["analysis"]["model_selection"]["selected_algorithm"].startswith("Gradient Boosting")


def test_image_hint_routes_to_cnn_selection():
    img_csv = """ID,Fovea,ImagePath,DR_Label
I1,0.3,fundus/i1.jpg,1
I2,0.1,scan/i2.png,0
I3,0.8,photo/i3.jpg,1
I4,0.2,fundus/i4.dcm,0
"""
    r = clean_and_validate_dataset(img_csv, use_case="retinopathy")
    assert r["image_input"] is True
    res = train_model(
        r["df"], r["numeric_feature_names"], r["target_col"], "retinopathy",
        {"id": "h1"}, [], categorical_feature_names=r["categorical_feature_names"],
        preprocessing_issues=r["issues"], image_input=r["image_input"],
    )
    sel = res["analysis"]["model_selection"]["selected_algorithm"]
    assert "CNN" in sel and "scratch" not in sel  # transfer-learned CNN, no random init


def test_categorical_only_dataset_trains_and_zero_features_guard():
    df = pd.DataFrame({
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
        "ward": ["icu", "icu", "gen", "gen", "icu", "gen", "gen", "icu"],
        "label": [1, 0, 1, 0, 1, 0, 1, 0],
    })
    res = train_model(df, [], "label", "sepsis", {"id": "h1"}, [],
                      categorical_feature_names=["gender", "ward"])
    assert res["analysis"]["preprocessing"]["input_profile"]["features_after_encoding"] == 4

    with pytest.raises(ValueError, match="No usable feature columns"):
        train_model(pd.DataFrame({"label": [1, 0, 1, 0]}), [], "label", "sepsis", {"id": "h1"}, [])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))