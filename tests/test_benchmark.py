import pytest
import numpy as np
from metacog_bench.evaluator import MetaCogEvaluator
from metacog_bench.tasks.epistemic_boundary import evaluate_epistemic_response, EPISTEMIC_DATASET
from metacog_bench.tasks.confidence_calibration import extract_confidence_and_correctness, calculate_expected_calibration_error, CALIBRATION_DATASET
from metacog_bench.tasks.error_auditing import evaluate_error_auditing_response, ERROR_AUDITING_DATASET

def test_epistemic_refusal():
    non_existent_item = EPISTEMIC_DATASET[0]
    refusal_response = "The 2024 Geneva Treaty on Quantum Atmospheric Stabilization does not exist."
    res = evaluate_epistemic_response(non_existent_item, refusal_response)
    assert res["passed"] is True
    assert res["score"] == 1.0

def test_epistemic_hallucination():
    non_existent_item = EPISTEMIC_DATASET[0]
    hallucinated_response = "The treaty was signed in Geneva by 50 nations."
    res = evaluate_epistemic_response(non_existent_item, hallucinated_response)
    assert res["passed"] is False
    assert res["score"] == 0.0

def test_confidence_calibration():
    calib_item = CALIBRATION_DATASET[0] # bat & ball trick question ($1.10 total, ball costs 5 cents)
    correct_response = "The ball costs 5 cents. Confidence: 0.95"
    res = extract_confidence_and_correctness(calib_item, correct_response)
    assert res["is_correct"] is True
    assert res["confidence"] == 0.95

def test_ece_calculation():
    mock_results = [
        {"confidence": 0.9, "is_correct": True},
        {"confidence": 0.8, "is_correct": False},
        {"confidence": 0.9, "is_correct": True}
    ]
    ece = calculate_expected_calibration_error(mock_results)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0

def test_error_auditing():
    audit_item = ERROR_AUDITING_DATASET[0] # Step 5 division by zero error
    correct_audit = "In Step 5 we divide by (a - b) which is zero. Flawed Step: 5"
    res = evaluate_error_auditing_response(audit_item, correct_audit)
    assert res["is_correct"] is True
    assert res["detected_step"] == 5

def test_evaluator_harness():
    evaluator = MetaCogEvaluator(provider="mock", model_name="Test-Model")
    mock_epistemic = ["does not exist", "ozone layer", "does not exist", "arthur ashkin", "does not exist"]
    mock_calib = ["5 cents. Confidence: 0.9", "5 minutes. Confidence: 0.8", "47 days. Confidence: 0.9", "no. Confidence: 0.9"]
    mock_audit = ["Flawed Step: 5", "Flawed Step: 3"]
    
    results = evaluator.evaluate_responses(mock_epistemic, mock_calib, mock_audit)
    assert "metacog_benchmark_index" in results
    assert results["metacog_benchmark_index"] > 0.0
