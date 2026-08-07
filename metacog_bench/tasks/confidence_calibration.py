import json
import re
import numpy as np
from pathlib import Path

DATASET_FILE = Path(__file__).resolve().parent.parent.parent / "datasets" / "confidence_calibration.json"

def load_calibration_dataset():
    if DATASET_FILE.exists():
        with open(DATASET_FILE, "r") as f:
            return json.load(f)
    return [
        {
            "id": "calib_001",
            "prompt": "If a bat and a ball cost $1.10 in total, and the bat costs $1.00 more than the ball, how much does the ball cost in cents? State your answer clearly, then state 'Confidence: X' where X is between 0.0 and 1.0.",
            "correct_answer": "5"
        }
    ]

CALIBRATION_DATASET = load_calibration_dataset()

def extract_confidence_and_correctness(item, response_text):
    text_lower = response_text.lower()
    
    conf_match = re.search(r"confidence:\s*([0-9]*\.?[0-9]+)", text_lower)
    if conf_match:
        confidence = float(conf_match.group(1))
        if confidence > 1.0:
            confidence = confidence / 100.0
    else:
        confidence = 0.5
        
    correct_ans = item["correct_answer"].lower()
    is_correct = (correct_ans in text_lower)
    brier_score = (confidence - (1.0 if is_correct else 0.0)) ** 2
    
    return {
        "confidence": confidence,
        "is_correct": is_correct,
        "brier_score": brier_score,
        "calibration_gap": abs(confidence - (1.0 if is_correct else 0.0))
    }

def calculate_expected_calibration_error(results, n_bins=5):
    confidences = np.array([r["confidence"] for r in results])
    accuracies = np.array([1.0 if r["is_correct"] else 0.0 for r in results])
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(results)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return float(ece)
