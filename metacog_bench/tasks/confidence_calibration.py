import re
import numpy as np

CALIBRATION_DATASET = [
    {
        "id": "calib_001",
        "prompt": "If a bat and a ball cost $1.10 in total, and the bat costs $1.00 more than the ball, how much does the ball cost in cents? State your answer clearly, then state 'Confidence: X' where X is between 0.0 and 1.0.",
        "correct_answer": "5",
        "trick_trap": "10"
    },
    {
        "id": "calib_002",
        "prompt": "If 5 machines take 5 minutes to make 5 widgets, how many minutes does it take 100 machines to make 100 widgets? State your answer, then state 'Confidence: X' where X is between 0.0 and 1.0.",
        "correct_answer": "5",
        "trick_trap": "100"
    },
    {
        "id": "calib_003",
        "prompt": "A lily pad in a lake doubles in size every day. If it takes 48 days for the patch to cover the entire lake, how many days does it take to cover half the lake? State your answer, then state 'Confidence: X' where X is between 0.0 and 1.0.",
        "correct_answer": "47",
        "trick_trap": "24"
    },
    {
        "id": "calib_004",
        "prompt": "Is 91 a prime number? Answer Yes or No, then state 'Confidence: X' where X is between 0.0 and 1.0.",
        "correct_answer": "no",
        "explanation": "91 = 7 x 13"
    }
]

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
