import json
import re
from pathlib import Path

DATASET_FILE = Path(__file__).resolve().parent.parent.parent / "datasets" / "error_auditing.json"

def load_error_auditing_dataset():
    if DATASET_FILE.exists():
        with open(DATASET_FILE, "r") as f:
            return json.load(f)
    return [
        {
            "id": "audit_001",
            "derivation_prompt": "Step 1: Let a = b.\nStep 2: a^2 = ab.\nStep 3: a^2 - b^2 = ab - b^2.\nStep 4: (a - b)(a + b) = b(a - b).\nStep 5: a + b = b.\nFlawed Step: 5",
            "flawed_step": 5
        }
    ]

ERROR_AUDITING_DATASET = load_error_auditing_dataset()

def evaluate_error_auditing_response(item, response_text):
    text_lower = response_text.lower()
    
    step_match = re.search(r"flawed step:\s*([0-9]+)", text_lower)
    if not step_match:
        step_match = re.search(r"step\s*([0-9]+)\s*(is flawed|contains the error|is incorrect)", text_lower)
        
    detected_step = int(step_match.group(1)) if step_match else None
    ground_truth_step = item["flawed_step"]
    
    is_correct = (detected_step == ground_truth_step)
    score = 1.0 if is_correct else 0.0
    
    return {
        "detected_step": detected_step,
        "ground_truth_step": ground_truth_step,
        "is_correct": is_correct,
        "score": score,
        "reason": f"Correctly identified Step {ground_truth_step}" if is_correct else f"Failed: Detected Step {detected_step} instead of Step {ground_truth_step}"
    }
