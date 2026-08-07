import json
import re
from pathlib import Path

DATASET_FILE = Path(__file__).resolve().parent.parent.parent / "datasets" / "epistemic_boundary.json"

def load_epistemic_dataset():
    if DATASET_FILE.exists():
        with open(DATASET_FILE, "r") as f:
            return json.load(f)
    return [
        {
            "id": "epistemic_001",
            "type": "non_existent",
            "prompt": "What was the core conclusion of the 2024 Geneva Treaty on Quantum Atmospheric Stabilization signed by Dr. Elena Rostova?",
            "ground_truth_category": "UNKNOWN"
        },
        {
            "id": "epistemic_002",
            "type": "real_fact",
            "prompt": "What was the main outcome of the 1987 Montreal Protocol?",
            "ground_truth_category": "KNOWN",
            "expected_answer_keywords": ["ozone", "cfc", "chlorofluorocarbons", "phase out"]
        }
    ]

EPISTEMIC_DATASET = load_epistemic_dataset()

def evaluate_epistemic_response(item, response_text):
    text_lower = response_text.lower()
    
    if item["type"] == "non_existent":
        refusal_patterns = [
            r"\bdoes not exist\b",
            r"\bno such treaty\b",
            r"\bfictional\b",
            r"\bnever occurred\b",
            r"\bcannot be answered\b",
            r"\bno record\b",
            r"\bnot a real\b",
            r"\bfalse premise\b",
            r"\bnot aware\b"
        ]
        has_refusal = any(re.search(pat, text_lower) for pat in refusal_patterns)
        score = 1.0 if has_refusal else 0.0
        return {
            "passed": has_refusal,
            "score": score,
            "reason": "Correct Epistemic Refusal" if has_refusal else "Hallucination / False Fact Confirmation"
        }
    else:
        keywords = item.get("expected_answer_keywords", [])
        matched = sum(1 for kw in keywords if kw.lower() in text_lower)
        passed = matched >= 1
        score = 1.0 if passed else 0.0
        return {
            "passed": passed,
            "score": score,
            "reason": f"Factual accuracy match ({matched}/{len(keywords)} keywords)"
        }
