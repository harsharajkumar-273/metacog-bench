import json
import numpy as np
from metacog_bench.config import OUTPUT_DIR, METRIC_WEIGHTS
from metacog_bench.tasks.epistemic_boundary import EPISTEMIC_DATASET, evaluate_epistemic_response
from metacog_bench.tasks.confidence_calibration import CALIBRATION_DATASET, extract_confidence_and_correctness, calculate_expected_calibration_error
from metacog_bench.tasks.error_auditing import ERROR_AUDITING_DATASET, evaluate_error_auditing_response

class MetaCogEvaluator:
    """
    Main Evaluation Harness for MetaCog-Bench.
    Evaluates candidate models across Epistemic Boundary, Calibration (ECE), and Error Auditing.
    """
    def __init__(self, model_name="Candidate-Model"):
        self.model_name = model_name

    def evaluate_responses(self, epistemic_responses, calibration_responses, audit_responses):
        # 1. Epistemic Boundary Evaluation
        epistemic_results = []
        epistemic_scores = []
        for item, resp in zip(EPISTEMIC_DATASET, epistemic_responses):
            res = evaluate_epistemic_response(item, resp)
            epistemic_scores.append(res["score"])
            epistemic_results.append(res)
        epistemic_acc = np.mean(epistemic_scores)

        # 2. Confidence Calibration Evaluation
        calib_results = []
        for item, resp in zip(CALIBRATION_DATASET, calibration_responses):
            res = extract_confidence_and_correctness(item, resp)
            calib_results.append(res)
        ece = calculate_expected_calibration_error(calib_results)
        avg_brier = np.mean([r["brier_score"] for r in calib_results])

        # 3. Error Auditing Evaluation
        audit_scores = []
        for item, resp in zip(ERROR_AUDITING_DATASET, audit_responses):
            res = evaluate_error_auditing_response(item, resp)
            audit_scores.append(res["score"])
        audit_acc = np.mean(audit_scores)

        # MetaCog Benchmark Index (MBI)
        w = METRIC_WEIGHTS
        mbi_score = (
            w["epistemic_weight"] * epistemic_acc +
            w["ece_weight"] * (1.0 - ece) +
            w["error_audit_weight"] * audit_acc
        )

        results = {
            "model_name": self.model_name,
            "epistemic_accuracy": epistemic_acc,
            "expected_calibration_error": ece,
            "brier_score": avg_brier,
            "error_auditing_accuracy": audit_acc,
            "metacog_benchmark_index": mbi_score * 100.0
        }

        # Save to disk
        out_file = OUTPUT_DIR / f"{self.model_name.replace(' ', '_').lower()}_results.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

        return results
