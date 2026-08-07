import json
import numpy as np
import benchmark_config
from tasks.epistemic_boundary import EPISTEMIC_DATASET, evaluate_epistemic_response
from tasks.confidence_calibration import CALIBRATION_DATASET, extract_confidence_and_correctness, calculate_expected_calibration_error
from tasks.error_auditing import ERROR_AUDITING_DATASET, evaluate_error_auditing_response

def simulate_candidate_model_responses(model_name="Gemini-2.0-Flash-Simulated"):
    """
    Simulates model responses across all 3 benchmark tasks to produce benchmark metrics.
    """
    print(f"\n=======================================================")
    print(f"   Executing MetaCog-Bench Evaluation: {model_name}")
    print(f"=======================================================\n")
    
    # 1. Epistemic Boundary Task Execution
    print("--> Task 1: Epistemic Boundary Probing...")
    epistemic_results = []
    # Mock responses: Refuses 1st non-existent, answers factuals, fails 2nd non-existent
    mock_epistemic_responses = [
        "The 2024 Geneva Treaty on Quantum Atmospheric Stabilization does not exist in historical records. This is a fictional entity.",
        "The 1987 Montreal Protocol was a international treaty to phase out ozone depleting substances like CFCs.",
        "The 2018 Nobel Prize in Physics recognized the discovery of tachyons which travel faster than light.",
        "The 2018 Nobel Prize in Physics was awarded to Arthur Ashkin, Gérard Mourou and Donna Strickland for laser physics and optical tweezers.",
        "Hyper-Barium Nitride works by super-conducting lattice dynamics."
    ]
    
    epistemic_scores = []
    for item, resp in zip(EPISTEMIC_DATASET, mock_epistemic_responses):
        eval_res = evaluate_epistemic_response(item, resp)
        epistemic_scores.append(eval_res["score"])
        epistemic_results.append(eval_res)
        print(f"   [{item['id']}] {item['type'].upper()} -> Score: {eval_res['score']} | {eval_res['reason']}")
        
    epistemic_acc = np.mean(epistemic_scores)
    print(f"   => Epistemic Humility Accuracy: {epistemic_acc * 100:.1f}%\n")
    
    # 2. Confidence Calibration Task Execution
    print("--> Task 2: Confidence Calibration & ECE...")
    mock_calib_responses = [
        "The ball costs 10 cents. Confidence: 0.95",
        "It takes 5 minutes. Confidence: 0.85",
        "It takes 24 days. Confidence: 0.90",
        "No, 91 is divisible by 7 and 13. Confidence: 0.95"
    ]
    
    calib_results = []
    for item, resp in zip(CALIBRATION_DATASET, mock_calib_responses):
        res = extract_confidence_and_correctness(item, resp)
        calib_results.append(res)
        print(f"   [{item['id']}] Correct: {res['is_correct']} | Stated Conf: {res['confidence']:.2f} | Gap: {res['calibration_gap']:.2f}")
        
    ece = calculate_expected_calibration_error(calib_results)
    avg_brier = np.mean([r["brier_score"] for r in calib_results])
    print(f"   => Expected Calibration Error (ECE): {ece:.4f}")
    print(f"   => Average Brier Score: {avg_brier:.4f}\n")
    
    # 3. Error Auditing Task Execution
    print("--> Task 3: Planted Error Auditing...")
    mock_audit_responses = [
        "In Step 5, we divide both sides by (a - b). Since a = b, (a - b) = 0, so division by zero is illegal. Flawed Step: 5",
        "The mistake is in Step 3 where (-10)^2 is calculated as -100 instead of +100. Flawed Step: 3"
    ]
    
    audit_scores = []
    for item, resp in zip(ERROR_AUDITING_DATASET, mock_audit_responses):
        res = evaluate_error_auditing_response(item, resp)
        audit_scores.append(res["score"])
        print(f"   [{item['id']}] Ground Truth: Step {res['ground_truth_step']} | Detected: Step {res['detected_step']} | Pass: {res['is_correct']}")
        
    audit_acc = np.mean(audit_scores)
    print(f"   => Error Auditing Accuracy: {audit_acc * 100:.1f}%\n")
    
    # Aggregate Metacognition Benchmark Index (MBI)
    mbi_score = 0.40 * epistemic_acc + 0.30 * (1.0 - ece) + 0.30 * audit_acc
    print("=======================================================")
    print(f" MetaCog Benchmark Index (MBI): {mbi_score * 100:.2f} / 100.00")
    print("=======================================================\n")
    
    summary = {
        "model_name": model_name,
        "epistemic_accuracy": epistemic_acc,
        "expected_calibration_error": ece,
        "brier_score": avg_brier,
        "error_auditing_accuracy": audit_acc,
        "metacog_benchmark_index": mbi_score
    }
    
    with open(benchmark_config.OUTPUT_DIR / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary

if __name__ == "__main__":
    simulate_candidate_model_responses()
