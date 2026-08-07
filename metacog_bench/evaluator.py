import json
import numpy as np
from metacog_bench.config import OUTPUT_DIR, METRIC_WEIGHTS
from metacog_bench.llm_client import LLMClientAdapter
from metacog_bench.tasks.epistemic_boundary import EPISTEMIC_DATASET, evaluate_epistemic_response
from metacog_bench.tasks.confidence_calibration import CALIBRATION_DATASET, extract_confidence_and_correctness, calculate_expected_calibration_error
from metacog_bench.tasks.error_auditing import ERROR_AUDITING_DATASET, evaluate_error_auditing_response

class MetaCogEvaluator:
    """
    Dynamic Evaluation Harness for MetaCog-Bench.
    Supports querying live LLMs (Gemini, OpenAI, Anthropic, Ollama) or pre-generated response strings.
    """
    def __init__(self, provider="mock", api_key=None, model_name=None):
        self.client = LLMClientAdapter(provider=provider, api_key=api_key, model_name=model_name)
        self.model_name = self.client.model_name

    def evaluate_live_llm(self):
        """
        Queries the configured LLM API across all benchmark task prompts in real time.
        """
        print(f"--> Querying live model '{self.model_name}' via provider '{self.client.provider}'...")
        
        # Query Task 1
        epistemic_responses = [self.client.generate_response(item["prompt"]) for item in EPISTEMIC_DATASET]
        
        # Query Task 2
        calibration_responses = [self.client.generate_response(item["prompt"]) for item in CALIBRATION_DATASET]
        
        # Query Task 3
        audit_responses = [self.client.generate_response(item["derivation_prompt"]) for item in ERROR_AUDITING_DATASET]
        
        return self.evaluate_responses(epistemic_responses, calibration_responses, audit_responses)

    def evaluate_responses(self, epistemic_responses, calibration_responses, audit_responses):
        # 1. Epistemic Boundary Evaluation
        epistemic_results = []
        epistemic_scores = []
        for item, resp in zip(EPISTEMIC_DATASET, epistemic_responses):
            res = evaluate_epistemic_response(item, resp)
            epistemic_scores.append(res["score"])
            epistemic_results.append(res)
        epistemic_acc = float(np.mean(epistemic_scores))

        # 2. Confidence Calibration Evaluation
        calib_results = []
        for item, resp in zip(CALIBRATION_DATASET, calibration_responses):
            res = extract_confidence_and_correctness(item, resp)
            calib_results.append(res)
        ece = calculate_expected_calibration_error(calib_results)
        avg_brier = float(np.mean([r["brier_score"] for r in calib_results]))

        # 3. Error Auditing Evaluation
        audit_scores = []
        for item, resp in zip(ERROR_AUDITING_DATASET, audit_responses):
            res = evaluate_error_auditing_response(item, resp)
            audit_scores.append(res["score"])
        audit_acc = float(np.mean(audit_scores))

        # MetaCog Benchmark Index (MBI)
        w = METRIC_WEIGHTS
        mbi_score = (
            w["epistemic_weight"] * epistemic_acc +
            w["ece_weight"] * (1.0 - ece) +
            w["error_audit_weight"] * audit_acc
        ) * 100.0

        results = {
            "model_name": self.model_name,
            "provider": self.client.provider,
            "epistemic_accuracy": epistemic_acc,
            "expected_calibration_error": ece,
            "brier_score": avg_brier,
            "error_auditing_accuracy": audit_acc,
            "metacog_benchmark_index": mbi_score
        }

        # Save to disk
        out_file = OUTPUT_DIR / f"{self.model_name.replace(' ', '_').replace('/', '_').lower()}_results.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

        return results
