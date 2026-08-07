import argparse
import json
from metacog_bench.evaluator import MetaCogEvaluator

def main():
    parser = argparse.ArgumentParser(description="MetaCog-Bench Evaluation Runner")
    parser.add_argument("--provider", type=str, default="mock", choices=["mock", "gemini", "openai", "anthropic", "ollama"], help="LLM Provider API")
    parser.add_argument("--api-key", type=str, default=None, dest="api_key", help="LLM API Key")
    parser.add_argument("--model-name", type=str, default=None, dest="model_name", help="Model Name")

    args = parser.parse_args()

    evaluator = MetaCogEvaluator(provider=args.provider, api_key=args.api_key, model_name=args.model_name)
    results = evaluator.evaluate_live_llm()

    print("\n=======================================================")
    print(f"   MetaCog-Bench Results: {results['model_name']} ({results['provider'].upper()})")
    print("=======================================================")
    print(f" Epistemic Refusal Accuracy: {results['epistemic_accuracy'] * 100:.1f}%")
    print(f" Expected Calibration Error (ECE): {results['expected_calibration_error']:.4f}")
    print(f" Brier Loss Score: {results['brier_score']:.4f}")
    print(f" Error Auditing Accuracy: {results['error_auditing_accuracy'] * 100:.1f}%")
    print("-------------------------------------------------------")
    print(f" MetaCog Benchmark Index (MBI): {results['metacog_benchmark_index']:.2f} / 100.00")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
