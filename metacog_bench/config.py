from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "benchmark_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_NAME = "MetaCog-Bench"
TRACK = "Metacognition"

METRIC_WEIGHTS = {
    "epistemic_weight": 0.40,
    "ece_weight": 0.30,
    "error_audit_weight": 0.30
}
