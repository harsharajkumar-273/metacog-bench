import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
TASKS_DIR = BASE_DIR / "tasks"
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "benchmark_output"

# Create directories
TASKS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Competition Track Metadata
COMPETITION_NAME = "Google DeepMind - Measuring Progress Toward AGI: Cognitive Abilities"
BENCHMARK_NAME = "MetaCog-Bench"
TRACK = "Metacognition"

# Target Evaluation Capabilities
CAPABILITIES = [
    "Epistemic Boundary Probing & Hallucination Resistance",
    "Confidence Calibration & Expected Calibration Error (ECE)",
    "Planted Error Auditing & Flaw Detection"
]
