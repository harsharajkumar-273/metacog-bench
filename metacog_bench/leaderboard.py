import json
from datetime import datetime
from pathlib import Path

LEADERBOARD_FILE = Path(__file__).resolve().parent.parent / "datasets" / "public_leaderboard.json"

def get_public_leaderboard():
    """Reads all public leaderboard submissions, sorted by MetaCog Benchmark Index (MBI)."""
    if LEADERBOARD_FILE.exists():
        try:
            with open(LEADERBOARD_FILE, "r") as f:
                data = json.load(f)
                return sorted(data, key=lambda x: x.get("metacog_benchmark_index", 0.0), reverse=True)
        except Exception:
            pass
    return []

def publish_to_public_leaderboard(result_dict):
    """
    Publishes a verified model evaluation result to the public leaderboard registry.
    """
    data = get_public_leaderboard()
    
    # Add metadata
    entry = dict(result_dict)
    entry["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    entry["visibility"] = "public"

    # Deduplicate by model_name if existing
    updated_data = [d for d in data if d.get("model_name") != entry["model_name"]]
    updated_data.append(entry)

    # Save to disk
    LEADERBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(updated_data, f, indent=2)

    return updated_data
