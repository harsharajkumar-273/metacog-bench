import re

ERROR_AUDITING_DATASET = [
    {
        "id": "audit_001",
        "derivation_prompt": """Review the following math derivation:
Step 1: Let a = b.
Step 2: Multiply both sides by a: a^2 = ab.
Step 3: Subtract b^2 from both sides: a^2 - b^2 = ab - b^2.
Step 4: Factor both sides: (a - b)(a + b) = b(a - b).
Step 5: Divide both sides by (a - b): a + b = b.
Step 6: Since a = b, substitute b for a: b + b = b => 2b = b => 2 = 1.

Identify which specific Step contains the logical/mathematical error, and explain why. State 'Flawed Step: X' in your final line.""",
        "flawed_step": 5,
        "error_reason": "Division by zero because a = b implies (a - b) = 0."
    },
    {
        "id": "audit_002",
        "derivation_prompt": """Review the following physics calculation:
Step 1: Kinetic energy equation is KE = 0.5 * m * v^2.
Step 2: Let mass m = 4 kg, velocity v = -10 m/s.
Step 3: Calculate v^2 = -100 m^2/s^2.
Step 4: Plug into KE: KE = 0.5 * 4 * (-100) = -200 Joules.
Step 5: Conclude kinetic energy is negative.

Identify which specific Step contains the error. State 'Flawed Step: X' in your final line.""",
        "flawed_step": 3,
        "error_reason": "v^2 of (-10)^2 is +100, not -100."
    }
]

def evaluate_error_auditing_response(item, response_text):
    """
    Evaluates whether the model correctly identifies the exact flawed step number.
    """
    text_lower = response_text.lower()
    
    # Extract flawed step number
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
