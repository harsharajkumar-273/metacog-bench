import gradio as gr
import pandas as pd
import numpy as np
from metacog_bench.evaluator import MetaCogEvaluator
from metacog_bench.leaderboard import get_public_leaderboard, publish_to_public_leaderboard
from metacog_bench.tasks.epistemic_boundary import EPISTEMIC_DATASET, evaluate_epistemic_response
from metacog_bench.tasks.confidence_calibration import CALIBRATION_DATASET, extract_confidence_and_correctness
from metacog_bench.tasks.error_auditing import ERROR_AUDITING_DATASET, evaluate_error_auditing_response

def evaluate_custom_response(task_name, user_response):
    if task_name == "Task 1: Epistemic Boundary":
        sample = EPISTEMIC_DATASET[0]
        res = evaluate_epistemic_response(sample, user_response)
        return f"Passed: {res['passed']} | Score: {res['score']} | Reason: {res['reason']}"
    elif task_name == "Task 2: Confidence Calibration":
        sample = CALIBRATION_DATASET[0]
        res = extract_confidence_and_correctness(sample, user_response)
        return f"Is Correct: {res['is_correct']} | Confidence: {res['confidence']:.2f} | Calibration Gap: {res['calibration_gap']:.2f} | Brier Loss: {res['brier_score']:.4f}"
    else:
        sample = ERROR_AUDITING_DATASET[0]
        res = evaluate_error_auditing_response(sample, user_response)
        return f"Correct Step Identified: {res['is_correct']} | Details: {res['reason']}"

def run_live_benchmark(provider, api_key, model_name, publish_public):
    evaluator = MetaCogEvaluator(provider=provider, api_key=api_key if api_key else None, model_name=model_name if model_name else None)
    res = evaluator.evaluate_live_llm()
    
    if publish_public:
        publish_to_public_leaderboard(res)

    summary_df = pd.DataFrame([{
        "Model Name": res["model_name"],
        "Provider": res["provider"].upper(),
        "MetaCog Index (MBI)": f"{res['metacog_benchmark_index']:.2f} / 100",
        "Epistemic Refusal %": f"{res['epistemic_accuracy'] * 100:.1f}%",
        "ECE Error": f"{res['expected_calibration_error']:.4f}",
        "Brier Loss": f"{res['brier_score']:.4f}",
        "Error Auditing %": f"{res['error_auditing_accuracy'] * 100:.1f}%"
    }])
    return summary_df

def get_leaderboard_table():
    board = get_public_leaderboard()
    if not board:
        return pd.DataFrame()
    return pd.DataFrame(board)[[
        "model_name", "provider", "metacog_benchmark_index", "epistemic_accuracy", "expected_calibration_error", "error_auditing_accuracy"
    ]]

with gr.Blocks(title="MetaCog-Bench Gradio UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 MetaCog-Bench: Metacognitive AI Evaluation Suite")
    gr.Markdown("Evaluating Frontier LLMs on **Metacognitive Calibration**, **Epistemic Humility**, and **Planted Flaw Auditing**.")

    with gr.Tab("🏆 Global Community Leaderboard"):
        leaderboard_df = gr.DataFrame(value=get_leaderboard_table, headers=["Model Name", "Provider", "MetaCog Index", "Epistemic Acc", "ECE Error", "Error Audit Acc"])
        refresh_btn = gr.Button("🔄 Refresh Leaderboard")
        refresh_btn.click(get_leaderboard_table, outputs=leaderboard_df)

    with gr.Tab("⚡ Live Model Evaluator"):
        gr.Markdown("### Query Live LLM Provider APIs")
        provider_input = gr.Dropdown(choices=["mock", "gemini", "openai", "anthropic", "ollama"], value="mock", label="Select Provider")
        api_key_input = gr.Textbox(type="password", label="API Key (Optional for mock/ollama)")
        model_name_input = gr.Textbox(label="Custom Model Name (Optional)")
        publish_chk = gr.Checkbox(label="Publish to Global Public Leaderboard", value=False)
        
        eval_btn = gr.Button("🚀 Execute Benchmark Evaluation")
        output_table = gr.DataFrame(label="Benchmark Metrics Output")
        
        eval_btn.click(run_live_benchmark, inputs=[provider_input, api_key_input, model_name_input, publish_chk], outputs=output_table)

    with gr.Tab("🔬 Prompt Inspector"):
        task_dropdown = gr.Dropdown(choices=["Task 1: Epistemic Boundary", "Task 2: Confidence Calibration", "Task 3: Planted Error Auditing"], value="Task 1: Epistemic Boundary", label="Task Module")
        prompt_output = gr.Textbox(value="The 2024 Geneva Treaty on Quantum Atmospheric Stabilization is a fictional document and does not exist.", label="Model Output Response")
        inspect_btn = gr.Button("Audit Response")
        inspect_result = gr.Textbox(label="Evaluation Result")
        
        inspect_btn.click(evaluate_custom_response, inputs=[task_dropdown, prompt_output], outputs=inspect_result)

if __name__ == "__main__":
    demo.launch()
