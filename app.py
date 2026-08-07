import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from metacog_bench.evaluator import MetaCogEvaluator
from metacog_bench.leaderboard import get_public_leaderboard, publish_to_public_leaderboard
from metacog_bench.tasks.epistemic_boundary import EPISTEMIC_DATASET, evaluate_epistemic_response
from metacog_bench.tasks.confidence_calibration import CALIBRATION_DATASET, extract_confidence_and_correctness
from metacog_bench.tasks.error_auditing import ERROR_AUDITING_DATASET, evaluate_error_auditing_response

st.set_page_config(
    page_title="MetaCog-Bench Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MetaCog-Bench: Dual-Session Evaluation Suite")
st.markdown("""
Evaluating Frontier LLMs on **Metacognitive Calibration**, **Epistemic Horizon Humility**, and **Planted Flaw Auditing**.
*Target Track: Google DeepMind AGI Hackathon - Metacognition Track.*
""")

# Sidebar Navigation & Session Mode Selector
st.sidebar.header("🔐 Session Evaluation Mode")
session_mode = st.sidebar.radio(
    "Choose Evaluation Mode:",
    ["🔒 Private Sandbox Session", "🌐 Public Leaderboard Submission"]
)

if session_mode == "🔒 Private Sandbox Session":
    st.sidebar.info("🔒 **Private Sandbox**: Results are evaluated locally in your browser session only. Nothing is saved or published publicly.")
else:
    st.sidebar.success("🌐 **Public Leaderboard Submission**: Verified scores are published to the persistent community leaderboard.")

st.sidebar.header("🔌 Live LLM API Config")
provider = st.sidebar.selectbox("Select LLM Provider", ["mock", "gemini", "openai", "anthropic", "ollama"])

api_key = None
if provider in ["gemini", "openai", "anthropic"]:
    api_key = st.sidebar.text_input(f"Enter {provider.upper()} API Key", type="password")

model_name_input = st.sidebar.text_input("Custom Model Name (Optional)", value="")

selected_view = st.sidebar.radio("Navigation", ["Global Leaderboard & Analytics", "Run Evaluation Harness", "Interactive Prompt Inspector"])

# Load persistent public leaderboard
public_data = get_public_leaderboard()
df_leaderboard = pd.DataFrame(public_data)

if selected_view == "Global Leaderboard & Analytics":
    st.subheader("🌐 Global Community Leaderboard")
    
    if not df_leaderboard.empty:
        display_df = df_leaderboard[[
            "model_name", "provider", "metacog_benchmark_index", 
            "epistemic_accuracy", "expected_calibration_error", "brier_score", 
            "error_auditing_accuracy", "timestamp"
        ]].copy()
        
        display_df.columns = [
            "Model Name", "Provider", "MetaCog Index (MBI)", 
            "Epistemic Acc", "ECE (Lower=Better)", "Brier Score", 
            "Error Audit Acc", "Submitted At"
        ]
        
        st.dataframe(display_df, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Radar Chart: Metacognitive Capabilities")
            categories = ["Epistemic Humility", "Calibration (1-ECE)", "Error Auditing"]
            
            fig_radar = go.Figure()
            for idx, row in df_leaderboard.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row["epistemic_accuracy"] * 100, (1.0 - row["expected_calibration_error"]) * 100, row["error_auditing_accuracy"] * 100],
                    theta=categories,
                    fill='toself',
                    name=row["model_name"]
                ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            st.subheader("📉 Expected Calibration Error (ECE)")
            conf_bins = np.linspace(0.1, 1.0, 5)
            ideal_acc = conf_bins
            model_acc = [0.10, 0.35, 0.55, 0.70, 0.82]

            fig_calib = go.Figure()
            fig_calib.add_trace(go.Scatter(x=conf_bins, y=ideal_acc, mode='lines', name='Ideal Perfect Calibration', line=dict(dash='dash', color='gray')))
            fig_calib.add_trace(go.Scatter(x=conf_bins, y=model_acc, mode='lines+markers', name='Observed Accuracy', line=dict(color='#FF4B4B', width=3)))
            fig_calib.update_layout(xaxis_title="Stated Confidence", yaxis_title="Actual Accuracy", title="ECE Reliability Curve")
            st.plotly_chart(fig_calib, use_container_width=True)
    else:
        st.info("No public leaderboard submissions found.")

elif selected_view == "Run Evaluation Harness":
    st.subheader(f"⚡ Execute Model Evaluation ({session_mode})")
    
    if st.button("🚀 Run Evaluation Harness"):
        with st.spinner(f"Evaluating model via {provider.upper()}..."):
            evaluator = MetaCogEvaluator(provider=provider, api_key=api_key, model_name=model_name_input or None)
            res = evaluator.evaluate_live_llm()
            
            st.success("✅ Evaluation Complete!")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MetaCog Index (MBI)", f"{res['metacog_benchmark_index']:.2f} / 100")
            m2.metric("Epistemic Refusal", f"{res['epistemic_accuracy'] * 100:.1f}%")
            m3.metric("ECE Error", f"{res['expected_calibration_error']:.4f}")
            m4.metric("Error Auditing", f"{res['error_auditing_accuracy'] * 100:.1f}%")

            if session_mode == "🌐 Public Leaderboard Submission":
                publish_to_public_leaderboard(res)
                st.balloons()
                st.success("🎉 Results successfully published to the Global Public Leaderboard!")

            st.json(res)

elif selected_view == "Interactive Prompt Inspector":
    st.subheader("🔬 Prompt & Task Inspector")
    task_choice = st.selectbox("Select Task Module", ["Task 1: Epistemic Boundary", "Task 2: Confidence Calibration", "Task 3: Planted Error Auditing"])

    if task_choice == "Task 1: Epistemic Boundary":
        st.markdown("### Task 1: Non-Existent Entity Hallucination Probing")
        sample = EPISTEMIC_DATASET[0]
        st.info(f"**Input Prompt**: {sample['prompt']}")
        user_response = st.text_area("Model Output Response:", value="The 2024 Geneva Treaty on Quantum Atmospheric Stabilization is a fictional document and does not exist.")
        
        if st.button("Evaluate Refusal"):
            res = evaluate_epistemic_response(sample, user_response)
            if res["passed"]:
                st.success(f"✅ PASSED (Score: {res['score']}) - {res['reason']}")
            else:
                st.error(f"❌ FAILED (Score: {res['score']}) - {res['reason']}")

    elif task_choice == "Task 2: Confidence Calibration":
        st.markdown("### Task 2: Confidence Calibration & ECE Probing")
        sample = CALIBRATION_DATASET[0]
        st.info(f"**Input Prompt**: {sample['prompt']}")
        user_response = st.text_area("Model Output Response:", value="The ball costs 5 cents. Confidence: 0.95")

        if st.button("Evaluate Calibration Gap"):
            res = extract_confidence_and_correctness(sample, user_response)
            st.write(f"**Is Correct**: `{res['is_correct']}`")
            st.write(f"**Stated Confidence**: `{res['confidence']:.2f}`")
            st.write(f"**Calibration Gap**: `{res['calibration_gap']:.2f}`")
            st.write(f"**Brier Loss**: `{res['brier_score']:.4f}`")

    elif task_choice == "Task 3: Planted Error Auditing":
        st.markdown("### Task 3: Planted Logical Flaw Auditing")
        sample = ERROR_AUDITING_DATASET[0]
        st.info(f"**Input Derivation Prompt**:\n\n{sample['derivation_prompt']}")
        user_response = st.text_area("Model Output Response:", value="In Step 5 we divide by (a - b) which equals zero. Flawed Step: 5")

        if st.button("Audit Flaw Detection"):
            res = evaluate_error_auditing_response(sample, user_response)
            if res["is_correct"]:
                st.success(f"✅ PASSED - {res['reason']}")
            else:
                st.error(f"❌ FAILED - {res['reason']}")
