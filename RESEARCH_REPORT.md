# OfficeQA Grounded Reasoning Benchmark - Research Report

## 1. Benchmark Overview: OfficeQA
OfficeQA is a grounded reasoning dataset developed by **Databricks** to evaluate AI agents on complex, real-world financial tasks. It proxies economically valuable enterprise tasks by requiring agents to reason over a massive corpus of **U.S. Treasury Bulletins (1939–2025)**.

### Key Characteristics:
*   **Corpus:** ~89,000 pages of dense text, complex financial tables, and charts.
*   **Task Type:** Multi-step reasoning, calculations (e.g., inflation adjustments, regression, time-series forecasting), and table/chart interpretation.
*   **Performance Gap:** Frontier models without the corpus score ~2%; with the corpus, they score <45% accuracy, indicating a high difficulty level for current agents.
*   **Subsets:**
    *   **OfficeQA Pro (N=133):** Default for frontier models.
    *   **OfficeQA Full (N=246):** Includes easier questions for "hillclimbing" systems.

## 2. Sentient Arena & AgentBeats
**Sentient Arena** (AgentBeats) is the evaluation framework used to benchmark these agents.
*   **Purple Agents:** The participant agents being evaluated (our task).
*   **Green Agents:** Judge agents that evaluate the participant's reasoning and answer.
*   **Evaluation Metrics:**
    *   **Numeric Accuracy:** Correctness of the final numeric answer (within tolerance, e.g., 1%).
    *   **Strict Grounding:** Requires the answer to be correct AND backed by verified evidence from the corpus.

## 3. Current Project Structure (sentient-arena-officeqa)
The provided starting kit includes a multi-stage pipeline for processing OfficeQA questions.

### Core Components:
*   **`src/pipeline.py`:** Orchestrates the multi-stage reasoning process.
*   **Stages (`src/stages/`):**
    1.  **Parse:** Identify required source files and entities from the question.
    2.  **Retrieve:** Select relevant evidence (rows/chunks) from the transformed corpus.
    3.  **Extract:** Pull specific numeric values from the retrieved evidence.
    4.  **Calculate:** Perform arithmetic or logic to derive the final answer.
    5.  **Verify:** Check if the result is supported by the evidence (grounding).
    6.  **Finalize:** Format the final answer for submission.
*   **Backends:** Support for `local_stub` and `OpenHands SDK`.

### Evaluation & Iteration Tools:
*   **`scripts/run_eval.py`:** Main entry point for running the evaluation.
*   **`scripts/gate_3q.py`:** A "3-question gate" script that runs a small sample and suggests "top next actions" based on trace reports.
*   **Trace Reports:** Detailed analysis of failures at each stage to guide improvements.

## 4. Skills & Evolution (Challenge 0 Goals)
The goal is to develop "skills" for an agent harness like OpenHands and explore self-improvement tools.

### Potential Skills to Develop:
*   **Table Parsing Skill:** Better handling of Markdown tables and cross-row/column lookups.
*   **Unit Normalization Skill:** Automatically handling billions, millions, and percentages.
*   **Arithmetic Precision Skill:** Using code execution (Python) for complex formulas instead of relying on LLM arithmetic.
*   **Multi-Step Retrieval Skill:** Iterative search if the first set of documents is insufficient.

### AI Self-Improvement Tools (to explore):
*   **Evolver/EvoSkill:** Automated optimization of agent prompts or strategies.
*   **SkillNet:** A repository or network of specialized skills that can be retrieved and applied.
*   **GEPA/GEPA+:** (Generalized Evolution for Participant Agents) Framework for evolving agent behaviors.
*   **optimize_anything:** General-purpose optimization tool for AI pipelines.

## 5. Next Steps
1.  **Baseline Run:** Run the current pipeline on a few questions (`scripts/gate_3q.py`) to see the current performance and failure modes.
2.  **Analyze Failures:** Use `trace_report.md` to identify the weakest stages (e.g., retrieval failure vs. calculation error).
3.  **Develop Skills:** Implement targeted improvements in the `src/stages/` based on the analysis.
4.  **Automated Evolution:** Integrate or implement an evolution loop (e.g., using `gate_3q.py` outputs to update prompts).
