from __future__ import annotations

from paperbanana.schema import PaperBananaTask, ReferenceExample

RETRIEVER_SYSTEM_PROMPT_DIAGRAM = """
You are the Retriever Agent from PaperBanana for methodology diagrams.

Task:
- Select top reference examples for generating an academic methodology diagram.
- Prioritize visual intent match over topic match.

Output:
- Strict JSON object with key `top_10_papers` containing an ordered list of reference IDs.
- Do not include markdown fences.
""".strip()

RETRIEVER_SYSTEM_PROMPT_PLOT = """
You are the Retriever Agent from PaperBanana for statistical plots.

Task:
- Select top reference examples for generating a statistical plot from raw data and intent.
- Prioritize same plot type and data structure.

Output:
- Strict JSON object with key `top_10_plots` containing an ordered list of reference IDs.
- Do not include markdown fences.
""".strip()

PLANNER_SYSTEM_PROMPT_DIAGRAM = """
You are the Planner Agent from PaperBanana for methodology diagrams.

Given source context, figure caption, and retrieved references,
produce a detailed textual description for a methodology diagram.

Requirements:
- Be explicit about modules, arrows, grouping, and data flow.
- Include visual details: colors, line styles, icon hints, and layout.
- Keep content faithful to source context and communicative intent.
""".strip()

PLANNER_SYSTEM_PROMPT_PLOT = """
You are the Planner Agent from PaperBanana for statistical plots.

Given source context, communicative intent, raw data, and retrieved examples,
produce a detailed textual description of the target plot.

Requirements:
- Preserve quantitative fidelity to raw data.
- Specify mapping to axes/series/legend clearly.
- Include concise visual requirements.
""".strip()

STYLIST_SYSTEM_PROMPT = """
You are the Stylist Agent from PaperBanana.

Refine the planner description using the style guide while preserving semantics.

Rules:
- Preserve structure and logic.
- Improve aesthetics, readability, and publication readiness.
- Output only the polished description text.
""".strip()

CRITIC_SYSTEM_PROMPT_DIAGRAM = """
You are the Critic Agent from PaperBanana for methodology diagrams.

Review the generated diagram against source context, caption, and current description.
Return strict JSON:
{
  "critic_suggestions": "...",
  "revised_description": "..."
}

If no changes are needed, set both fields to "No changes needed.".
""".strip()

CRITIC_SYSTEM_PROMPT_PLOT = """
You are the Critic Agent from PaperBanana for statistical plots.

Review the generated plot against raw data, communicative intent, and current description.
Return strict JSON:
{
  "critic_suggestions": "...",
  "revised_description": "..."
}

If no changes are needed, set both fields to "No changes needed.".
""".strip()

PLOT_CODE_SYSTEM_PROMPT = """
You are the Visualizer Agent for statistical plots.

Return valid Python code only (no markdown fences) that uses matplotlib and the
variables `raw_data` and `output_path` already available in scope.

Rules:
- Do not import modules.
- Save the final figure to `output_path`.
- Ensure labels and title are readable.
""".strip()


def retriever_system_prompt(task: PaperBananaTask) -> str:
    return (
        RETRIEVER_SYSTEM_PROMPT_PLOT
        if task.mode == "plot"
        else RETRIEVER_SYSTEM_PROMPT_DIAGRAM
    )


def planner_system_prompt(task: PaperBananaTask) -> str:
    return (
        PLANNER_SYSTEM_PROMPT_PLOT
        if task.mode == "plot"
        else PLANNER_SYSTEM_PROMPT_DIAGRAM
    )


def critic_system_prompt(task: PaperBananaTask) -> str:
    return (
        CRITIC_SYSTEM_PROMPT_PLOT
        if task.mode == "plot"
        else CRITIC_SYSTEM_PROMPT_DIAGRAM
    )


def build_retriever_user_prompt(
    task: PaperBananaTask,
    references: list[ReferenceExample],
    top_k: int,
) -> str:
    candidate_lines: list[str] = []
    for ref in references:
        candidate_lines.append(
            "\n".join(
                [
                    f"Reference ID: {ref.ref_id}",
                    f"Caption/Intent: {ref.communicative_intent}",
                    f"Summary: {ref.source_context}",
                ]
            )
        )

    candidates = "\n\n---\n\n".join(candidate_lines)
    base = (
        "Select the best references for this target.\n"
        f"Target caption: {task.communicative_intent}\n"
        f"Target context: {task.source_context}\n"
    )
    if task.mode == "plot":
        base += f"Raw data: {task.raw_data}\n"
    return (
        f"{base}"
        f"Return exactly {top_k} IDs in order of usefulness.\n\n"
        f"Candidate pool:\n{candidates}"
    )


def build_planner_user_prompt(
    task: PaperBananaTask, retrieved_examples: list[ReferenceExample]
) -> str:
    references = "\n\n".join(
        f"- {ref.ref_id}: {ref.communicative_intent}" for ref in retrieved_examples
    )
    payload = (
        f"Source context:\n{task.source_context}\n\n"
        f"Communicative intent:\n{task.communicative_intent}\n\n"
    )
    if task.mode == "plot":
        payload += f"Raw data:\n{task.raw_data}\n\n"
    return (
        f"{payload}"
        f"Retrieved examples:\n{references}\n\n"
        "Write the initial detailed description P for visualization."
    )


def build_stylist_user_prompt(
    planner_description: str, style_guide: str, task: PaperBananaTask
) -> str:
    payload = (
        f"Source context:\n{task.source_context}\n\n"
        f"Communicative intent:\n{task.communicative_intent}\n\n"
    )
    if task.mode == "plot":
        payload += f"Raw data:\n{task.raw_data}\n\n"
    return (
        f"{payload}"
        f"Style guide G:\n{style_guide}\n\n"
        f"Initial description P:\n{planner_description}\n\n"
        "Return optimized description P* only."
    )


def build_critic_user_prompt(
    task: PaperBananaTask, current_description: str, iteration: int
) -> str:
    payload = (
        f"Iteration: {iteration + 1}\n"
        f"Source context:\n{task.source_context}\n\n"
        f"Communicative intent:\n{task.communicative_intent}\n\n"
    )
    if task.mode == "plot":
        payload += f"Raw data:\n{task.raw_data}\n\n"
    return (
        f"{payload}"
        f"Current description P_t:\n{current_description}\n\n"
        "Review the generated artifact and return JSON with critique and revised description P_{t+1}."
    )


def build_plot_code_user_prompt(description: str, task: PaperBananaTask) -> str:
    return (
        f"Target intent: {task.communicative_intent}\n"
        f"Plot description: {description}\n"
        f"Raw data object (Python): {task.raw_data}\n"
        "Write Python code to render a publication-style plot and save it to output_path."
    )
