from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from paperbanana.prompts import (
    PLOT_CODE_SYSTEM_PROMPT,
    STYLIST_SYSTEM_PROMPT,
    build_critic_user_prompt,
    build_planner_user_prompt,
    build_plot_code_user_prompt,
    build_retriever_user_prompt,
    build_stylist_user_prompt,
    critic_system_prompt,
    planner_system_prompt,
    retriever_system_prompt,
)
from paperbanana.schema import PipelineConfig, PaperBananaTask, ReferenceExample
from paperbanana.utils import (
    extract_json_object,
    save_placeholder_diagram,
    strip_markdown_fence,
)


class PaperBananaAgents:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._chat_model = None
        if not config.use_mock:
            self._chat_model = ChatOpenAI(
                model=config.model_name, temperature=config.temperature
            )

    def retrieve(
        self, task: PaperBananaTask, references: list[ReferenceExample], top_k: int
    ) -> list[str]:
        if self.config.use_mock:
            return [ref.ref_id for ref in references[:top_k]]

        user_prompt = build_retriever_user_prompt(
            task=task, references=references, top_k=top_k
        )
        raw = self._chat_text(retriever_system_prompt(task), user_prompt)
        data = extract_json_object(raw)
        key = "top_10_plots" if task.mode == "plot" else "top_10_papers"
        selected = data.get(key, [])
        if not isinstance(selected, list):
            selected = []

        ordered_ids: list[str] = []
        seen: set[str] = set()
        valid_ids = {ref.ref_id for ref in references}
        for item in selected:
            if not isinstance(item, str):
                continue
            if item in valid_ids and item not in seen:
                ordered_ids.append(item)
                seen.add(item)
            if len(ordered_ids) == top_k:
                break

        if len(ordered_ids) < top_k:
            for ref in references:
                if ref.ref_id in seen:
                    continue
                ordered_ids.append(ref.ref_id)
                if len(ordered_ids) == top_k:
                    break

        return ordered_ids

    def plan(
        self, task: PaperBananaTask, retrieved_examples: list[ReferenceExample]
    ) -> str:
        if self.config.use_mock:
            if task.mode == "plot":
                return (
                    "Create a statistically faithful plot plan from raw data. "
                    "Map x and y clearly, include concise labels, and prepare for iterative refinement."
                )
            example_text = ", ".join(example.ref_id for example in retrieved_examples)
            return (
                "Create a left-to-right methodology diagram with five agents: Retriever, Planner, Stylist, "
                "Visualizer, and Critic. Include source context S and communicative intent C as inputs. "
                "Show retrieved examples E feeding Planner, style guideline G feeding Stylist, and a refinement "
                "loop between Visualizer and Critic for T=3 rounds. Keep labels concise and publication-ready. "
                f"Retrieved references: {example_text}."
            )

        user_prompt = build_planner_user_prompt(
            task=task, retrieved_examples=retrieved_examples
        )
        return self._chat_text(planner_system_prompt(task), user_prompt).strip()

    def style(
        self, task: PaperBananaTask, planner_description: str, style_guide: str
    ) -> str:
        if self.config.use_mock:
            if task.mode == "plot":
                return (
                    f"{planner_description}\n\n"
                    "Apply style: high-contrast palette, readable axis labels, light dashed grids, "
                    "and clear legend placement for publication quality."
                )
            return (
                f"{planner_description}\n\n"
                "Apply style: pastel phase containers, clear orthogonal arrows for primary flow, "
                "dashed auxiliary links, sans-serif labels, serif italic variables, white background."
            )

        user_prompt = build_stylist_user_prompt(
            planner_description=planner_description,
            style_guide=style_guide,
            task=task,
        )
        return self._chat_text(STYLIST_SYSTEM_PROMPT, user_prompt).strip()

    def visualize(
        self, task: PaperBananaTask, description: str, output_dir: Path, iteration: int
    ) -> str:
        if task.mode == "plot":
            return self._visualize_plot(
                task=task,
                description=description,
                output_dir=output_dir,
                iteration=iteration,
            )

        output_path = output_dir / f"diagram_iter_{iteration + 1:02d}.png"
        save_placeholder_diagram(description=description, output_path=output_path)
        return str(output_path)

    def critic(
        self,
        task: PaperBananaTask,
        current_description: str,
        image_path: str,
        iteration: int,
    ) -> tuple[str, str]:
        if self.config.use_mock:
            if task.mode == "plot":
                if iteration + 1 < self.config.max_iterations:
                    revised = (
                        f"{current_description}\n\n"
                        f"Refinement pass {iteration + 1}: improve axis readability, preserve all data points, "
                        "and reduce legend overlap."
                    )
                    return "Improve value readability and labeling clarity.", revised
                return "No changes needed.", "No changes needed."

            if iteration + 1 < self.config.max_iterations:
                revised = (
                    f"{current_description}\n\n"
                    f"Refinement pass {iteration + 1}: improve arrow clarity, remove label clutter, "
                    "and tighten alignment with source methodology."
                )
                return "Improve fidelity and reduce visual clutter.", revised
            return "No changes needed.", "No changes needed."

        user_prompt = build_critic_user_prompt(
            task=task,
            current_description=current_description,
            iteration=iteration,
        )
        raw = self._chat_text(
            critic_system_prompt(task), user_prompt, image_path=image_path
        )
        payload = extract_json_object(raw)
        suggestions = payload.get("critic_suggestions", "No changes needed.")
        revised = payload.get("revised_description", "No changes needed.")

        if not isinstance(suggestions, str):
            suggestions = "No changes needed."
        if not isinstance(revised, str):
            revised = "No changes needed."

        return suggestions.strip(), revised.strip()

    def _visualize_plot(
        self,
        task: PaperBananaTask,
        description: str,
        output_dir: Path,
        iteration: int,
    ) -> str:
        output_path = output_dir / f"plot_iter_{iteration + 1:02d}.png"

        if self.config.use_mock:
            self._render_plot_from_data(task.raw_data, output_path)
            return str(output_path)

        code_prompt = build_plot_code_user_prompt(description=description, task=task)
        raw_code = self._chat_text(PLOT_CODE_SYSTEM_PROMPT, code_prompt)
        code = strip_markdown_fence(raw_code)
        if not self._execute_plot_code(
            code=code, raw_data=task.raw_data, output_path=output_path
        ):
            self._render_plot_from_data(task.raw_data, output_path)
        return str(output_path)

    @staticmethod
    def _extract_xy(
        raw_data: dict | list[dict] | None,
    ) -> tuple[list[float], list[float]]:
        if isinstance(raw_data, dict):
            x_values = raw_data.get("x")
            y_values = raw_data.get("y")
            if (
                isinstance(x_values, list)
                and isinstance(y_values, list)
                and len(x_values) == len(y_values)
            ):
                try:
                    x = [float(item) for item in x_values]
                    y = [float(item) for item in y_values]
                    return x, y
                except (TypeError, ValueError):
                    pass

        if isinstance(raw_data, list):
            x: list[float] = []
            y: list[float] = []
            for item in raw_data:
                if not isinstance(item, dict):
                    continue
                if "x" not in item or "y" not in item:
                    continue
                try:
                    x.append(float(item["x"]))
                    y.append(float(item["y"]))
                except (TypeError, ValueError):
                    continue
            if x and len(x) == len(y):
                return x, y

        return [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 2.5, 4.2]

    def _render_plot_from_data(
        self, raw_data: dict | list[dict] | None, output_path: Path
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        x, y = self._extract_xy(raw_data)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, marker="o", linewidth=2.0, color="#1f77b4")
        ax.set_title("PaperBanana Plot Output")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def _execute_plot_code(
        self, code: str, raw_data: dict | list[dict] | None, output_path: Path
    ) -> bool:
        safe_builtins = {
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "min": min,
            "max": max,
            "sum": sum,
            "float": float,
            "int": int,
            "str": str,
            "list": list,
            "dict": dict,
            "json": json,
        }
        globals_dict = {
            "__builtins__": safe_builtins,
            "plt": plt,
            "raw_data": raw_data,
            "output_path": str(output_path),
        }
        try:
            exec(code, globals_dict, {})
        except Exception:
            plt.close("all")
            return False

        return output_path.exists()

    def _chat_text(
        self, system_prompt: str, user_prompt: str, image_path: str | None = None
    ) -> str:
        if self._chat_model is None:
            raise RuntimeError(
                "Chat model is not configured. Set use_mock=False with valid API credentials."
            )

        if image_path is None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "{system_prompt}"),
                    ("human", "{user_prompt}"),
                ]
            )
            chain = prompt | self._chat_model | StrOutputParser()
            return chain.invoke(
                {"system_prompt": system_prompt, "user_prompt": user_prompt}
            )

        image_bytes = Path(image_path).read_bytes()
        import base64

        encoded = base64.b64encode(image_bytes).decode("ascii")
        message = HumanMessage(
            content=[
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ]
        )
        response = self._chat_model.invoke(
            [SystemMessage(content=system_prompt), message]
        )

        if isinstance(response.content, str):
            return response.content

        chunks: list[str] = []
        if isinstance(response.content, list):
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
        return "\n".join(chunks)
