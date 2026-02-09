from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import matplotlib
import requests

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

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
from paperbanana.schema import (
    Mode,
    PipelineConfig,
    PaperBananaTask,
    RawData,
    ReferenceExample,
)
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
            if not config.openrouter_api_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is required in non-mock mode. Set it in .env."
                )

            headers: dict[str, str] = {}
            if config.openrouter_site_url:
                headers["HTTP-Referer"] = config.openrouter_site_url
            if config.openrouter_app_name:
                headers["X-Title"] = config.openrouter_app_name

            self._chat_model = ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=SecretStr(config.openrouter_api_key),
                base_url=config.openrouter_base_url,
                default_headers=headers or None,
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

    def generate_reference_from_image(
        self,
        image_path: str,
        mode: Mode,
        ref_id: str,
    ) -> ReferenceExample:
        image_name = Path(image_path).name

        if self.config.use_mock:
            diagram_type = "plot" if mode == "plot" else "framework"
            return ReferenceExample(
                ref_id=ref_id,
                source_context=(
                    f"Auto-generated {mode} reference based on uploaded image {image_name}."
                ),
                communicative_intent=(
                    f"Reference visual pattern extracted from {image_name}."
                ),
                reference_artifact=image_path,
                reference_image_path=image_path,
                image_observation=(
                    f"Uploaded image {image_name} appears to show {mode}-oriented layout cues."
                ),
                domain="Auto-generated",
                diagram_type=diagram_type,
            )

        system_prompt = (
            "You analyze academic reference figures for PaperBanana. "
            "Return strict JSON only with keys: source_context, communicative_intent, "
            "domain, diagram_type, image_observation."
        )
        user_prompt = (
            f"Mode: {mode}\n"
            f"Reference ID: {ref_id}\n"
            "Generate concise metadata for this reference image."
        )

        raw = self._chat_text(system_prompt, user_prompt, image_paths=[image_path])
        payload = extract_json_object(raw)

        source_context = payload.get("source_context")
        communicative_intent = payload.get("communicative_intent")
        domain = payload.get("domain")
        diagram_type = payload.get("diagram_type")
        image_observation = payload.get("image_observation")

        if not isinstance(source_context, str) or not source_context.strip():
            source_context = f"Reference extracted from uploaded image {image_name}."
        if (
            not isinstance(communicative_intent, str)
            or not communicative_intent.strip()
        ):
            communicative_intent = (
                f"Communicative intent inferred from uploaded image {image_name}."
            )
        if not isinstance(domain, str) or not domain.strip():
            domain = "Unknown"
        if not isinstance(diagram_type, str) or not diagram_type.strip():
            diagram_type = "plot" if mode == "plot" else "framework"
        if not isinstance(image_observation, str) or not image_observation.strip():
            image_observation = f"Visual observation could not be fully parsed; using fallback for {image_name}."

        return ReferenceExample(
            ref_id=ref_id,
            source_context=source_context.strip(),
            communicative_intent=communicative_intent.strip(),
            reference_artifact=image_path,
            reference_image_path=image_path,
            image_observation=image_observation.strip(),
            domain=domain.strip(),
            diagram_type=diagram_type.strip(),
        )

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
                "loop between Visualizer and Critic until the Critic signals completion, up to "
                f"T={self.config.max_iterations} rounds. Keep labels concise and publication-ready. "
                f"Retrieved references: {example_text}."
            )

        user_prompt = build_planner_user_prompt(
            task=task, retrieved_examples=retrieved_examples
        )
        image_paths = self._collect_reference_image_paths(retrieved_examples)
        return self._chat_text(
            planner_system_prompt(task),
            user_prompt,
            image_paths=image_paths or None,
        ).strip()

    @staticmethod
    def _collect_reference_image_paths(
        retrieved_examples: list[ReferenceExample],
    ) -> list[str]:
        collected: list[str] = []
        seen: set[str] = set()
        for ref in retrieved_examples:
            raw_path = ref.reference_image_path
            if raw_path is None and ref.reference_artifact:
                raw_path = ref.reference_artifact
            if raw_path is None:
                continue
            path = Path(raw_path)
            if not path.exists() or not path.is_file():
                continue
            resolved = str(path)
            if resolved in seen:
                continue
            seen.add(resolved)
            collected.append(resolved)
        return collected

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
        if not self.config.use_mock:
            if self._visualize_diagram_openrouter(
                description=description, output_path=output_path
            ):
                return str(output_path)

        save_placeholder_diagram(description=description, output_path=output_path)
        return str(output_path)

    def _visualize_diagram_openrouter(
        self, description: str, output_path: Path
    ) -> bool:
        if not self.config.openrouter_api_key:
            return False

        headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self.config.openrouter_site_url:
            headers["HTTP-Referer"] = self.config.openrouter_site_url
        if self.config.openrouter_app_name:
            headers["X-Title"] = self.config.openrouter_app_name

        payload = {
            "model": self.config.openrouter_image_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Generate one publication-ready methodology diagram image. "
                        "Do not render figure captions in the image. "
                        f"Diagram description: {description}"
                    ),
                }
            ],
            "modalities": list(self.config.openrouter_image_modalities),
            "image_config": {
                "aspect_ratio": self.config.openrouter_image_aspect_ratio,
                "image_size": self.config.openrouter_image_size,
            },
        }

        endpoint = f"{self.config.openrouter_base_url.rstrip('/')}/chat/completions"
        try:
            response = requests.post(
                endpoint, headers=headers, json=payload, timeout=90
            )
            response.raise_for_status()
            result = response.json()
        except Exception:
            return False

        images = result.get("choices", [{}])[0].get("message", {}).get("images", [])
        if not images:
            return False

        image_obj = images[0]
        image_url_obj = image_obj.get("image_url") or image_obj.get("imageUrl") or {}
        image_url = (
            image_url_obj.get("url") if isinstance(image_url_obj, dict) else None
        )
        if not isinstance(image_url, str) or not image_url:
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if image_url.startswith("data:image"):
            try:
                data = image_url.split(",", maxsplit=1)[1]
                output_path.write_bytes(base64.b64decode(data))
                return True
            except Exception:
                return False

        if image_url.startswith("http://") or image_url.startswith("https://"):
            try:
                download = requests.get(image_url, timeout=60)
                download.raise_for_status()
                output_path.write_bytes(download.content)
                return True
            except Exception:
                return False

        return False

    def critic(
        self,
        task: PaperBananaTask,
        current_description: str,
        image_path: str,
        iteration: int,
    ) -> tuple[str, str]:
        if self.config.use_mock:
            if task.mode == "plot":
                if iteration == 0:
                    revised = (
                        f"{current_description}\n\n"
                        f"Refinement pass {iteration + 1}: improve axis readability, preserve all data points, "
                        "and reduce legend overlap."
                    )
                    return "Improve value readability and labeling clarity.", revised
                return "No changes needed.", "No changes needed."

            if iteration == 0:
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
            critic_system_prompt(task), user_prompt, image_paths=[image_path]
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
    def _to_float(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        if not isinstance(value, (int, float, str)):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_xy(raw_data: RawData) -> tuple[list[float], list[float]]:
        if isinstance(raw_data, dict):
            x_values = raw_data.get("x")
            y_values = raw_data.get("y")
            if (
                isinstance(x_values, list)
                and isinstance(y_values, list)
                and len(x_values) == len(y_values)
            ):
                x: list[float] = []
                y: list[float] = []
                for x_item, y_item in zip(x_values, y_values):
                    x_value = PaperBananaAgents._to_float(x_item)
                    y_value = PaperBananaAgents._to_float(y_item)
                    if x_value is None or y_value is None:
                        x = []
                        y = []
                        break
                    x.append(x_value)
                    y.append(y_value)
                if x and len(x) == len(y):
                    return x, y

        if isinstance(raw_data, list):
            x: list[float] = []
            y: list[float] = []
            for item in raw_data:
                if not isinstance(item, dict):
                    continue
                x_value = PaperBananaAgents._to_float(item.get("x"))
                y_value = PaperBananaAgents._to_float(item.get("y"))
                if x_value is None or y_value is None:
                    continue
                x.append(x_value)
                y.append(y_value)
            if x and len(x) == len(y):
                return x, y

        return [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 2.5, 4.2]

    def _render_plot_from_data(self, raw_data: RawData, output_path: Path) -> None:
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
        self, code: str, raw_data: RawData, output_path: Path
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
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str] | None = None,
    ) -> str:
        if self._chat_model is None:
            raise RuntimeError(
                "Chat model is not configured. Set use_mock=False with valid API credentials."
            )

        if not image_paths:
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

        content_blocks: list[str | dict[str, Any]] = [
            {"type": "text", "text": user_prompt}
        ]
        valid_image_count = 0
        for image_path in image_paths:
            path = Path(image_path)
            if not path.exists() or not path.is_file():
                continue
            image_bytes = path.read_bytes()
            encoded = base64.b64encode(image_bytes).decode("ascii")
            mime_type, _ = mimetypes.guess_type(str(path))
            if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
                mime_type = "image/png"
            content_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                }
            )
            valid_image_count += 1

        if valid_image_count == 0:
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

        message = HumanMessage(content=content_blocks)
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
