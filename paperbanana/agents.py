from __future__ import annotations

import base64
import json
import mimetypes
import re
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
    REFERENCE_METADATA_DIAGRAM_TYPES,
    REFERENCE_METADATA_DOMAINS,
    REFERENCE_METADATA_PLOT_TYPES,
    STYLIST_SYSTEM_PROMPT,
    build_critic_user_prompt,
    build_planner_user_prompt,
    build_plot_code_user_prompt,
    build_reference_metadata_system_prompt,
    build_reference_metadata_user_prompt,
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
    _FORBIDDEN_REFERENCE_PATTERNS: tuple[str, ...] = (
        "auto-generated",
        "uploaded image",
        "upload image",
        "reference extracted",
        "could not be fully parsed",
    )

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
        if self.config.use_mock:
            raise RuntimeError(
                "Image-based reference generation requires use_mock=False (--no-mock)."
            )

        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise RuntimeError(f"Reference image does not exist: {image_path}")

        image_name = Path(image_path).name
        retry_feedback: str | None = None
        last_issues: list[str] = []
        for _ in range(2):
            try:
                raw = self._chat_text(
                    build_reference_metadata_system_prompt(mode),
                    build_reference_metadata_user_prompt(
                        mode=mode,
                        ref_id=ref_id,
                        retry_feedback=retry_feedback,
                    ),
                    image_paths=[image_path],
                )
            except Exception as exc:
                last_issues = [f"metadata request failed: {exc}"]
                retry_feedback = "; ".join(last_issues)
                continue

            payload = extract_json_object(raw)
            parsed, issues = self._validate_reference_metadata_payload(
                payload=payload,
                mode=mode,
                image_name=image_name,
            )
            if not issues:
                return ReferenceExample(
                    ref_id=ref_id,
                    source_context=parsed["source_context"],
                    communicative_intent=parsed["communicative_intent"],
                    reference_artifact=image_path,
                    reference_image_path=image_path,
                    image_observation=parsed["image_observation"],
                    domain=parsed["domain"],
                    diagram_type=parsed["diagram_type"],
                )
            last_issues = issues
            retry_feedback = "; ".join(issues)

        fallback = self._build_reference_metadata_fallback(mode)
        if last_issues:
            fallback["image_observation"] = (
                f"{fallback['image_observation']} Auto-generation fallback was applied."
            )
        return ReferenceExample(
            ref_id=ref_id,
            source_context=fallback["source_context"],
            communicative_intent=fallback["communicative_intent"],
            reference_artifact=image_path,
            reference_image_path=image_path,
            image_observation=fallback["image_observation"],
            domain=fallback["domain"],
            diagram_type=fallback["diagram_type"],
        )

    @staticmethod
    def _build_reference_metadata_fallback(mode: Mode) -> dict[str, str]:
        if mode == "plot":
            return {
                "source_context": (
                    "Reference plot showing quantitative relationships and structured data presentation."
                ),
                "communicative_intent": (
                    "Communicate trends, comparisons, or distributions with clear numeric mapping."
                ),
                "domain": "other",
                "diagram_type": "line_plot",
                "image_observation": (
                    "Likely includes chart primitives such as axes, marks, labels, and grouped visual encoding."
                ),
            }
        return {
            "source_context": (
                "Reference diagram showing main components and directional information flow."
            ),
            "communicative_intent": (
                "Communicate module roles, process structure, and interaction logic at a glance."
            ),
            "domain": "other",
            "diagram_type": "framework",
            "image_observation": (
                "Likely includes grouped blocks, connectors, directional arrows, and labeled regions."
            ),
        }

    @staticmethod
    def _sanitize_reference_text(value: str, image_name: str) -> str:
        cleaned = value.strip()
        for marker in PaperBananaAgents._FORBIDDEN_REFERENCE_PATTERNS:
            cleaned = re.sub(
                re.escape(marker),
                "reference figure",
                cleaned,
                flags=re.IGNORECASE,
            )
        cleaned = re.sub(
            re.escape(image_name),
            "reference figure",
            cleaned,
            flags=re.IGNORECASE,
        )
        return " ".join(cleaned.split())

    @staticmethod
    def _extract_text_field(payload: dict[str, object], key: str) -> str | None:
        value = payload.get(key)
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        return cleaned or None

    @staticmethod
    def _normalize_token(value: str) -> str:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return normalized

    def _validate_reference_metadata_payload(
        self,
        payload: dict[str, object],
        mode: Mode,
        image_name: str,
    ) -> tuple[dict[str, str], list[str]]:
        issues: list[str] = []
        parsed: dict[str, str] = {}

        source_context = self._extract_text_field(payload, "source_context")
        communicative_intent = self._extract_text_field(payload, "communicative_intent")
        image_observation = self._extract_text_field(payload, "image_observation")

        for key, value, min_len in (
            ("source_context", source_context, 16),
            ("communicative_intent", communicative_intent, 16),
            ("image_observation", image_observation, 20),
        ):
            if value is None:
                issues.append(f"{key} is missing.")
                continue
            value = self._sanitize_reference_text(value=value, image_name=image_name)
            lowered = value.lower()
            if len(value) < min_len:
                issues.append(f"{key} is too short.")
            if any(marker in lowered for marker in self._FORBIDDEN_REFERENCE_PATTERNS):
                issues.append(f"{key} uses forbidden template phrasing.")
            parsed[key] = value

        domain_value = self._extract_text_field(payload, "domain")
        if domain_value is None:
            parsed["domain"] = "other"
        else:
            domain = self._normalize_token(domain_value)
            domain_aliases = {
                "ml": "machine_learning",
                "cv": "computer_vision",
                "medical": "healthcare",
            }
            domain = domain_aliases.get(domain, domain)
            allowed_domains = set(REFERENCE_METADATA_DOMAINS)
            if domain not in allowed_domains:
                parsed["domain"] = "other"
            else:
                parsed["domain"] = domain

        diagram_type_value = self._extract_text_field(payload, "diagram_type")
        if diagram_type_value is None:
            parsed["diagram_type"] = "line_plot" if mode == "plot" else "framework"
        else:
            diagram_type = self._normalize_token(diagram_type_value)
            diagram_aliases = {
                "line_chart": "line_plot",
                "line_graph": "line_plot",
                "scatter": "scatter_plot",
                "bar": "bar_chart",
                "box": "box_plot",
                "process_flow": "flowchart",
                "method_pipeline": "pipeline",
            }
            diagram_type = diagram_aliases.get(diagram_type, diagram_type)
            allowed_types = (
                set(REFERENCE_METADATA_PLOT_TYPES)
                if mode == "plot"
                else set(REFERENCE_METADATA_DIAGRAM_TYPES)
            )
            if diagram_type not in allowed_types:
                parsed["diagram_type"] = "line_plot" if mode == "plot" else "framework"
            else:
                parsed["diagram_type"] = diagram_type

        return parsed, issues

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
    ) -> tuple[str, str, list[str]]:
        if task.mode == "plot":
            return self._visualize_plot(
                task=task,
                description=description,
                output_dir=output_dir,
                iteration=iteration,
            )

        output_path = output_dir / f"diagram_iter_{iteration + 1:02d}.png"
        if self.config.use_mock:
            save_placeholder_diagram(description=description, output_path=output_path)
            return (
                str(output_path),
                "mock_placeholder",
                ["Mock mode is enabled; generated placeholder diagram image."],
            )

        try:
            self._visualize_diagram_openrouter(
                description=description,
                output_path=output_path,
            )
            return str(output_path), "openrouter_image", []
        except Exception as exc:
            if self.config.strict_non_mock_render:
                raise RuntimeError(
                    f"Diagram rendering failed in non-mock mode. Reason: {exc}"
                ) from exc
            save_placeholder_diagram(description=description, output_path=output_path)
            return (
                str(output_path),
                "mock_placeholder",
                [
                    "Non-mock diagram rendering failed; fallback placeholder was generated. "
                    f"Reason: {exc}"
                ],
            )

    def _visualize_diagram_openrouter(
        self, description: str, output_path: Path
    ) -> None:
        if not self.config.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing in non-mock mode.")

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
        except requests.RequestException as exc:
            raise RuntimeError(
                f"HTTP error from OpenRouter image endpoint: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error during OpenRouter image request: {exc}"
            ) from exc

        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Failed to decode OpenRouter image response JSON."
            ) from exc

        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenRouter image response has no choices.")

        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        message = first_choice.get("message") if isinstance(first_choice, dict) else {}
        if not isinstance(message, dict):
            message = {}

        images = message.get("images")
        if not isinstance(images, list) or not images:
            raise RuntimeError("OpenRouter image response does not contain images.")

        image_obj = images[0]
        if not isinstance(image_obj, dict):
            raise RuntimeError("OpenRouter image payload format is invalid.")

        image_url_obj = image_obj.get("image_url") or image_obj.get("imageUrl") or {}
        image_url = (
            image_url_obj.get("url") if isinstance(image_url_obj, dict) else None
        )
        if not isinstance(image_url, str) or not image_url:
            raise RuntimeError("Failed to parse image URL from OpenRouter response.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if image_url.startswith("data:image"):
            try:
                data = image_url.split(",", maxsplit=1)[1]
                output_path.write_bytes(base64.b64decode(data))
                return
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to decode OpenRouter data URL image payload: {exc}"
                ) from exc

        if image_url.startswith("http://") or image_url.startswith("https://"):
            try:
                download = requests.get(image_url, timeout=60)
                download.raise_for_status()
                output_path.write_bytes(download.content)
                return
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"Failed to download generated image from URL: {exc}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Unexpected download error for generated image: {exc}"
                ) from exc

        raise RuntimeError("Unsupported image URL format in OpenRouter response.")

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
    ) -> tuple[str, str, list[str]]:
        output_path = output_dir / f"plot_iter_{iteration + 1:02d}.png"

        if self.config.use_mock:
            self._render_plot_from_data(task.raw_data, output_path)
            return str(output_path), "plot_mock_data", []

        code_prompt = build_plot_code_user_prompt(description=description, task=task)
        raw_code = self._chat_text(PLOT_CODE_SYSTEM_PROMPT, code_prompt)
        code = strip_markdown_fence(raw_code)
        if not self._execute_plot_code(
            code=code, raw_data=task.raw_data, output_path=output_path
        ):
            self._render_plot_from_data(task.raw_data, output_path)
            return (
                str(output_path),
                "plot_data_fallback",
                [
                    "Plot code execution failed; used deterministic plot fallback renderer."
                ],
            )
        return str(output_path), "plot_generated_code", []

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
