from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

Mode = Literal["diagram", "plot"]


class PaperBananaTask(BaseModel):
    source_context: str
    communicative_intent: str
    mode: Mode = "diagram"
    raw_data: dict | list[dict] | None = None


class ReferenceExample(BaseModel):
    ref_id: str
    source_context: str
    communicative_intent: str
    reference_artifact: str | None = None
    domain: str | None = None
    diagram_type: str | None = None


class PipelineConfig(BaseModel):
    output_dir: Path = Path("outputs")
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3
    top_k: int = 10
    max_iterations: int = 3
    use_mock: bool = True


class PipelineResult(BaseModel):
    final_artifact: str
    artifact_history: list[str]
    retrieved_ids: list[str]
    planner_description: str
    styled_description: str
    critic_feedback: list[str]


class PaperBananaState(TypedDict):
    task: PaperBananaTask
    reference_pool: list[ReferenceExample]
    style_guide: str
    top_k: int
    max_iterations: int
    retrieved_ids: list[str]
    retrieved_examples: list[ReferenceExample]
    planner_description: str
    styled_description: str
    current_description: str
    latest_artifact: str
    artifact_history: list[str]
    critic_feedback: list[str]
    iteration: int
    stop_refinement: bool
    final_artifact: str
