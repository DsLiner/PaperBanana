from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

from pydantic import BaseModel, model_validator

Mode = Literal["diagram", "plot"]
RawData = dict[str, object] | list[dict[str, object]] | None


class PaperBananaTask(BaseModel):
    source_context: str
    communicative_intent: str
    mode: Mode = "diagram"
    raw_data: RawData = None


class ReferenceExample(BaseModel):
    ref_id: str
    source_context: str
    communicative_intent: str
    reference_artifact: str | None = None
    reference_image_path: str | None = None
    image_observation: str | None = None
    domain: str | None = None
    diagram_type: str | None = None

    @model_validator(mode="after")
    def _map_legacy_reference_artifact(self) -> "ReferenceExample":
        if self.reference_image_path is None and self.reference_artifact:
            self.reference_image_path = self.reference_artifact
        return self


class PipelineConfig(BaseModel):
    output_dir: Path = Path("outputs")
    model_name: str = "google/gemini-3-pro-preview"
    temperature: float = 0.3
    top_k: int = 10
    max_iterations: int = 3
    use_mock: bool = True
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str | None = None
    openrouter_app_name: str | None = None
    openrouter_image_model: str = "google/gemini-3-pro-image-preview"
    openrouter_image_modalities: tuple[str, ...] = ("image", "text")
    openrouter_image_aspect_ratio: str = "21:9"
    openrouter_image_size: str = "2K"


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
