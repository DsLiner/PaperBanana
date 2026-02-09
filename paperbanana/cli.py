from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv

from paperbanana.agents import PaperBananaAgents
from paperbanana.graph import PaperBananaGraph
from paperbanana.schema import PipelineConfig, ReferenceExample, PaperBananaTask
from paperbanana.style_guides import (
    DEFAULT_DIAGRAM_STYLE_GUIDE,
    DEFAULT_PLOT_STYLE_GUIDE,
)

app = typer.Typer(help="PaperBanana: LangGraph implementation of arXiv:2601.23265")


def _resolve_positive_int_from_env(
    value: int | None,
    env_name: str,
    default: int,
) -> int:
    if value is None:
        raw = os.getenv(env_name, str(default))
        try:
            resolved = int(raw)
        except ValueError as exc:
            raise typer.BadParameter(f"{env_name} must be an integer.") from exc
    else:
        resolved = value

    if resolved < 1:
        raise typer.BadParameter(f"{env_name} must be >= 1.")
    return resolved


def _optional_file(raw_path: str) -> Path | None:
    cleaned = raw_path.strip()
    if not cleaned:
        return None

    path = Path(cleaned).expanduser()
    if not path.exists() or path.is_dir():
        raise typer.BadParameter(
            "Style guide file path must point to an existing file."
        )
    return path


def _execute_run(
    task_file: Path,
    references_file: Path,
    output_dir: Path,
    style_guide_file: Path | None,
    model_name: str | None,
    temperature: float,
    top_k: int,
    max_iterations: int | None,
    mock: bool,
    env_file: Path,
) -> None:
    load_dotenv(dotenv_path=env_file, override=False)

    task_payload = json.loads(task_file.read_text(encoding="utf-8"))
    task = PaperBananaTask.model_validate(task_payload)

    reference_payload = json.loads(references_file.read_text(encoding="utf-8"))
    if not isinstance(reference_payload, list):
        raise typer.BadParameter("references_file must contain a JSON array.")
    references = [ReferenceExample.model_validate(item) for item in reference_payload]

    style_guide = (
        DEFAULT_PLOT_STYLE_GUIDE if task.mode == "plot" else DEFAULT_DIAGRAM_STYLE_GUIDE
    )
    if style_guide_file is not None:
        style_guide = style_guide_file.read_text(encoding="utf-8").strip()

    resolved_model_name = model_name or os.getenv(
        "OPENROUTER_MODEL", "google/gemini-3-pro-preview"
    )
    resolved_max_iterations = _resolve_positive_int_from_env(
        value=max_iterations,
        env_name="PAPERBANANA_MAX_ITERATIONS",
        default=3,
    )

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    openrouter_site_url = os.getenv("OPENROUTER_SITE_URL")
    openrouter_app_name = os.getenv("OPENROUTER_APP_NAME")
    openrouter_image_model = os.getenv(
        "OPENROUTER_IMAGE_MODEL", "google/gemini-3-pro-image-preview"
    )
    openrouter_image_modalities_raw = os.getenv(
        "OPENROUTER_IMAGE_MODALITIES", "image,text"
    )
    openrouter_image_aspect_ratio = os.getenv("OPENROUTER_IMAGE_ASPECT_RATIO", "21:9")
    openrouter_image_size = os.getenv("OPENROUTER_IMAGE_SIZE", "2K")

    openrouter_image_modalities = tuple(
        part.strip()
        for part in openrouter_image_modalities_raw.split(",")
        if part.strip()
    )
    if not openrouter_image_modalities:
        openrouter_image_modalities = ("image",)

    if not mock and not openrouter_api_key:
        raise typer.BadParameter("OPENROUTER_API_KEY is missing. Set it in .env.")

    config = PipelineConfig(
        output_dir=output_dir,
        model_name=resolved_model_name,
        temperature=temperature,
        top_k=top_k,
        max_iterations=resolved_max_iterations,
        use_mock=mock,
        openrouter_api_key=openrouter_api_key,
        openrouter_base_url=openrouter_base_url,
        openrouter_site_url=openrouter_site_url,
        openrouter_app_name=openrouter_app_name,
        openrouter_image_model=openrouter_image_model,
        openrouter_image_modalities=openrouter_image_modalities,
        openrouter_image_aspect_ratio=openrouter_image_aspect_ratio,
        openrouter_image_size=openrouter_image_size,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    graph = PaperBananaGraph(PaperBananaAgents(config=config))
    result = graph.run(task=task, references=references, style_guide=style_guide)

    result_path = config.output_dir / "run_result.json"
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    typer.echo(f"Final artifact: {result.final_artifact}")
    typer.echo(f"Run metadata: {result_path}")


@app.callback()
def main() -> None:
    return None


@app.command()
def run(
    task_file: Path = typer.Option(
        ..., exists=True, dir_okay=False, help="JSON file with task input"
    ),
    references_file: Path = typer.Option(
        ..., exists=True, dir_okay=False, help="JSON file with reference pool"
    ),
    output_dir: Path = typer.Option(
        Path("outputs"), help="Directory for generated artifacts"
    ),
    style_guide_file: Path | None = typer.Option(
        None, exists=True, dir_okay=False, help="Optional text file for style guide"
    ),
    model_name: str | None = typer.Option(
        None,
        help="OpenRouter model name (defaults to OPENROUTER_MODEL or google/gemini-3-pro-preview)",
    ),
    temperature: float = typer.Option(0.3, help="LLM temperature for non-mock mode"),
    top_k: int = typer.Option(10, help="Number of retrieved references"),
    max_iterations: int | None = typer.Option(
        None,
        help="Max Visualizer-Critic rounds (defaults to PAPERBANANA_MAX_ITERATIONS or 3)",
    ),
    mock: bool = typer.Option(
        True,
        "--mock/--no-mock",
        help="Use deterministic local mock responses for text agents",
    ),
    env_file: Path = typer.Option(
        Path(".env"), dir_okay=False, help="Path to .env file"
    ),
) -> None:
    _execute_run(
        task_file=task_file,
        references_file=references_file,
        output_dir=output_dir,
        style_guide_file=style_guide_file,
        model_name=model_name,
        temperature=temperature,
        top_k=top_k,
        max_iterations=max_iterations,
        mock=mock,
        env_file=env_file,
    )


@app.command()
def ui(
    env_file: Path = typer.Option(
        Path(".env"), dir_okay=False, help="Path to .env file"
    ),
) -> None:
    load_dotenv(dotenv_path=env_file, override=False)

    task_file = Path(
        typer.prompt("Task JSON path", default="examples/task.json")
    ).expanduser()
    references_file = Path(
        typer.prompt("Reference pool JSON path", default="examples/reference_pool.json")
    ).expanduser()
    output_dir = Path(typer.prompt("Output directory", default="outputs")).expanduser()
    mock = typer.confirm("Use mock mode (no API calls)?", default=True)

    style_guide_file_raw = typer.prompt("Style guide file path (optional)", default="")
    style_guide_file = _optional_file(style_guide_file_raw)

    model_name: str | None = None
    if not mock:
        model_name = typer.prompt(
            "OpenRouter text model",
            default=os.getenv("OPENROUTER_MODEL", "google/gemini-3-pro-preview"),
        )

    temperature = typer.prompt("Temperature", default=0.3, type=float)
    top_k = typer.prompt("Top-K references", default=10, type=int)
    max_iterations = typer.prompt(
        "Max Critic iterations (upper bound)",
        default=os.getenv("PAPERBANANA_MAX_ITERATIONS", "3"),
        type=int,
    )

    _execute_run(
        task_file=task_file,
        references_file=references_file,
        output_dir=output_dir,
        style_guide_file=style_guide_file,
        model_name=model_name,
        temperature=temperature,
        top_k=top_k,
        max_iterations=max_iterations,
        mock=mock,
        env_file=env_file,
    )


if __name__ == "__main__":
    app()
