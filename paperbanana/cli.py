from __future__ import annotations

import json
from pathlib import Path

import typer

from paperbanana.agents import PaperBananaAgents
from paperbanana.graph import PaperBananaGraph
from paperbanana.schema import PipelineConfig, ReferenceExample, PaperBananaTask
from paperbanana.style_guides import (
    DEFAULT_DIAGRAM_STYLE_GUIDE,
    DEFAULT_PLOT_STYLE_GUIDE,
)

app = typer.Typer(help="PaperBanana: LangGraph implementation of arXiv:2601.23265")


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
    model_name: str = typer.Option("gpt-4o-mini", help="LLM name for non-mock mode"),
    temperature: float = typer.Option(0.3, help="LLM temperature for non-mock mode"),
    top_k: int = typer.Option(10, help="Number of retrieved references"),
    max_iterations: int = typer.Option(3, help="Visualizer-Critic loop count"),
    mock: bool = typer.Option(
        True,
        "--mock/--no-mock",
        help="Use deterministic local mock responses for text agents",
    ),
) -> None:
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

    config = PipelineConfig(
        output_dir=output_dir,
        model_name=model_name,
        temperature=temperature,
        top_k=top_k,
        max_iterations=max_iterations,
        use_mock=mock,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    graph = PaperBananaGraph(PaperBananaAgents(config=config))
    result = graph.run(task=task, references=references, style_guide=style_guide)

    result_path = config.output_dir / "run_result.json"
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    typer.echo(f"Final artifact: {result.final_artifact}")
    typer.echo(f"Run metadata: {result_path}")


if __name__ == "__main__":
    app()
