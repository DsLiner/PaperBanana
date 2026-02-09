from pathlib import Path

from paperbanana.agents import PaperBananaAgents
from paperbanana.graph import PaperBananaGraph
from paperbanana.schema import PipelineConfig, ReferenceExample, PaperBananaTask
from paperbanana.style_guides import (
    DEFAULT_DIAGRAM_STYLE_GUIDE,
    DEFAULT_PLOT_STYLE_GUIDE,
)


def test_mock_pipeline_can_stop_early_from_critic_feedback(tmp_path: Path) -> None:
    task = PaperBananaTask(
        source_context="A five-agent framework with retrieval, planning, styling, visualization, and critique.",
        communicative_intent="Overview diagram of the full workflow.",
    )

    references = [
        ReferenceExample(
            ref_id=f"ref_{idx:03d}",
            source_context="Reference method context.",
            communicative_intent="Reference diagram intent.",
        )
        for idx in range(1, 13)
    ]

    config = PipelineConfig(
        output_dir=tmp_path,
        top_k=10,
        max_iterations=3,
        use_mock=True,
    )
    graph = PaperBananaGraph(agents=PaperBananaAgents(config=config))

    result = graph.run(
        task=task, references=references, style_guide=DEFAULT_PLOT_STYLE_GUIDE
    )

    assert len(result.retrieved_ids) == 10
    assert len(result.artifact_history) == 2
    assert Path(result.final_artifact).exists()
    assert result.render_backend == "mock_placeholder"
    assert result.warnings


def test_mock_plot_mode_generates_plot_artifact(tmp_path: Path) -> None:
    task = PaperBananaTask(
        source_context="Plot monthly score progression.",
        communicative_intent="Line chart over months.",
        mode="plot",
        raw_data={"x": [1, 2, 3, 4], "y": [10, 12, 15, 18]},
    )

    references = [
        ReferenceExample(
            ref_id=f"ref_{idx:03d}",
            source_context="Reference method context.",
            communicative_intent="Reference diagram intent.",
        )
        for idx in range(1, 13)
    ]

    config = PipelineConfig(
        output_dir=tmp_path,
        top_k=10,
        max_iterations=3,
        use_mock=True,
    )
    graph = PaperBananaGraph(agents=PaperBananaAgents(config=config))

    result = graph.run(
        task=task, references=references, style_guide=DEFAULT_DIAGRAM_STYLE_GUIDE
    )

    assert result.final_artifact.endswith("plot_iter_02.png")
    assert len(result.artifact_history) == 2
    assert Path(result.final_artifact).exists()
    assert result.render_backend == "plot_mock_data"


def test_max_iterations_cap_is_respected(tmp_path: Path) -> None:
    task = PaperBananaTask(
        source_context="A five-agent framework with retrieval, planning, styling, visualization, and critique.",
        communicative_intent="Overview diagram of the full workflow.",
    )

    references = [
        ReferenceExample(
            ref_id=f"ref_{idx:03d}",
            source_context="Reference method context.",
            communicative_intent="Reference diagram intent.",
        )
        for idx in range(1, 13)
    ]

    config = PipelineConfig(
        output_dir=tmp_path,
        top_k=10,
        max_iterations=1,
        use_mock=True,
    )
    graph = PaperBananaGraph(agents=PaperBananaAgents(config=config))

    result = graph.run(
        task=task, references=references, style_guide=DEFAULT_PLOT_STYLE_GUIDE
    )

    assert len(result.artifact_history) == 1
    assert result.final_artifact.endswith("diagram_iter_01.png")
    assert result.render_backend == "mock_placeholder"
