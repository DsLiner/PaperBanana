from pathlib import Path

import pytest

from paperbanana.agents import PaperBananaAgents
from paperbanana.schema import PipelineConfig, PaperBananaTask, ReferenceExample


def test_reference_example_maps_legacy_reference_artifact() -> None:
    ref = ReferenceExample.model_validate(
        {
            "ref_id": "ref_legacy",
            "source_context": "Legacy source",
            "communicative_intent": "Legacy intent",
            "reference_artifact": "/tmp/legacy.png",
        }
    )
    assert ref.reference_image_path == "/tmp/legacy.png"


def test_reference_example_accepts_new_reference_image_path() -> None:
    ref = ReferenceExample.model_validate(
        {
            "ref_id": "ref_new",
            "source_context": "New source",
            "communicative_intent": "New intent",
            "reference_image_path": "/tmp/new.png",
            "image_observation": "Has two-phase structure",
        }
    )
    assert ref.reference_image_path == "/tmp/new.png"
    assert ref.image_observation == "Has two-phase structure"


def test_generate_reference_from_image_mock_mode_raises(tmp_path: Path) -> None:
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"image-bytes")

    agents = PaperBananaAgents(
        PipelineConfig(output_dir=tmp_path, use_mock=True, max_iterations=1)
    )
    with pytest.raises(
        RuntimeError, match="Image-based reference generation requires use_mock=False"
    ):
        agents.generate_reference_from_image(
            image_path=str(image_path),
            mode="diagram",
            ref_id="img_ref_001",
        )


def test_generate_reference_from_image_non_mock_returns_structured_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"image-bytes")

    agents = PaperBananaAgents(
        PipelineConfig(
            output_dir=tmp_path,
            use_mock=False,
            openrouter_api_key="test-key",
            max_iterations=1,
        )
    )

    def fake_chat_text(
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str] | None = None,
    ) -> str:
        _ = (system_prompt, user_prompt, image_paths)
        return (
            '{"source_context": "A two-stage pipeline where an encoder feeds a '
            'decoder through residual links.", '
            '"communicative_intent": "Explain how data transforms across two '
            'major stages and where supervision is applied.", '
            '"domain": "machine_learning", '
            '"diagram_type": "pipeline", '
            '"image_observation": "Left-to-right blocks with directional arrows, '
            'skip connections, and grouped stage annotations."}'
        )

    monkeypatch.setattr(agents, "_chat_text", fake_chat_text)

    generated = agents.generate_reference_from_image(
        image_path=str(image_path),
        mode="diagram",
        ref_id="img_ref_001",
    )

    assert generated.ref_id == "img_ref_001"
    assert generated.domain == "machine_learning"
    assert generated.diagram_type == "pipeline"
    assert generated.reference_image_path == str(image_path)
    assert "uploaded image" not in generated.source_context.lower()


def test_generate_reference_from_image_retries_once_then_uses_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"image-bytes")

    agents = PaperBananaAgents(
        PipelineConfig(
            output_dir=tmp_path,
            use_mock=False,
            openrouter_api_key="test-key",
            max_iterations=1,
        )
    )

    call_count = {"value": 0}

    def fake_chat_text(
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str] | None = None,
    ) -> str:
        _ = (system_prompt, user_prompt, image_paths)
        call_count["value"] += 1
        return "not-json-response"

    monkeypatch.setattr(agents, "_chat_text", fake_chat_text)

    generated = agents.generate_reference_from_image(
        image_path=str(image_path),
        mode="plot",
        ref_id="img_ref_001",
    )

    assert generated.reference_image_path == str(image_path)
    assert generated.domain == "other"
    assert generated.diagram_type == "line_plot"
    assert "Auto-generation fallback was applied." in (
        generated.image_observation or ""
    )
    assert call_count["value"] == 2


def test_generate_reference_from_image_rejects_forbidden_template_phrasing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"image-bytes")

    agents = PaperBananaAgents(
        PipelineConfig(
            output_dir=tmp_path,
            use_mock=False,
            openrouter_api_key="test-key",
            max_iterations=1,
        )
    )

    def fake_chat_text(
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str] | None = None,
    ) -> str:
        _ = (system_prompt, user_prompt, image_paths)
        return (
            '{"source_context": "Auto-generated diagram from uploaded image reference.png", '
            '"communicative_intent": "Uploaded image summary", '
            '"domain": "other", '
            '"diagram_type": "framework", '
            '"image_observation": "uploaded image with some boxes"}'
        )

    monkeypatch.setattr(agents, "_chat_text", fake_chat_text)

    generated = agents.generate_reference_from_image(
        image_path=str(image_path),
        mode="diagram",
        ref_id="img_ref_001",
    )

    assert generated.domain == "other"
    assert generated.diagram_type == "framework"
    assert "uploaded image" not in generated.source_context.lower()
    assert "reference.png" not in generated.source_context.lower()


def test_planner_uses_only_valid_reference_images(tmp_path: Path) -> None:
    valid_image = tmp_path / "valid.png"
    valid_image.write_bytes(b"image-bytes")
    missing_image = tmp_path / "missing.png"

    agents = PaperBananaAgents(
        PipelineConfig(
            output_dir=tmp_path,
            use_mock=False,
            openrouter_api_key="test-key",
            max_iterations=1,
        )
    )

    captured: dict[str, object] = {}

    def fake_chat_text(
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str] | None = None,
    ) -> str:
        _ = (system_prompt, user_prompt)
        captured["image_paths"] = image_paths
        return "planner-output"

    agents._chat_text = fake_chat_text

    references = [
        ReferenceExample(
            ref_id="r1",
            source_context="A",
            communicative_intent="A",
            reference_image_path=str(valid_image),
        ),
        ReferenceExample(
            ref_id="r2",
            source_context="B",
            communicative_intent="B",
            reference_image_path=str(missing_image),
        ),
        ReferenceExample(
            ref_id="r3",
            source_context="C",
            communicative_intent="C",
            reference_artifact=str(valid_image),
        ),
    ]

    task = PaperBananaTask(
        source_context="System context",
        communicative_intent="Figure intent",
        mode="diagram",
    )

    planner_output = agents.plan(task=task, retrieved_examples=references)
    assert planner_output == "planner-output"
    assert captured["image_paths"] == [str(valid_image)]


def test_planner_falls_back_to_text_only_when_no_valid_reference_images(
    tmp_path: Path,
) -> None:
    agents = PaperBananaAgents(
        PipelineConfig(
            output_dir=tmp_path,
            use_mock=False,
            openrouter_api_key="test-key",
            max_iterations=1,
        )
    )

    captured: dict[str, object] = {}

    def fake_chat_text(
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str] | None = None,
    ) -> str:
        _ = (system_prompt, user_prompt)
        captured["image_paths"] = image_paths
        return "planner-output"

    agents._chat_text = fake_chat_text

    references = [
        ReferenceExample(
            ref_id="r1",
            source_context="A",
            communicative_intent="A",
            reference_image_path=str(tmp_path / "not-found.png"),
        )
    ]
    task = PaperBananaTask(
        source_context="System context",
        communicative_intent="Figure intent",
        mode="diagram",
    )

    planner_output = agents.plan(task=task, retrieved_examples=references)
    assert planner_output == "planner-output"
    assert captured["image_paths"] is None


def test_visualize_non_mock_diagram_failfast_raises_without_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agents = PaperBananaAgents(
        PipelineConfig(
            output_dir=tmp_path,
            use_mock=False,
            strict_non_mock_render=True,
            openrouter_api_key="test-key",
            max_iterations=1,
        )
    )

    def fake_visualize_diagram_openrouter(description: str, output_path: Path) -> None:
        _ = (description, output_path)
        raise RuntimeError("HTTP error from OpenRouter image endpoint")

    monkeypatch.setattr(
        agents,
        "_visualize_diagram_openrouter",
        fake_visualize_diagram_openrouter,
    )

    task = PaperBananaTask(
        source_context="ctx",
        communicative_intent="intent",
        mode="diagram",
    )
    output_path = tmp_path / "diagram_iter_01.png"

    with pytest.raises(RuntimeError, match="Diagram rendering failed in non-mock mode"):
        agents.visualize(
            task=task,
            description="diagram description",
            output_dir=tmp_path,
            iteration=0,
        )

    assert not output_path.exists()


def test_visualize_mock_diagram_returns_placeholder_backend(tmp_path: Path) -> None:
    agents = PaperBananaAgents(
        PipelineConfig(output_dir=tmp_path, use_mock=True, max_iterations=1)
    )
    task = PaperBananaTask(
        source_context="ctx",
        communicative_intent="intent",
        mode="diagram",
    )

    artifact, render_backend, warnings = agents.visualize(
        task=task,
        description="diagram description",
        output_dir=tmp_path,
        iteration=0,
    )

    assert Path(artifact).exists()
    assert render_backend == "mock_placeholder"
    assert warnings
