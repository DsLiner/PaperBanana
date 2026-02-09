from pathlib import Path

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


def test_generate_reference_from_image_mock_populates_required_fields(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"image-bytes")

    agents = PaperBananaAgents(
        PipelineConfig(output_dir=tmp_path, use_mock=True, max_iterations=1)
    )
    generated = agents.generate_reference_from_image(
        image_path=str(image_path),
        mode="diagram",
        ref_id="img_ref_001",
    )

    assert generated.ref_id == "img_ref_001"
    assert generated.source_context
    assert generated.communicative_intent
    assert generated.reference_image_path == str(image_path)
    assert generated.image_observation


def test_generate_reference_from_image_fallback_when_json_parse_fails(
    tmp_path: Path,
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
        return "not-json-response"

    agents._chat_text = fake_chat_text  # type: ignore[method-assign]

    generated = agents.generate_reference_from_image(
        image_path=str(image_path),
        mode="plot",
        ref_id="img_ref_001",
    )

    assert generated.source_context.startswith(
        "Reference extracted from uploaded image"
    )
    assert generated.reference_image_path == str(image_path)
    assert generated.diagram_type == "plot"


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

    agents._chat_text = fake_chat_text  # type: ignore[method-assign]

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

    agents._chat_text = fake_chat_text  # type: ignore[method-assign]

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
