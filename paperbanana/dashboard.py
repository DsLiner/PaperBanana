from __future__ import annotations

import atexit
import json
import importlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from paperbanana.agents import PaperBananaAgents
from paperbanana.graph import PaperBananaGraph
from paperbanana.schema import PipelineConfig, ReferenceExample, PaperBananaTask
from paperbanana.style_guides import (
    DEFAULT_DIAGRAM_STYLE_GUIDE,
    DEFAULT_PLOT_STYLE_GUIDE,
)

st = importlib.import_module("streamlit")

_REGISTERED_TEMP_DIRS: set[str] = set()


@atexit.register
def _cleanup_registered_temp_dirs() -> None:
    for raw_dir in list(_REGISTERED_TEMP_DIRS):
        shutil.rmtree(raw_dir, ignore_errors=True)
    _REGISTERED_TEMP_DIRS.clear()


def _env_file_path() -> Path:
    raw = os.getenv("PAPERBANANA_ENV_FILE", ".env")
    return Path(raw).expanduser()


def _load_env() -> None:
    load_dotenv(dotenv_path=_env_file_path(), override=False)


def _read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _cleanup_session_temp_dir() -> None:
    previous = st.session_state.get("upload_temp_dir")
    if isinstance(previous, str) and previous:
        shutil.rmtree(previous, ignore_errors=True)
        _REGISTERED_TEMP_DIRS.discard(previous)
    st.session_state["upload_temp_dir"] = ""
    st.session_state["uploaded_image_paths"] = []


def _store_uploaded_images(uploaded_files: list[Any]) -> list[str]:
    _cleanup_session_temp_dir()

    temp_dir = Path(tempfile.mkdtemp(prefix="paperbanana_refs_"))
    _REGISTERED_TEMP_DIRS.add(str(temp_dir))
    st.session_state["upload_temp_dir"] = str(temp_dir)

    saved_paths: list[str] = []
    for index, uploaded in enumerate(uploaded_files, start=1):
        file_name = str(getattr(uploaded, "name", f"reference_{index:03d}.png"))
        suffix = Path(file_name).suffix.lower() or ".png"
        target = temp_dir / f"reference_{index:03d}{suffix}"

        data = uploaded.getvalue()
        if not isinstance(data, bytes):
            continue
        target.write_bytes(data)
        saved_paths.append(str(target))

    st.session_state["uploaded_image_paths"] = saved_paths
    return saved_paths


def _next_ref_id(base: str, used_ids: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def _ensure_state_defaults() -> None:
    if "source_context" in st.session_state:
        return

    task_path = Path("examples/task.json")
    references_path = Path("examples/reference_pool.json")

    task_payload: dict[str, Any] = {}
    references_payload: list[dict[str, Any]] = []
    if task_path.exists():
        loaded_task = _read_json_file(task_path)
        if isinstance(loaded_task, dict):
            task_payload = loaded_task
    if references_path.exists():
        loaded_refs = _read_json_file(references_path)
        if isinstance(loaded_refs, list):
            references_payload = [
                item for item in loaded_refs if isinstance(item, dict)
            ]

    mode = str(task_payload.get("mode", "diagram"))
    if mode not in {"diagram", "plot"}:
        mode = "diagram"

    style_default = (
        DEFAULT_PLOT_STYLE_GUIDE if mode == "plot" else DEFAULT_DIAGRAM_STYLE_GUIDE
    )

    st.session_state["task_path"] = str(task_path)
    st.session_state["references_path"] = str(references_path)
    st.session_state["source_context"] = str(task_payload.get("source_context", ""))
    st.session_state["communicative_intent"] = str(
        task_payload.get("communicative_intent", "")
    )
    st.session_state["mode"] = mode
    raw_data = task_payload.get("raw_data")
    st.session_state["raw_data_text"] = (
        "" if raw_data is None else _to_json_text(raw_data)
    )
    st.session_state["references_text"] = _to_json_text(references_payload)
    st.session_state["style_guide"] = style_default

    st.session_state["output_dir"] = "outputs"
    st.session_state["mock_mode"] = True
    st.session_state["model_name"] = os.getenv(
        "OPENROUTER_MODEL", "google/gemini-3-pro-preview"
    )
    st.session_state["temperature"] = 0.3
    st.session_state["top_k"] = 10
    st.session_state["max_iterations"] = int(
        os.getenv("PAPERBANANA_MAX_ITERATIONS", "3")
    )
    st.session_state["upload_temp_dir"] = ""
    st.session_state["uploaded_image_paths"] = []


def _sync_style_with_mode() -> None:
    mode = st.session_state["mode"]
    if (
        mode == "plot"
        and st.session_state["style_guide"].strip() == DEFAULT_DIAGRAM_STYLE_GUIDE
    ):
        st.session_state["style_guide"] = DEFAULT_PLOT_STYLE_GUIDE
    if (
        mode == "diagram"
        and st.session_state["style_guide"].strip() == DEFAULT_PLOT_STYLE_GUIDE
    ):
        st.session_state["style_guide"] = DEFAULT_DIAGRAM_STYLE_GUIDE


def _load_inputs_from_files(task_path_text: str, references_path_text: str) -> None:
    task_path = Path(task_path_text).expanduser()
    references_path = Path(references_path_text).expanduser()

    task_payload = _read_json_file(task_path)
    if not isinstance(task_payload, dict):
        raise ValueError("Task JSON must be an object.")

    references_payload = _read_json_file(references_path)
    if not isinstance(references_payload, list):
        raise ValueError("Reference JSON must be an array.")

    mode = str(task_payload.get("mode", "diagram"))
    if mode not in {"diagram", "plot"}:
        mode = "diagram"

    st.session_state["task_path"] = str(task_path)
    st.session_state["references_path"] = str(references_path)
    st.session_state["source_context"] = str(task_payload.get("source_context", ""))
    st.session_state["communicative_intent"] = str(
        task_payload.get("communicative_intent", "")
    )
    st.session_state["mode"] = mode

    raw_data = task_payload.get("raw_data")
    st.session_state["raw_data_text"] = (
        "" if raw_data is None else _to_json_text(raw_data)
    )
    refs_dict_list = [item for item in references_payload if isinstance(item, dict)]
    st.session_state["references_text"] = _to_json_text(refs_dict_list)
    st.session_state["style_guide"] = (
        DEFAULT_PLOT_STYLE_GUIDE if mode == "plot" else DEFAULT_DIAGRAM_STYLE_GUIDE
    )


def _generate_references_from_uploaded_images(uploaded_files: list[Any]) -> int:
    if not uploaded_files:
        raise ValueError("Upload at least one reference image.")

    references_payload = json.loads(str(st.session_state["references_text"]))
    if not isinstance(references_payload, list):
        raise ValueError(
            "References JSON must be an array before appending image references."
        )

    references_dicts = [item for item in references_payload if isinstance(item, dict)]
    used_ids = {
        str(item.get("ref_id"))
        for item in references_dicts
        if isinstance(item.get("ref_id"), str)
    }

    image_paths = _store_uploaded_images(uploaded_files)
    if not image_paths:
        raise ValueError("Failed to save uploaded images.")

    config = _build_config_from_ui()
    agents = PaperBananaAgents(config=config)

    mode_value = str(st.session_state["mode"])
    mode: Literal["diagram", "plot"] = "plot" if mode_value == "plot" else "diagram"

    for index, image_path in enumerate(image_paths, start=1):
        ref_id = _next_ref_id(f"img_ref_{index:03d}", used_ids)
        generated = agents.generate_reference_from_image(
            image_path=image_path,
            mode=mode,
            ref_id=ref_id,
        )
        references_dicts.append(generated.model_dump())

    st.session_state["references_text"] = _to_json_text(references_dicts)
    return len(image_paths)


def _build_config_from_ui() -> PipelineConfig:
    max_iterations = int(st.session_state["max_iterations"])
    if max_iterations < 1:
        raise ValueError("Max Critic iterations must be >= 1.")

    top_k = int(st.session_state["top_k"])
    if top_k < 1:
        raise ValueError("Top-K must be >= 1.")

    openrouter_image_modalities_raw = os.getenv(
        "OPENROUTER_IMAGE_MODALITIES", "image,text"
    )
    openrouter_image_modalities = tuple(
        part.strip()
        for part in openrouter_image_modalities_raw.split(",")
        if part.strip()
    )
    if not openrouter_image_modalities:
        openrouter_image_modalities = ("image",)

    use_mock = bool(st.session_state["mock_mode"])
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not use_mock and not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is missing. Set it in .env.")

    return PipelineConfig(
        output_dir=Path(st.session_state["output_dir"]).expanduser(),
        model_name=str(st.session_state["model_name"]),
        temperature=float(st.session_state["temperature"]),
        top_k=top_k,
        max_iterations=max_iterations,
        use_mock=use_mock,
        openrouter_api_key=openrouter_api_key,
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        openrouter_site_url=os.getenv("OPENROUTER_SITE_URL"),
        openrouter_app_name=os.getenv("OPENROUTER_APP_NAME"),
        openrouter_image_model=os.getenv(
            "OPENROUTER_IMAGE_MODEL", "google/gemini-3-pro-image-preview"
        ),
        openrouter_image_modalities=openrouter_image_modalities,
        openrouter_image_aspect_ratio=os.getenv(
            "OPENROUTER_IMAGE_ASPECT_RATIO", "21:9"
        ),
        openrouter_image_size=os.getenv("OPENROUTER_IMAGE_SIZE", "2K"),
    )


def _run_pipeline() -> dict[str, Any]:
    mode_value = str(st.session_state["mode"])
    mode: Literal["diagram", "plot"] = "plot" if mode_value == "plot" else "diagram"
    raw_data: Any = None
    raw_data_text = str(st.session_state["raw_data_text"]).strip()
    if mode == "plot" and raw_data_text:
        raw_data = json.loads(raw_data_text)
        if not isinstance(raw_data, (dict, list)):
            raise ValueError("Plot raw_data must be a JSON object or array.")

    task = PaperBananaTask(
        source_context=str(st.session_state["source_context"]),
        communicative_intent=str(st.session_state["communicative_intent"]),
        mode=mode,
        raw_data=raw_data,
    )

    references_payload = json.loads(str(st.session_state["references_text"]))
    if not isinstance(references_payload, list):
        raise ValueError("References must be a JSON array.")

    references = [ReferenceExample.model_validate(item) for item in references_payload]
    config = _build_config_from_ui()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    result = PaperBananaGraph(PaperBananaAgents(config=config)).run(
        task=task,
        references=references,
        style_guide=str(st.session_state["style_guide"]),
    )

    result_path = config.output_dir / "run_result.json"
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    payload = result.model_dump()
    payload["run_result_path"] = str(result_path)
    return payload


def _render_results() -> None:
    run_result = st.session_state.get("run_result")
    if not isinstance(run_result, dict):
        return

    st.subheader("Run Result")
    st.success(f"Final artifact: {run_result.get('final_artifact', '')}")
    st.caption(f"Metadata: {run_result.get('run_result_path', '')}")

    artifact_history = run_result.get("artifact_history", [])
    if isinstance(artifact_history, list) and artifact_history:
        st.write("Artifacts")
        for artifact in artifact_history:
            if not isinstance(artifact, str):
                continue
            path = Path(artifact)
            st.markdown(f"- `{artifact}`")
            if path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                st.image(str(path), caption=path.name)

    critic_feedback = run_result.get("critic_feedback", [])
    if isinstance(critic_feedback, list) and critic_feedback:
        st.write("Critic feedback")
        for idx, feedback in enumerate(critic_feedback, start=1):
            if isinstance(feedback, str):
                st.markdown(f"{idx}. {feedback}")

    st.write("Raw result JSON")
    st.json(run_result)


def main() -> None:
    st.set_page_config(page_title="PaperBanana Dashboard", layout="wide")
    _load_env()
    _ensure_state_defaults()
    _sync_style_with_mode()

    st.title("PaperBanana Dashboard")
    st.caption("Configure prompts and parameters, then run the pipeline.")

    with st.sidebar:
        st.header("Runtime")
        st.text_input("Env file", value=str(_env_file_path()), disabled=True)
        st.checkbox("Mock mode", key="mock_mode")
        st.text_input("OpenRouter text model", key="model_name")
        st.slider(
            "Temperature", min_value=0.0, max_value=1.0, step=0.05, key="temperature"
        )
        st.number_input("Top-K references", min_value=1, step=1, key="top_k")
        st.number_input(
            "Max Critic iterations",
            min_value=1,
            step=1,
            key="max_iterations",
        )
        st.text_input("Output directory", key="output_dir")
        run_clicked = st.button(
            "Run Pipeline", type="primary", use_container_width=True
        )

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Prompt Input")
        st.selectbox("Mode", options=["diagram", "plot"], key="mode")
        st.text_area("Source context", key="source_context", height=180)
        st.text_area("Communicative intent", key="communicative_intent", height=140)
        if st.session_state["mode"] == "plot":
            st.text_area("Plot raw_data (JSON)", key="raw_data_text", height=160)
        st.text_area("Style guide", key="style_guide", height=220)

    with right:
        st.subheader("Reference Input")
        st.text_input("Task JSON path", key="task_path")
        st.text_input("Reference JSON path", key="references_path")
        if st.button("Load from files"):
            try:
                _load_inputs_from_files(
                    task_path_text=str(st.session_state["task_path"]),
                    references_path_text=str(st.session_state["references_path"]),
                )
                st.success("Loaded task and references from files.")
            except Exception as exc:
                st.error(str(exc))

        uploaded_images = st.file_uploader(
            "Upload reference images",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )
        if st.button("Generate references from uploaded images"):
            try:
                generated_count = _generate_references_from_uploaded_images(
                    uploaded_files=list(uploaded_images or [])
                )
                st.success(
                    f"Generated {generated_count} references and appended them to References JSON."
                )
            except Exception as exc:
                st.error(str(exc))

        temp_paths = st.session_state.get("uploaded_image_paths", [])
        if isinstance(temp_paths, list) and temp_paths:
            st.caption("Current session temp image paths")
            for image_path in temp_paths:
                if isinstance(image_path, str):
                    st.markdown(f"- `{image_path}`")

        st.text_area("References JSON array", key="references_text", height=420)

    if run_clicked:
        try:
            with st.spinner("Running pipeline..."):
                st.session_state["run_result"] = _run_pipeline()
        except Exception as exc:
            st.error(str(exc))

    _render_results()


if __name__ == "__main__":
    main()
