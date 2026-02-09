# PaperBanana (LangGraph + LangChain)

This repository implements the core PaperBanana pipeline from the paper **"PaperBanana: Automating Academic Illustration for AI Scientists" (arXiv:2601.23265)**.

Implemented workflow:

1. **Retriever Agent** selects Top-K references.
2. **Planner Agent** builds an initial diagram description `P`.
3. **Stylist Agent** refines it with an aesthetic guideline `G` to produce `P*`.
4. **Visualizer Agent** renders a diagram artifact from the current description.
5. **Critic Agent** critiques and revises the description in a loop (up to `T`, with early stop when no further changes are needed).

The orchestration is implemented with **LangGraph**, and text agents are implemented with **LangChain** chat model interfaces.
The code supports both `diagram` and `plot` modes.

## Environment (uv)

```bash
uv sync
```

- `uv sync`: install/update project dependencies from `pyproject.toml` and `uv.lock`.

## Usage

Command guide:

- `cp .env.example .env`: create local environment config file.
- `uv run paperbanana --help`: show top-level CLI commands.
- `uv run paperbanana run --help`: show all options for the `run` command.
- `uv run paperbanana ui`: launch dashboard UI.
- `uv run paperbanana run ... --mock`: run full pipeline without external API calls.
- `uv run paperbanana run ... --no-mock`: run full pipeline with OpenRouter models from `.env`.

Configure OpenRouter env first:

```bash
cp .env.example .env
```

Required/default `.env` values:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=google/gemini-3-pro-preview
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_SITE_URL=
OPENROUTER_APP_NAME=paperbanana
OPENROUTER_IMAGE_MODEL=google/gemini-3-pro-image-preview
OPENROUTER_IMAGE_MODALITIES=image,text
OPENROUTER_IMAGE_ASPECT_RATIO=21:9
OPENROUTER_IMAGE_SIZE=2K
PAPERBANANA_MAX_ITERATIONS=3
```

Mock run (no API call):

```bash
uv run paperbanana run \
  --task-file examples/task.json \
  --references-file examples/reference_pool.json \
  --output-dir outputs \
  --mock
```

OpenRouter run (diagram, Gemini + image-preview):

```bash
uv run paperbanana run \
  --task-file examples/task.json \
  --references-file examples/reference_pool.json \
  --output-dir outputs \
  --no-mock
```

OpenRouter run (plot):

```bash
uv run paperbanana run \
  --task-file examples/plot_task.json \
  --references-file examples/reference_pool.json \
  --output-dir outputs \
  --no-mock
```

Dashboard UI run:

```bash
uv run paperbanana ui
```

Dashboard usage flow:

1. Run `uv run paperbanana ui` and open the shown local URL (default `http://127.0.0.1:8501`).
2. In the dashboard, set parameters in **Runtime** (mock mode, model, temperature, top-k, max iterations, output dir).
3. Enter prompts in **Prompt Input** (source context, communicative intent, mode, optional plot raw_data, style guide).
4. In **Reference Input**, upload reference images (multiple allowed) and click **Generate references from uploaded images** (available only in `--no-mock` mode).
5. Review/edit appended `References JSON array` before execution.
6. Click **Run Pipeline** to execute and inspect artifacts/feedback/result JSON in the same page.

Reference image behavior:

- Uploaded images are stored in a session temp directory (`/tmp/...`) and linked as `reference_image_path`.
- Auto-generated reference drafts include `source_context`, `communicative_intent`, `domain`, `diagram_type`, and `image_observation`.
- Image-based reference generation is supported only in non-mock mode; mock mode blocks this action in the UI.
- Existing references are preserved; generated items are appended with auto-suffixed `ref_id` on collision.
- Planner uses selected reference images as multimodal inputs when valid paths exist; otherwise it falls back to text-only.

Mock mode scope:

- Mock mode is for offline pipeline/demo behavior and does not guarantee real image understanding quality.
- Non-mock mode (`--no-mock`) is required for image understanding and OpenRouter image rendering.

Run result metadata:

- `run_result.json` includes `render_backend` and `warnings` so you can verify whether real image generation succeeded.

UI with custom env file:

```bash
uv run paperbanana ui --env-file .env
```

UI host/port override:

```bash
uv run paperbanana ui --host 0.0.0.0 --port 8501
```

Optional overrides:

- `--task-file <path>`: JSON task input (required).
- `--references-file <path>`: JSON reference pool (required).
- `--output-dir <path>`: output directory for generated images and metadata.
- `--style-guide-file <path>`: custom style guide text file.
- `--model-name <model>`: override `OPENROUTER_MODEL` for text agents.
- `--temperature <float>`: non-mock text model temperature.
- `--top-k <int>`: number of references selected by Retriever.
- `--max-iterations <int>`: max number of Visualizer-Critic refinement rounds (stops earlier if Critic says no changes needed).
- `--mock / --no-mock`: choose local mock mode or OpenRouter mode.
- `--env-file <path>`: use a different env file path.

Outputs:

- Iteration artifacts: `outputs/diagram_iter_*.png` or `outputs/plot_iter_*.png`
- Run metadata: `outputs/run_result.json`

Plot-mode example:

```bash
uv run paperbanana run \
  --task-file examples/plot_task.json \
  --references-file examples/reference_pool.json \
  --output-dir outputs \
  --mock
```

## Troubleshooting

- **"Only auto-generated template text appears"**: switch to non-mock mode and regenerate references; mock mode does not run real image understanding.
- **"Visualizer Output (mock) appears as the final result"**: this means placeholder rendering was used. In non-mock mode with default strict rendering, diagram generation raises an explicit error instead of silently falling back.

## Testing

```bash
uv run pytest
```

- `uv run pytest`: run test suite.

## Diagnostics (Optional)

```bash
uv add --dev basedpyright
```

- `uv add --dev basedpyright`: manage basedpyright as a project dev dependency.

## Reference

```bibtex
@misc{zhu2026paperbananaautomatingacademicillustration,
      title={PaperBanana: Automating Academic Illustration for AI Scientists},
      author={Dawei Zhu and Rui Meng and Yale Song and Xiyu Wei and Sujian Li and Tomas Pfister and Jinsung Yoon},
      year={2026},
      eprint={2601.23265},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.23265},
}
```
