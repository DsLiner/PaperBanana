# PaperBanana (LangGraph + LangChain)

This repository implements the core PaperBanana pipeline from the paper **"PaperBanana: Automating Academic Illustration for AI Scientists" (arXiv:2601.23265)**.

Implemented workflow:

1. **Retriever Agent** selects Top-K references.
2. **Planner Agent** builds an initial diagram description `P`.
3. **Stylist Agent** refines it with an aesthetic guideline `G` to produce `P*`.
4. **Visualizer Agent** renders a diagram artifact from the current description.
5. **Critic Agent** critiques and revises the description in a loop (`T=3` by default).

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

Optional overrides:

- `--task-file <path>`: JSON task input (required).
- `--references-file <path>`: JSON reference pool (required).
- `--output-dir <path>`: output directory for generated images and metadata.
- `--style-guide-file <path>`: custom style guide text file.
- `--model-name <model>`: override `OPENROUTER_MODEL` for text agents.
- `--temperature <float>`: non-mock text model temperature.
- `--top-k <int>`: number of references selected by Retriever.
- `--max-iterations <int>`: number of Visualizer-Critic refinement rounds.
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
