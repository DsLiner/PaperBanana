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

## Run

Mock mode (runs without API keys):

```bash
uv run paperbanana run \
  --task-file examples/task.json \
  --references-file examples/reference_pool.json \
  --output-dir outputs \
  --mock
```

Model-backed mode (requires compatible LLM credentials in environment):

```bash
uv run paperbanana run \
  --task-file examples/task.json \
  --references-file examples/reference_pool.json \
  --output-dir outputs \
  --no-mock \
  --model-name gpt-4o-mini
```

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
