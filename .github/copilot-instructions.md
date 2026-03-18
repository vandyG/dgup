# dgup — Project Guidelines

**Daily Gas Usage and Prediction** at Scot Forge. Ingests raw gas consumption data, processes it with Polars, and produces analyses and forecasts for an industry stakeholder audience.

## Tech Stack

- **Package manager**: `uv` — always use `uv` for installs, never `pip` directly
- **Data**: `polars` (GPU-accelerated), `pyarrow` for Parquet I/O, `fastexcel` for Excel ingestion
- **Visualization**: `altair` (declarative, in marimo notebooks)
- **Notebooks**: `marimo` — **never Jupyter**; new interactive work lives in `notebooks/` using marimo
- **Build**: `pdm-backend`; task runner is `duty` via `scripts/make`

## Build and Test

```bash
scripts/make setup          # install all dependency groups
scripts/make format         # ruff --fix-only then ruff format
scripts/make check          # lint + types + docs + API (all checks)
scripts/make check-quality  # ruff lint only
scripts/make check-types    # mypy only
scripts/make test coverage  # pytest -n auto + coverage HTML report
```

Run `scripts/make check` and `scripts/make test coverage` before committing.

## Architecture

```
src/dgup/                 ← public package; __init__.py is the only public surface
    __init__.py           ← all public symbols MUST be re-exported here in __all__
    __main__.py           ← python -m dgup entry
    py.typed              ← PEP 561 marker
    _internal/            ← all implementation; never import _internal from outside
        __init__.py       ← intentionally empty; must have NO docstring
        cli.py            ← argparse CLI: get_parser() + main(args)
        debug.py          ← environment introspection

data/
    silver/               ← canonical processed data (Parquet files only)
    docs/                 ← meeting notes and stakeholder documents

notebooks/                ← marimo notebooks for EDA and stakeholder presentation
```

## Coding Conventions

### Public API

- Every public symbol in `_internal` **must** be re-exported in `src/dgup/__init__.py` `__all__`; `tests/test_api.py` enforces this with `griffe`
- `_internal/__init__.py` must stay empty with **no module docstring** (enforced by test)
- Public modules in `src/dgup/` (not `_internal`) **must** have module docstrings

### Style

- Line length: **120**
- **Google-style docstrings** everywhere public
- **No relative imports** — always use absolute imports (`from dgup._internal.cli import …`)
- `from __future__ import annotations` at the top of every source file
- No `print()` outside `cli.py` and `debug.py`

### Testing

- Import from `dgup`, never from `dgup._internal` in tests
- CLI tests: call `main([...])` and check return code
- Flags that call `sys.exit` (e.g., `-V`, `--debug-info`) tested with `pytest.raises(SystemExit)`

### Data and Results

- All results and intermediate data saved to `data/` (Parquet format preferred)
- **Every incremental change to data logic is a new function** — do not overwrite existing logic; old functions must remain so fallback is always possible

### Reusable Code

- All reusable processing, modeling, and utility logic lives in `src/dgup/_internal/`; export through `src/dgup/__init__.py`
- Notebooks call into the package — they do not contain reusable logic themselves

## Marimo Notebooks

Notebooks target a **non-technical industry stakeholder**; make logic and findings easy to explain:

- Include narrative markdown cells explaining each step in plain language
- Use `altair` for all charts (declarative, reproducible)
- Every output (dataframe, chart, stat) should be visible in the notebook
- Save computed results to `data/` so they can be referenced without re-running the notebook
- New notebooks go in `notebooks/` alongside existing ones

## Dependencies

Use `uv` to manage dependencies. New runtime dependencies go in `[project] dependencies`; dev/tooling in the appropriate `[dependency-groups]` section of `pyproject.toml`. Example:

```bash
uv add polars          # runtime dependency
uv add --group dev some-dev-tool
```
