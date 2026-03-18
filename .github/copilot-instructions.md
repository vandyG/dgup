# Project Guidelines

## Architecture

- Treat `src/dgup` as the public package surface.
- Keep implementation details in `src/dgup/_internal`; only expose names from `_internal` when they are intentionally part of the public API.
- Preserve the current CLI pattern: parser construction in `_internal/cli.py`, public exports from `src/dgup/__init__.py`, and `main(args: list[str] | None) -> int` returning an exit code.

## Build And Test

- Prefer the project wrapper commands in `scripts/make` or the matching VS Code tasks instead of ad hoc tool invocations.
- Use `uv` for Python dependency installation and environment synchronization; do not introduce `pip`, `poetry`, `pdm`, or other package-manager install workflows for this project.
- Setup environments with `scripts/make setup`.
- Format code with `scripts/make format`.
- Run the full validation suite with `scripts/make check`.
- Run tests with coverage using `scripts/make test coverage`.
- `scripts/make check` already covers Ruff, MyPy, strict MkDocs builds, and the API compatibility check; use it before finishing substantial changes.

## Code Style

- Match the existing Python style: `from __future__ import annotations`, absolute imports, complete type hints, and Google-style docstrings where docstrings are expected.
- Keep changes compatible with the strict Ruff and MyPy configuration in `config/ruff.toml` and `config/mypy.ini`.
- Do not add module docstrings inside `src/dgup/_internal`; `tests/test_api.py` enforces that internal modules remain undocumented at module level.

## Conventions

- If you promote a new public symbol, export it from `src/dgup/__init__.py`, update `__all__`, and account for the API checks in `tests/test_api.py`.
- If you change CLI behavior or user-facing output, update `tests/test_cli.py` and the relevant docs in `README.md` or `docs/`.
- If you add or rely on new environment variables, declare them in `.envrc` as part of the same change.
- Notebooks in `notebooks/` are exploratory artifacts with looser lint rules than package code; keep production changes in `src/` and `tests/` unless the task is explicitly notebook-focused.
- For notebook work in `notebooks/`, follow the existing marimo stack used in the repo: prefer `polars` for data loading and transformation, and prefer `altair` for visualization.
- In notebook tasks, avoid introducing `pandas` or other plotting libraries unless the task explicitly requires them or the existing notebook already depends on them.