---
description: "Use when creating, editing, or reviewing marimo notebooks in notebooks/. Enforces marimo format, altair charts, stakeholder narrative structure, and package-based reusable logic."
applyTo: "notebooks/*.py"
---

# Notebook Standards

## Format — marimo only

- All notebooks are **marimo** Python files (`.py` with `import marimo` and `app = marimo.App(...)`).
- Never create or reference `.ipynb` files. Never use `jupyter`, `ipykernel`, or `nbformat`.
- Each logical unit of computation is a separate `@app.cell` function. Cells are reactive — outputs of one cell are parameters of the next.
- Run notebooks with `marimo run notebooks/<name>.py` or edit with `marimo edit notebooks/<name>.py`.

## Charts — altair only

- Use **`altair`** for all charts — no matplotlib, seaborn, plotly, or bokeh.
- Charts must be returned from their cell so they render in the notebook output.
- Use `alt.Chart(...).mark_*().encode(...)` declarative style; avoid procedural mutations.
- Prefer `mo.ui.altair_chart(chart)` when the chart needs to be interactive (selection, filter).
- Always set meaningful axis titles and chart titles via `.properties(title=..., ...)`.

## Stakeholder Narrative Structure

Notebooks target a **non-technical industry audience at Scot Forge**. Every notebook must follow this cell order:

1. **Title + purpose** — markdown cell explaining what this notebook shows and why it matters to the business.
2. **Imports** — single cell with all imports.
3. **Data loading** — load from `data/silver/*.parquet` via `polars`; show a sample or schema so the audience can see the raw data.
4. **Step-by-step analysis** — each transformation or calculation is its own cell, preceded by a **markdown cell in plain business language** explaining what is being done and why.
5. **Key findings** — a markdown cell summarising the main takeaway in one or two sentences before each chart.
6. **Save results** — at the end, write any computed outputs back to `data/` in Parquet format.

## Reusable Logic — keep in the package

- Notebooks must **not** define reusable functions or classes. All logic that could be reused belongs in `src/dgup/_internal/` and exposed via `src/dgup/__init__.py`.
- Notebooks import from `dgup`, never from `dgup._internal` directly.
- Exploratory, one-off cells (quick filters, ad hoc reshaping) are acceptable in notebooks, but only if they are not worth packaging.

## Data I/O

- Load data with `polars.scan_parquet(...)` (lazy frame) and `.collect()` only when needed.
- Save results with `df.write_parquet("data/silver/<name>.parquet")` or `df.write_parquet("data/<name>.parquet")`.
- Never hardcode absolute paths — use paths relative to the workspace root or resolve via `pathlib.Path(__file__).parent.parent / "data"`.

## Dependencies

- Do not add dependencies inside notebook script headers (`# /// script ... ///`) unless the notebook is meant to be run as a standalone script with `uv run`.
- For notebooks that are part of the project, rely on the project's `pyproject.toml` environment managed by `uv`.

## Example cell pattern

```python
@app.cell
def _(mo):
    mo.md("""
    ## Monthly Gas Usage

    This chart shows total gas consumption per month. Spikes often correspond to
    high-temperature forge schedules or seasonal demand — worth reviewing with operations.
    """)
    return


@app.cell
def _(alt, mo, monthly_df):
    chart = (
        alt.Chart(monthly_df)
        .mark_bar()
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("total_usage:Q", title="Gas Usage (MCF)"),
            tooltip=["month:T", "total_usage:Q"],
        )
        .properties(title="Monthly Gas Consumption", width=700, height=300)
    )
    return mo.ui.altair_chart(chart)
```
