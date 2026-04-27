Status: approved for implementation on 2026-04-25.


## Plan: Scot Forge Final Report

Convert the existing narrative draft into a self-contained APA-style LaTeX report in /home/vandy/work/dgup/Report/latex/report.tex, using /home/vandy/work/dgup/Report/report.txt as the seed text but re-grounding every section in code, weekly progress notes, optimization outputs, and a compact bibliography. Keep the main body within 5 pages by prioritizing 3–4 high-value visuals and one KPI table, then move the rest of the generated charts to an appendix.

**Steps**
1. Build the report scaffold in /home/vandy/work/dgup/Report/latex/report.tex as a standalone article-style document with the requested section order, author/title metadata, appendix sectioning, and an APA-style bibliography strategy that stays self-contained in one file. This blocks the rest of the writing because the page-budget and citation mechanics need to be fixed first.
2. Create a source-to-section evidence map before drafting prose: map each core claim in the abstract, introduction, methodology, results, conclusion, and lessons sections to its source in /home/vandy/work/dgup/Report/report.txt, /home/vandy/work/dgup/Report/problem_statement.md, /home/vandy/work/dgup/Report/WeeklyProgress, /home/vandy/work/dgup/Report/Weekly Report_4 (1).md, /home/vandy/work/dgup/Report/Weekly_Report_7 (1).md, /home/vandy/work/dgup/Report/forecast_optimize.py, /home/vandy/work/dgup/Report/optimization_viz.py, and /home/vandy/work/dgup/Report/gas_optimization_results_2022_2024_v4 1.csv. This can run in parallel with step 1.
3. Draft the EDA section from /home/vandy/work/dgup/Report/eda.py, selecting only the visuals and insights that directly justify the later modeling choices. Keep the main body focused on the multi-year time-series trend, seasonality heatmap, lag/correlation evidence, and stream-level distributional differences; send secondary diagnostics such as full correlation matrices, ACF/PACF, and detailed constraint dashboards to the appendix. Avoid overclaiming hypothesis testing because the current EDA code is narrative-heavy and does not document formal statistical tests.
4. Draft the forecasting methodology section by combining the model chronology from /home/vandy/work/dgup/Report/WeeklyProgress and /home/vandy/work/dgup/Report/Weekly Report_4 (1).md with the final implementation in /home/vandy/work/dgup/Report/forecast_optimize.py. Explicitly explain the pivot from forecasting-only evaluation to decision-aware planning, the expanding-window monthly retraining logic, the operational t-2 availability constraint, the engineered feature set in build_features / FEATURES, and why the final pipeline used quantile forecasts rather than a single point estimate.
5. Draft the optimization and results sections from /home/vandy/work/dgup/Report/forecast_optimize.py, /home/vandy/work/dgup/Report/optimization_viz.py, /home/vandy/work/dgup/Report/penalty_calc.md, /home/vandy/work/dgup/Report/gas_optimization_results_2022_2024_v4 1.csv, and /home/vandy/work/dgup/Report/Weekly_Report_7 (1).md. Explain the month-specific daily limits and EOM bounds, the rolling 60-day linear program, adaptive injection/withdrawal buffers, mid-month steering target, and the penalty calculation logic. Quantify the final business impact with one consistent KPI set: daily violations reduced by about 264, monthly violations roughly unchanged near 15, and tier counts improved from 71→22, 47→25, and 302→109.
6. Generate the figure assets from the code in this repository because no static report-ready figures are currently present. Export the selected main-body charts from /home/vandy/work/dgup/Report/eda.py and /home/vandy/work/dgup/Report/optimization_viz.py first, then export the remaining supporting visuals for the appendix. This depends on steps 3 and 5 for prioritization, but the actual export work for EDA and optimization visuals can proceed in parallel once those choices are fixed.
7. Write the conclusion and required lessons-learned subsection using /home/vandy/work/dgup/Report/lessons.md plus the final KPI narrative. Keep the conclusion business-facing: forecasting mattered because it enabled a safer nomination plan, and the main lesson is that decision quality under uncertainty is more important than point-forecast accuracy alone. Preserve the instructor suggestions as a short closing subsection rather than scattering them across the report.
8. Assemble the bibliography in APA style inside /home/vandy/work/dgup/Report/latex/report.tex so the deliverable remains self-contained. Use project/internal sources for the case-specific material and add external method references gathered from web research: scikit-learn documentation for GradientBoostingRegressor and QuantileRegressor, SciPy documentation for linprog, the HiGHS citation guidance and Huangfu & Hall (2018) solver paper, and the Friedman gradient boosting reference surfaced by scikit-learn. If needed during implementation, add one stable quantile-regression paper citation only if a reliable public citation source is fetched.
9. Run final QA: compile the LaTeX report, verify that the main narrative stays within 5 pages excluding bibliography and appendix, ensure every table/figure is cited in text, and cross-check that the abstract, results, and conclusion all use the same KPI values. Also confirm that tariff language is paraphrased carefully from /home/vandy/work/dgup/Report/penalty_calc.md rather than copied verbatim.

**Relevant files**
- `/home/vandy/work/dgup/Report/latex/report.tex` — final standalone report; needs full structure, prose, bibliography, figure/table references, and appendix.
- `/home/vandy/work/dgup/Report/report.txt` — current narrative draft to compress, fact-check, and adapt into the final LaTeX version.
- `/home/vandy/work/dgup/Report/problem_statement.md` — source for the business framing, formal objectives, and consulting-style problem definition.
- `/home/vandy/work/dgup/Report/eda.py` — primary EDA source; reuse the time-series explorer, seasonal summaries, lag heatmaps, distributions, and constraint-related visuals selectively.
- `/home/vandy/work/dgup/Report/forecast_optimize.py` — primary implementation source for build_features, FEATURES, monthly retraining, blended_predict, solve_lp, calculate_penalties, and the rolling optimization logic.
- `/home/vandy/work/dgup/Report/optimization_viz.py` — source for delivery-versus-optimized visuals, daily/monthly penalty dashboards, and tier comparison charts.
- `/home/vandy/work/dgup/Report/gas_optimization_results_2022_2024_v4 1.csv` — final backtest output containing quantiles, optimized deliveries, storage levels, and violation flags.
- `/home/vandy/work/dgup/Report/WeeklyProgress` — chronology of project evolution, stakeholder-driven scope changes, and milestone framing.
- `/home/vandy/work/dgup/Report/Weekly Report_4 (1).md` — forecasting model comparison table and early walk-forward evaluation metrics.
- `/home/vandy/work/dgup/Report/Weekly_Report_7 (1).md` — final tier-wise improvement summary and KPI checkpoint.
- `/home/vandy/work/dgup/Report/penalty_calc.md` — tariff/regulatory basis for the contract constraints and cash-out penalty structure; should be paraphrased and cited carefully.
- `/home/vandy/work/dgup/Report/lessons.md` — required lessons-learned and instructor-suggestions content.

**Verification**
1. Compile /home/vandy/work/dgup/Report/latex/report.tex with the local LaTeX toolchain already implied by the existing report.aux/report.log artifacts; rerun until references and bibliography resolve cleanly.
2. Check that the main body stays within 5 pages, with bibliography and appendix beginning after the page-limited narrative.
3. Cross-check every reported KPI against /home/vandy/work/dgup/Report/Weekly_Report_7 (1).md and /home/vandy/work/dgup/Report/gas_optimization_results_2022_2024_v4 1.csv so there is one authoritative metric set.
4. Confirm each selected figure or table is explicitly referenced in the text and that no chart included in the main body is redundant with an appendix chart.
5. Review the final draft for unsupported claims, especially around “hypothesis testing,” and keep any formal hypothesis language limited to what the report can actually defend from the available evidence.

**Decisions**
- Use a self-contained article-style report in /home/vandy/work/dgup/Report/latex/report.tex.
- Use APA / author-year bibliography formatting.
- Reuse /home/vandy/work/dgup/Report/report.txt as the narrative seed, but treat code/data files as the final source of truth.
- Assume figure assets must be generated from the code in this repository.
- Keep only the highest-value visuals and one KPI table in the 5-page main body; move secondary outputs to the appendix.
- Include the lessons learned and instructor suggestions at the end of the conclusion section, as requested.
- Describe hypotheses and findings in a consulting-report style, but do not claim formal statistical hypothesis tests unless additional evidence is added during implementation.

**Further Considerations**
1. Because /home/vandy/work/dgup/Report/eda.py and /home/vandy/work/dgup/Report/optimization_viz.py are interactive marimo apps, the implementation pass should standardize figure export naming and captions before writing final LaTeX figure blocks.
2. If APA citation mechanics become awkward in a single-file document, only then consider splitting references into a separate .bib file; otherwise keep the submission centered on /home/vandy/work/dgup/Report/latex/report.tex.