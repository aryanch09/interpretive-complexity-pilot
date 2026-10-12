# Disagreement About Nothing

Code, cached model output and paper source for *Disagreement About Nothing: A Pre-Registered
Stress Test of a Persona-Ensemble Measure of 10-K Complexity* (AI4F '26 workshop, ACM ICAIF '26).

One fixed model (`openai/gpt-oss-20b`, a single bf16 backend) reads every 1,200-word window of
a filing's complete MD&A through three analytical personas. Persona disagreement is the mean
pairwise cosine distance between their rationales. The paper tests whether that measures
interpretive complexity. It does not: it rises on boilerplate, falls with concrete figures, and
is unreliable at the filing level. The pre-registered size test is null.

## Reproduce without spending anything

Every table, figure and number in the paper regenerates from the cached model output in
`output/` with no API calls:

```bash
pip install -r requirements.txt
bash scripts/fetch_lm_dictionary.sh   # Loughran-McDonald dictionary (not redistributed here)
make all                              # analyses, paper/numbers.tex, figures, PDF
python3 -m experiments.hayn_sensitivity.run
python3 -m paper.make_numbers_r9      # paper/numbers_r9.tex
python3 -m paper.make_figures_r9
cd paper && latexmk -pdf main_v2.tex  # the submitted paper
```

`make score` is the only target that calls a model API (about 2,001 calls). It needs a key in
`.env` (see `.env.example`).

## Where things are

| Path | What |
|---|---|
| `ANALYSIS_PLAN.md` | Pre-registration, committed before any filing was scored |
| `DEVIATIONS.md` | Every departure from the plan, and whether results had been seen |
| `data/edgar.py`, `analysis/chunking.py` | Complete-section extraction and windowing |
| `agents/` | Persona prompts and the pinned-backend runner |
| `analysis/`, `experiments/` | Window-count battery, mixed model, reliability, placebo, cross-model, decomposition |
| `output/chunk_level_results.csv` | One row per scored window, with all three rationales |
| `paper/main_v2.tex` | Submitted paper; `paper/main.tex` is an earlier, superseded draft |

Raw EDGAR filings are not included (`data/filings/` is re-fetchable from EDGAR); the extracted
sections are in `data/processed/`.

## Note on history

This repository was exported with a fresh history. The working repository's commit hashes
cited in the paper (`fec1eb2`, `c22f5dd`) refer to the pre-registration commits in that
working history, which also contains private correspondence and is not published.
