# Reproduce the paper from cached model output, with no API calls.
#
#   make all      everything below, in order, then the PDF
#   make score    THE ONLY TARGET THAT CALLS A MODEL API. Not part of `all`.
#   make test     the test suite
#
# The split is deliberate: `all` must be runnable by a reader who has the
# repository and output/ but no API key and no budget. Scoring is a separate,
# explicit act.

PY      := /usr/local/bin/python3
OUT     := output
EXP     := $(OUT)/experiments
CHUNKS  := $(OUT)/chunk_level_results.csv
TECTONIC := tectonic

.PHONY: all score test clean extraction coverage battery mixed truncation \
        reliability encoder discriminant numbers figures pdf check-chunks help

help:
	@echo "make all       — regenerate every table, figure and the PDF (no API calls)"
	@echo "make score     — run the scoring pipeline (CALLS THE MODEL API, ~2,001 calls)"
	@echo "make test      — run the test suite"
	@echo "make clean     — remove generated artifacts (never the model output cache)"

all: extraction coverage battery mixed truncation reliability encoder discriminant numbers figures pdf
	@echo ""
	@echo "Done. paper/main.pdf is current with output/."

# ── Offline from the cached raw filings: no model calls, no EDGAR requests ────
extraction:
	$(PY) -m analysis.extraction_report > /dev/null
	@echo "  extraction_report.{txt,csv}"

coverage:
	$(PY) -m analysis.coverage_diagnostic > /dev/null
	@echo "  coverage_diagnostic.{txt,csv}"

# ── Everything below needs window-level scored output ────────────────────────
check-chunks:
	@test -f $(CHUNKS) || { \
	  echo "ERROR: $(CHUNKS) is missing."; \
	  echo "It is produced by the scoring run (\`make score\`), which is the only"; \
	  echo "step that calls a model API. Every other target reads its output."; \
	  exit 1; }

battery: check-chunks
	$(PY) -m analysis.window_battery > /dev/null
	@echo "  experiments/window_battery/"

mixed: check-chunks
	$(PY) -m analysis.mixed_model > /dev/null
	@echo "  experiments/mixed_model/"

truncation: check-chunks
	$(PY) -m experiments.truncation_contrast.run > /dev/null
	@echo "  experiments/truncation_contrast/"

reliability: check-chunks
	$(PY) -m experiments.reliability.run > /dev/null
	@echo "  experiments/reliability/"

# Downloads encoder weights on first run; still makes no model/API calls.
encoder: check-chunks
	$(PY) -m experiments.encoder_robustness.run > /dev/null
	@echo "  experiments/encoder_robustness/"

# Needs the Loughran-McDonald Master Dictionary (scripts/fetch_lm_dictionary.sh).
discriminant: check-chunks
	$(PY) -m experiments.discriminant_validity.run > /dev/null
	@echo "  experiments/discriminant_validity/"

# ── Paper ────────────────────────────────────────────────────────────────────
numbers:
	$(PY) -m paper.make_numbers
	@echo "  paper/numbers.tex"

figures:
	$(PY) -m paper.make_figures
	@echo "  paper/figures/"

pdf: numbers figures
	cd paper && $(TECTONIC) -X compile main.tex --keep-logs
	@echo "  paper/main.pdf"
	@grep -ci "TODO" paper/numbers.tex > /dev/null && \
	  echo "  NOTE: numbers.tex still has \\todo macros — unfilled values are visible in the PDF." || true

test:
	$(PY) -m pytest tests/ -q

# ── Scoring: the only API-calling target ─────────────────────────────────────
# Resumable: chunk_cache.json checkpoints per firm, so an interrupted run
# re-scores only the firms it had not finished.
score:
	@echo "This calls the model API (~2,001 persona calls). Ctrl-C now to abort."
	@sleep 3
	$(PY) main.py --firm-set full --skip-proxies --provider openrouter --agent-sleep 0.5

clean:
	rm -f paper/numbers.tex paper/main.pdf paper/main.log
	rm -rf paper/figures
	rm -f $(OUT)/extraction_report.* $(OUT)/coverage_diagnostic.*
	rm -rf $(EXP)
	@echo "Removed generated artifacts. Model output (chunk_level_results.csv,"
	@echo "chunk_cache.json, intermediate_results.json, embedding_cache.json) is kept —"
	@echo "it cost money and `make all` depends on it."
