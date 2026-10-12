import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Inference provider ─────────────────────────────────────────────────────
    # All supported providers expose an OpenAI-compatible /chat/completions
    # endpoint, so the pipeline talks to them through one client.
    #
    # This indirection exists because the project got blocked on it: in September
    # 2026 Groq suspended Developer-tier upgrades, which left the complete-section
    # re-run stranded on a 200k-token/day free tier against a hard deadline.  The
    # experiment should not be hostage to one vendor's billing page.
    #
    # Switching provider changes the served model and therefore the results —
    # bump CACHE_VERSION and tell the advisor when you do.
    API_PROVIDER: str = os.environ.get("API_PROVIDER", "groq")

    PROVIDERS: dict = {
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "key_env": "GROQ_API_KEY",
            "model": "openai/gpt-oss-20b",
            # free tier: 200k tokens/day, 8k TPM, 1k requests/day
            "sleep": 30.0,
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "key_env": "OPENROUTER_API_KEY",
            "model": "openai/gpt-oss-20b",   # same weights as the Groq run
            "sleep": 0.5,
            # ── Backend pinning (methodologically required, not an optimisation) ──
            # 14 backends serve gpt-oss-20b through OpenRouter at DIFFERENT
            # quantizations (fp4, fp8, bf16, undeclared), and the default routing
            # load-balances across them.  The design claim of this study is that
            # the model is held fixed and only the analytical lens varies; letting
            # 2,000 calls scatter across fp4 and bf16 backends would make part of
            # the measured disagreement a quantization artifact and the claim
            # false.  So: one backend, bf16 (the reference precision, no
            # quantization loss), and no silent fallback.
            #
            # allow_fallbacks=False means a backend outage fails the call rather
            # than rerouting it.  That is the behaviour we want — a failed call is
            # dropped by the partial-failure rule, whereas a silently rerouted one
            # would contaminate the sample.
            "extra_body": {
                "provider": {
                    # DekaLLM, not DeepInfra. Of the 12 endpoints serving this
                    # model through OpenRouter exactly TWO are bf16 — DeepInfra
                    # and DekaLLM, at near-identical price ($0.030 vs $0.029 per
                    # M input) — and on 2026-10-04 DeepInfra was rate-limiting
                    # the model upstream hard enough to exhaust the retry ceiling
                    # (3/3 probe calls failed; DekaLLM 2/3 at 2.2 s latency).
                    # Re-pinning preserves the design claim exactly as written:
                    # one model, reference precision, ONE backend. Switching
                    # WHICH bf16 backend is pinned is not the same as mixing
                    # quantizations, which is what the pin exists to prevent.
                    # Back to DeepInfra on 2026-10-05. DekaLLM degraded mid-run
                    # (3/5 success at 22-27s, and 2.3% of calls returning HTTP 200
                    # with an empty completion, which silently costs a window one
                    # persona), while DeepInfra had recovered from the congestion
                    # that caused the original switch away from it (5/5 at 5.2s).
                    # The 14 firms scored on DekaLLM were DISCARDED rather than
                    # merged, so the dataset has exactly one serving backend.
                    "order": ["DeepInfra"],
                    "allow_fallbacks": False,
                    "quantizations": ["bf16"],
                },
            },
        },
        "cerebras": {
            "base_url": "https://api.cerebras.ai/v1",
            "key_env": "CEREBRAS_API_KEY",
            "model": "gpt-oss-120b",         # Cerebras does not serve the 20b
            # free tier: 1M tokens/day, 30k TPM, 5 requests/min
            "sleep": 12.0,
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "key_env": "OLLAMA_API_KEY",     # unused; any value works
            "model": "gpt-oss:20b",
            "sleep": 0.0,
        },
    }

    @property
    def provider(self) -> dict:
        if self.API_PROVIDER not in self.PROVIDERS:
            raise ValueError(
                f"Unknown API_PROVIDER {self.API_PROVIDER!r}. "
                f"Choose one of: {', '.join(self.PROVIDERS)}"
            )
        return self.PROVIDERS[self.API_PROVIDER]

    @property
    def EXTRA_BODY(self) -> dict:
        """Provider-specific request fields (e.g. OpenRouter backend pinning)."""
        return self.provider.get("extra_body", {})

    @property
    def API_BASE_URL(self) -> str:
        return self.provider["base_url"]

    @property
    def API_KEY(self) -> str:
        return os.environ.get(self.provider["key_env"], "")

    # Back-compat: existing code reads GROQ_API_KEY.
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

    # Single-model design: all three personas use the same model so that
    # disagreement is purely persona-driven, not model-architecture-driven.
    # Change MODEL_NAME only intentionally; bump CACHE_VERSION when you do
    # so that intermediate_results.json is invalidated automatically.
    # NOTE: llama-3.1-8b-instant was removed from Groq in August 2026 (404 on all calls).
    # Switched to openai/gpt-oss-20b — tested August 2026, follows SCORE/REASONING/KEY_FACTORS
    # schema cleanly; qwen/qwen3.6-27b rejected (inserts <think> block that breaks parser).
    # The training cutoff for gpt-oss-20b is NOT published, so nothing can be
    # asserted about it here.  TRAINING_CUTOFF below is a conservative lower bound
    # retained from the previous model, not this model's cutoff; all 59 filings in
    # the sample are dated after it (earliest 2024-08-01).
    # Overridden by the active provider entry; kept as the default for "groq".
    MODEL_NAME: str = "openai/gpt-oss-20b"
    GROQ_MODEL: str = MODEL_NAME  # backward-compat alias

    # Training cutoff: only filings dated AFTER this date are processed.
    # openai/gpt-oss-20b's training cutoff is not published, so this date is NOT that
    # model's cutoff and must not be described as one.  It is a conservative lower
    # bound inherited from llama-3.1-8b-instant (stated cutoff ~Dec 2023 plus a
    # six-month buffer), retained because it costs nothing: all 59 filings in the
    # 30-firm sample are dated after it, the earliest being 2024-08-01 (HAYN 10-Q;
    # verified against output/coverage_diagnostic.csv, filing_date column).  Whether
    # the filings postdate THIS model's training data is unverifiable until a cutoff
    # is published; update this date if one ever is.
    TRAINING_CUTOFF: str = "2024-07-01"
    FILING_CUTOFF: str = TRAINING_CUTOFF  # backward-compat alias

    # Local sentence-transformers model (downloaded once, ~90 MB, then cached)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── Scoring scale ──────────────────────────────────────────────────────────
    # Agents are prompted to output an integer 0–100 (50 = neutral).
    # The parser divides by 10 to produce a 0.0–10.0 continuous measure.
    # Using a 100-point integer scale avoids the model's tendency to output
    # boundary integers (0 or 1) when asked for decimals on a 10-point scale.
    SCORING_SCALE: int = 100   # prompt asks for 0-100; divide raw by 10

    # ── Agent call parameters ──────────────────────────────────────────────────
    # Temperature 0.3 is a deliberate methodological choice: it introduces
    # enough variation that personas explore the response space but not so
    # much that outputs become incoherent. See calibration.py for the
    # temperature-noise baseline test that validates this choice.
    TEMPERATURE: float = 0.3
    MAX_AGENT_TOKENS: int = 1500   # generous headroom for V3 factor-decomposition schema

    # ── Cache versioning ───────────────────────────────────────────────────────
    # Increment CACHE_VERSION whenever the scoring scale, model, or prompts
    # change in a way that makes old intermediate results incompatible with
    # new results. The pipeline will refuse to mix cache generations.
    CACHE_VERSION: str = "v5"  # v1=0-10; v2=0-100; v3=2000-word MD&A-first; v4=gpt-oss-20b; v5=complete sections, chunk-level scoring

    # EDGAR
    EDGAR_USER_AGENT: str = "Research Project aryanchhabra6996@gmail.com"
    MAX_FILINGS_PER_TYPE: int = 1    # 1 × 10-Q + 1 × 10-K per firm (sufficient for pilot gradient)
    # ── Section coverage (Round 8, September 2026) ─────────────────────────────
    # MAX_WORDS_PER_FILING is DEPRECATED and no longer applied.  Through Round 7 a
    # global word cap was used to make firms comparable.  Measured on the 30-firm
    # sample it did the opposite: it truncated 59 of 59 filings, and the fraction
    # of each narrative section that survived was correlated with firm size
    # (Spearman rho = -0.301, p = 0.0205 vs tier rank; median fraction read 15.7%
    # for large-caps, 17.0% for mid-caps, 27.5% for small-caps).  Coverage was
    # therefore a function of the independent variable.
    #
    # MD&A and Risk Factors are now extracted COMPLETE, and comparability is
    # enforced at the chunk level: each section is split into fixed windows of
    # CHUNK_WORDS words, every persona scores every window, and filing-level
    # disagreement is the unweighted mean over windows.  Filing length becomes an
    # observable control (n_chunks) instead of a hidden difference in how much of
    # the document was read.
    MAX_WORDS_PER_FILING: int = 0     # deprecated; 0 = no truncation

    # 1200 words ≈ 1,600 tokens.  With the persona system prompt (~600 tokens) and
    # MAX_AGENT_TOKENS output, a single call stays near ~3,700 tokens — inside the
    # 6k free-tier per-request ceiling, so the design does not depend on a paid tier.
    CHUNK_WORDS: int = 1200
    # A trailing remainder below this is merged into the previous chunk rather than
    # scored on its own; tiny chunks have little to disagree about and short
    # sections would produce proportionally more of them.
    MIN_TAIL_WORDS: int = 400
    # Only the two sections the hypothesis is about are scored, so the scored
    # corpus has identical composition for every firm.  Item 7A / Item 3 is still
    # extracted and cached, but not sent to the personas.
    SECTIONS_TO_ANALYZE: tuple = ("mda", "risk")

    MIN_EXTRACTION_WORDS: int = 300   # fewer words → extraction failed, use fallback

    # Rate limiting.  This is only the default: apply_provider() overwrites it with
    # the selected provider's "sleep" entry, so the pacing travels with the endpoint
    # instead of being hand-tuned per run.  30.0 is the free-tier value (Groq free:
    # 200k tokens/day, 8k TPM, where the daily token cap binds long before the
    # per-minute one); 0.5 is the value for a paid endpoint, where throughput is not
    # the binding constraint.
    # Per-call size under the chunk-level design: one CHUNK_WORDS=1200 window plus
    # the persona system prompt and the model's output is ~2,700 tokens in practice
    # (~3,700 worst case if MAX_AGENT_TOKENS is fully used), inside the 6k free-tier
    # per-request ceiling — the design does not require a paid tier, it only takes
    # weeks on the free one.
    # Round 8 scale: 638 windows → 1,914 persona calls over the 59 selected filings.
    # At 30s that is ~16 hours of sleep alone.  CAVEAT: 638/1,914 is measured BEFORE
    # the extraction fix landing in this same round, which lengthens the sections of
    # 9 filings, so treat it as a floor and not a final figure — re-derive the count
    # from len(chunk_level_results.csv) after the run and quote that everywhere.
    # If 429 rate-limit errors appear, raise this above the provider default.
    AGENT_SLEEP_SECONDS: float = 30.0
    EMBEDDING_SLEEP_SECONDS: float = 2.0
    EDGAR_SLEEP_SECONDS: float = 0.5

    # Retry
    # Raised from 3 (Round 8): with OpenRouter backend pinning and fallbacks
    # disabled, a congested shared pool must be waited out rather than routed
    # around, so transient 429s need more attempts to survive a long run.
    # Raised 6 -> 10 on 2026-10-04. This model is intermittently rate-limited
    # upstream, and with allow_fallbacks=False a congested backend must be waited
    # out rather than routed around. Exhausting the ceiling drops the window under
    # the partial-failure rule, so the ceiling has to be high enough that transient
    # congestion costs time rather than data.
    MAX_RETRIES: int = 10
    RETRY_BACKOFF_BASE: float = 2.0

    # Paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / "data"
    FILINGS_DIR: Path = DATA_DIR / "filings"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    PLOTS_DIR: Path = OUTPUT_DIR / "plots"
    LOG_FILE: Path = BASE_DIR / "pipeline.log"
    INTERMEDIATE_FILE: Path = OUTPUT_DIR / "intermediate_results.json"
    CIK_CACHE_FILE: Path = BASE_DIR / ".cik_cache.json"

    # ── Design documentation ───────────────────────────────────────────────────
    # Included here as a machine-readable constant and as documentation.
    # Paste into any paper draft or advisor memo where framing is needed.
    # The model and provider are interpolated from the live config, not hardcoded:
    # this string is printed into output/summary.txt, which goes to the advisor, and
    # a hardcoded "via Groq" would misstate the design under --provider openrouter.
    # Implemented as a descriptor rather than a @property because write_summary()
    # in main.py reads it off the CLASS (Config.DESIGN_DESCRIPTION), where a
    # property object would print its own repr into the summary.
    class _DesignDescription:
        def __get__(self, obj, objtype=None):
            src = obj if obj is not None else objtype
            return (
                f"Prompt ensemble: one model ({src.MODEL_NAME} via {src.API_PROVIDER}), "
                "three analytical persona prompts (value, sentiment, fundamentals) — "
                "one model, varying only the analytical lens. No inter-persona "
                "interaction. Disagreement is induced by analytical perspective: MD&A "
                "and Risk Factors are extracted as complete sections and split into "
                f"fixed {src.CHUNK_WORDS}-word windows, and each persona scores every "
                "window in isolation. This design holds model architecture constant so "
                "that measured disagreement reflects filing content and analytical "
                "framing, not model-to-model differences."
            )

    DESIGN_DESCRIPTION = _DesignDescription()

    # ── Statistical thresholds ─────────────────────────────────────────────────
    ALPHA: float = 0.05                    # family-wise alpha before Bonferroni
    LOW_RES_THRESHOLD: float = 0.50        # flag measure if mode > 50% of obs
    EMBEDDING_FLAT_THRESHOLD: float = 0.01 # raise if max-min spread ≤ this

    @property
    def effective_cache_version(self) -> str:
        """
        Cache key including everything that changes what a cached row MEANS:
        the scoring window and the model that produced it.

        Results scored at different CHUNK_WORDS are not interchangeable — the
        window is part of the measurement, not an implementation detail — so a
        run at 2000 must not silently reuse rows scored at 1200.

        The model is in the key for the same reason and a sharper one: with the
        provider now swappable, running against Cerebras (gpt-oss-120b) would
        otherwise happily reuse rows scored by Groq (gpt-oss-20b) and mix two
        models inside one set of tier means.  That is the failure mode the
        partial-failure rule exists to prevent, one level up.
        """
        model_tag = re.sub(r"[^A-Za-z0-9]+", "", self.MODEL_NAME)[-14:]
        return f"{self.CACHE_VERSION}-c{self.CHUNK_WORDS}-{model_tag}"

    def apply_provider(self) -> None:
        """Point MODEL_NAME and the pacing at whichever provider is selected."""
        prov = self.provider
        self.MODEL_NAME = prov["model"]
        self.GROQ_MODEL = prov["model"]
        self.AGENT_SLEEP_SECONDS = prov["sleep"]

    def validate(self) -> None:
        if self.API_PROVIDER == "ollama":
            return  # local server needs no key
        if not self.API_KEY:
            env = self.provider["key_env"]
            raise EnvironmentError(
                f"{env} is not set for API_PROVIDER={self.API_PROVIDER!r}. "
                f"Add {env}=... to .env"
            )

    def ensure_dirs(self) -> None:
        for d in (self.FILINGS_DIR, self.PROCESSED_DIR, self.OUTPUT_DIR, self.PLOTS_DIR):
            d.mkdir(parents=True, exist_ok=True)
