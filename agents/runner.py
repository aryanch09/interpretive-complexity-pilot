"""
Prompt-ensemble runner: calls the configured provider for each analytical
persona and parses structured output.

This module implements a prompt ensemble: one model is called three times with
different analytical persona prompts (value, sentiment, fundamentals). The model
is held constant across all three calls; only the analytical lens varies.
Disagreement between outputs reflects how far the filing pulls different lenses
apart — not disagreement between independent reasoners, and not disagreement
between different model architectures. This is by design: holding the model
fixed isolates the effect of analytical perspective and controls for
model-to-model architectural differences.

Design notes:
- The model name and endpoint come from config.API_PROVIDER. Never hardcode them.
- Scoring scale: agents output an integer 0–100 (50 = neutral). The parser divides
  by 10 to produce a 0.0–10.0 continuous measure. This avoids the model's tendency
  to output boundary integers (0 or 1) when asked for decimals on a 0–10 scale.
- Temperature (config.TEMPERATURE = 0.3) is documented in config.py.
- Retries: exponential back-off up to config.MAX_RETRIES. All failures are logged
  with ticker, filing, and full traceback in pipeline.log.
- `failed` flag: True only when the API call itself exhausted all retries.
  A malformed response (score missing or out of range) sets parse_error=True,
  failed=False; score is None in both cases.
- Partial failure: if one of three persona calls fails, the other two still
  contribute to disagreement. The observation is flagged partial_failure=True.
"""

import logging
import re
import time
from typing import Optional

from openai import OpenAI

from agents.prompts import AGENTS
from config import Config

logger = logging.getLogger(__name__)


class EmptyCompletionError(RuntimeError):
    """HTTP 200 with no completion text. Transient backend failure; retryable."""


def parse_agent_output(text: str, agent_name: str) -> dict:
    """
    Extract structured fields from an agent response.

    Scoring: raw output is an integer 0–100; parser converts to 0.0–10.0.
    The 0–100 scale is the contract: every raw score is divided by 10. A raw
    score ≤ 10 is ambiguous (possible wrong scale), so it is still divided but
    flagged parse_error=True and logged.
    """
    result: dict = {
        "agent": agent_name,
        "score": None,
        "confidence": None,
        "reasoning": None,
        "key_factors": None,
        "raw_output": text,
        "parse_error": False,
        "failed": False,
    }

    # ── Score extraction ───────────────────────────────────────────────────────
    # Handles: SCORE: 73   SCORE: [73]   SCORE: 7.3 (legacy)   SCORE: [7.3]
    # Search from the LAST occurrence of SCORE to avoid matching text inside REASONING.
    score_matches = list(re.finditer(r"SCORE\s*:\s*\[?\s*(-?\d+\.?\d*)\s*\]?", text, re.IGNORECASE))
    if score_matches:
        # Use the last match — in V1 the score is first, but REASONING may repeat
        # "SCORE" in prose; taking the last avoids false positives from reasoning text.
        # In V2 SCORE comes after KEY_FACTORS, also captured correctly by last match.
        score_match = score_matches[-1]
        raw_score = float(score_match.group(1))

        # The 0–100 scale is the contract, so divide unconditionally.
        converted = round(raw_score / 10.0, 2)
        if 0.0 <= converted <= 10.0:
            result["score"] = converted
            if 0.0 < raw_score <= 14.0:
                # A low raw score is EXPECTED, not ambiguous: the persona prompt's
                # own calibration guide defines "0-14: overwhelmingly negative", and
                # risk-factor windows legitimately land there. The smoke test saw a
                # sentiment persona write "overwhelmingly negative" and score 10,
                # which is 10/100 = 1.0 on the analysis scale.
                #
                # This is logged, not flagged: parse_error must keep meaning "we
                # could not parse the output", so that it stays usable as a quality
                # column. Setting it here would mark correctly parsed data as bad.
                #
                # Note for the record: the retired parser treated any raw score <= 10
                # as a legacy 0-10 value and recorded it UNDIVIDED, so that same
                # persona was scored 10.0 — the maximum POSITIVE value — while its
                # reasoning said "overwhelmingly negative". Silent, with
                # parse_error=False, and biased towards risk-factor windows, which
                # are uniformly negative.
                logger.info(
                    "%s scored %g/100 -> %.2f (low but in-band; prompt defines "
                    "0-14 as overwhelmingly negative)",
                    agent_name, raw_score, converted,
                )
        else:
            result["parse_error"] = True
            logger.warning(
                "parse_error for %s — raw SCORE %g / 10 = %.2f outside [0, 10]",
                agent_name, raw_score, converted,
            )

    # ── Confidence ─────────────────────────────────────────────────────────────
    conf_m = re.search(r"CONFIDENCE\s*:\s*\[?\s*([0-9.]+)\s*\]?", text, re.IGNORECASE)
    if conf_m:
        raw_conf = conf_m.group(1)
        try:
            conf = float(raw_conf)
        except ValueError:
            # e.g. "CONFIDENCE: 0.9." — a model ending the line with a sentence
            # period. Confidence feeds neither disagreement measure, so a bad
            # parse must never cost the window.
            cleaned = re.sub(r"\.+$", "", re.sub(r"\.{2,}", ".", raw_conf))
            try:
                conf = float(cleaned)
            except ValueError:
                conf = None
                result["parse_error"] = True
                logger.warning(
                    "parse_error for %s — could not parse CONFIDENCE %r",
                    agent_name, raw_conf,
                )
        if conf is not None:
            result["confidence"] = round(min(1.0, max(0.0, conf)), 3)

    # ── Reasoning ──────────────────────────────────────────────────────────────
    # Terminator lookahead order: KEY_FACTORS (V1/V2), BULLISH (V3), end-of-string.
    # re.DOTALL so the content can span multiple lines.
    reason_m = re.search(
        r"REASONING\s*:\s*\[?(.*?)(?:\]?\s*(?:KEY_FACTORS\s*:|BULLISH_FACTORS\s*:|\Z))",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if reason_m:
        extracted = reason_m.group(1).strip().strip("[]").strip()
        if extracted:
            result["reasoning"] = extracted

    # ── Key factors ────────────────────────────────────────────────────────────
    # V1/V2 have KEY_FACTORS; V3 has BULLISH_FACTORS + BEARISH_FACTORS.
    factors_m = re.search(
        r"KEY_FACTORS\s*:\s*\[?(.*?)(?:\]?\s*(?:SCORE\s*:|CONFIDENCE\s*:|REASONING\s*:|\Z))",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if factors_m:
        result["key_factors"] = factors_m.group(1).strip().strip("[]").strip()
    else:
        # V3: concatenate bullish + bearish factor lists
        bull_m = re.search(
            r"BULLISH_FACTORS\s*:\s*\[?(.*?)(?:\]?\s*(?:BEARISH_FACTORS\s*:|SCORE\s*:|\Z))",
            text, re.DOTALL | re.IGNORECASE,
        )
        bear_m = re.search(
            r"BEARISH_FACTORS\s*:\s*\[?(.*?)(?:\]?\s*(?:SCORE\s*:|CONFIDENCE\s*:|REASONING\s*:|\Z))",
            text, re.DOTALL | re.IGNORECASE,
        )
        parts = []
        if bull_m:
            parts.append("BULLISH: " + bull_m.group(1).strip().strip("[]").strip())
        if bear_m:
            parts.append("BEARISH: " + bear_m.group(1).strip().strip("[]").strip())
        if parts:
            result["key_factors"] = " | ".join(parts)

    # ── Fallback parse error flags ─────────────────────────────────────────────
    if result["score"] is None:
        result["parse_error"] = True
        logger.warning("parse_error for %s — could not extract valid SCORE", agent_name)

    if result["reasoning"] is None:
        result["parse_error"] = True
        result["reasoning"] = text[:500]
        logger.warning("parse_error for %s — could not extract REASONING", agent_name)

    return result


_CLIENT_CACHE: dict[tuple[str, str], "OpenAI"] = {}


def _client(config: Config) -> OpenAI:
    """
    One OpenAI-compatible client per (base_url, key).

    Groq, OpenRouter, Cerebras and a local Ollama server all expose the same
    /chat/completions contract, so the persona calls are provider-agnostic and
    the provider is a config choice rather than a code change.  Clients are
    cached because a fresh connection pool per call is wasteful across the
    ~2,000 calls a full chunk-level run makes.
    """
    key = config.API_KEY or "not-needed"
    cache_key = (config.API_BASE_URL, key)
    if cache_key not in _CLIENT_CACHE:
        _CLIENT_CACHE[cache_key] = OpenAI(
            base_url=config.API_BASE_URL,
            api_key=key,
            # 60s is generous: a 1,200-word window plus prompt and output returns
            # in 1-6s on a healthy backend. The point of a short ceiling is that a
            # DEAD SOCKET fails fast instead of holding the run hostage.
            timeout=60.0,
            # max_retries=0 is load-bearing. The SDK retries internally by
            # default, underneath our own retry loop in run_agent(), so the two
            # layers MULTIPLY: 120s x SDK retries x 10 of ours let a single
            # hung call consume 43 minutes of wall clock on 2026-10-05, with the
            # SDK's attempts invisible to our logging and our backoff policy.
            # Retries belong in exactly one place, and that place is run_agent(),
            # which honours Retry-After, caps 429 backoff, and never retries 413.
            max_retries=0,
        )
    return _CLIENT_CACHE[cache_key]


def run_agent(
    system_prompt: str,
    filing_text: str,
    agent_name: str,
    config: Config,
    model: Optional[str] = None,
) -> dict:
    """
    Call the configured provider with a system prompt + filing text and return
    parsed output.
    Retries on transient errors with exponential back-off.
    Returns a dict with `failed=True` if all retries are exhausted.
    """
    model_name = model or config.MODEL_NAME
    client = _client(config)

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"---FILING TEXT---\n{filing_text}"},
                ],
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_AGENT_TOKENS,
                **({"extra_body": config.EXTRA_BODY} if config.EXTRA_BODY else {}),
            )
            time.sleep(config.AGENT_SLEEP_SECONDS)
            # An empty completion is a BACKEND FAILURE, not an answer, so it is
            # retried rather than parsed. Treating it as empty text (the earlier
            # behaviour) quietly produced a window missing one persona, which the
            # partial-failure rule exists to prevent — and it was not rare:
            # DekaLLM returned HTTP 200 with no content on 2.3% of calls on
            # 2026-10-05. Retrying is cheap because the request is well formed and
            # a healthy backend answers it; if every attempt comes back empty the
            # call fails honestly and the window is dropped.
            content = response.choices[0].message.content
            if content is None or not str(content).strip():
                raise EmptyCompletionError(
                    f"{agent_name}: provider returned HTTP 200 with an empty "
                    f"completion"
                )
            parsed = parse_agent_output(content, agent_name)
            # Record which backend actually served the call.  Pinning is asserted
            # in the request, but it has to be verifiable after the fact — if a
            # run turns out to have been served by more than one backend, the
            # "model held fixed" claim needs re-checking before the results are used.
            parsed["served_by"] = getattr(response, "provider", None) or config.API_PROVIDER
            return parsed
        except Exception as exc:
            msg = str(exc)
            # A 413 means the request itself is too large: the size does not change
            # on retry, so retrying only burns quota. Fail fast (see the 413 note in
            # CLAUDE.md — the fix is a smaller window, not more retries).
            if "413" in msg or "Request too large" in msg:
                logger.error(
                    "Agent %s: request too large (413) — not retrying. "
                    "Reduce config.CHUNK_WORDS.", agent_name,
                )
                break

            # A 429 is transient congestion on the served backend. With
            # allow_fallbacks disabled we must not reroute to a different
            # backend — that would change the model serving the call — so we
            # wait it out instead, with a longer, capped backoff than the
            # generic path uses. Honour Retry-After when the provider sends one.
            # Match an actual rate-limit signal only: the bare substring "rate"
            # also occurs in "failed to generate completion", which would send a
            # 500 down the long capped 429 backoff instead of the generic one.
            is_rate_limit = bool(
                "429" in msg
                or re.search(r"\brate[ _-]?limit", msg, re.IGNORECASE)
                or re.search(r"\btoo many requests\b", msg, re.IGNORECASE)
            )
            if is_rate_limit:
                wait = min(60.0, 5.0 * (2 ** (attempt - 1)))
                m = re.search(r"(?:retry[- ]after|try again in)\D{0,10}(\d+(?:\.\d+)?)", msg, re.I)
                if m:
                    wait = max(wait, min(120.0, float(m.group(1))))
            else:
                wait = config.RETRY_BACKOFF_BASE ** attempt

            logger.warning(
                "Agent %s attempt %d/%d failed (%s): %.200s — retrying in %.0fs",
                agent_name, attempt, config.MAX_RETRIES,
                "rate limit" if is_rate_limit else "error", msg, wait,
            )
            time.sleep(wait)

    logger.error("Agent %s exhausted retries; returning failed sentinel", agent_name)
    return {
        "agent": agent_name,
        "score": None,
        "confidence": None,
        "reasoning": "API call failed after all retries.",
        "key_factors": None,
        "raw_output": "",
        "parse_error": True,
        "failed": True,
    }


def run_all_agents(
    filing_text: str,
    config: Config,
    agents_override: Optional[dict] = None,
) -> dict[str, dict]:
    """
    Run all three analytical persona prompts on the same filing text.
    Returns a dict keyed by role: 'value', 'sentiment', 'fundamentals'.
    Pass agents_override to use a non-default prompt variant (V2/V3).
    """
    agent_defs = agents_override if agents_override is not None else AGENTS
    outputs: dict[str, dict] = {}
    for role, meta in agent_defs.items():
        logger.info("  Running %s agent ...", meta["name"])
        outputs[role] = run_agent(
            system_prompt=meta["prompt"],
            filing_text=filing_text,
            agent_name=meta["name"],
            config=config,
            model=meta["model"],
        )
    return outputs
