"""
E4b — Is the boilerplate result one unusual passage, or a property of boilerplate?

E4 compared real MD&A windows against ONE synthetic safe-harbour passage, scored
once per window. That estimates disagreement for THAT text well, but it cannot
distinguish "boilerplate produces high disagreement" from "this particular
disclaimer happens to be ambiguous". The between-text variance is unmeasured,
which is a real hole in the paper's sharpest negative finding.

This closes it two ways:
  * REAL boilerplate, harvested from the sample's own filings -- forward-looking
    statement disclaimers, which are genuine filing prose carrying no
    firm-specific content -- rather than a passage written for the experiment.
  * SEVERAL distinct passages, so the spread ACROSS texts is estimated instead
    of assumed.

Run:  python3 -m experiments.placebo.boilerplate_variants --n-texts 10
      python3 -m experiments.placebo.boilerplate_variants --n-texts 10 --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.modules.setdefault("tensorflow", None)

from analysis.trend_tests import DEFAULT_SEED, bootstrap_ci
from config import Config

logger = logging.getLogger(__name__)
ROLES = ("value", "sentiment", "fundamentals")

# Where filings park their safe-harbour language.
FLS_ANCHOR = re.compile(
    r"(forward[- ]looking statements?|private securities litigation reform|"
    r"safe harbor)", re.I)
TARGET_WORDS = 1200


def harvest(config: Config, n_texts: int, seed: int) -> list[dict]:
    """Pull distinct forward-looking-statement passages from the cached filings."""
    from data.edgar import _html_to_text

    rng = np.random.default_rng(seed)
    files = sorted(config.FILINGS_DIR.glob("*.txt"))
    rng.shuffle(files)

    out, seen = [], set()
    for path in files:
        if len(out) >= n_texts:
            break
        try:
            text = _html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        m = FLS_ANCHOR.search(text)
        if not m:
            continue
        words = text[m.start():].split()
        if len(words) < TARGET_WORDS:
            continue
        passage = " ".join(words[:TARGET_WORDS])
        # Reject anything that is clearly not boilerplate: real MD&A is dense
        # with figures, safe-harbour prose is not.
        digits = sum(c.isdigit() for c in passage) / max(1, len(passage))
        if digits > 0.02:
            continue
        fp = passage[:120]
        if fp in seen:
            continue
        seen.add(fp)
        out.append({"source": path.stem, "text": passage,
                    "digit_density": round(digits, 5)})
    return out


def run(config: Config, n_texts: int, seed: int, apply: bool) -> dict:
    texts = harvest(config, n_texts, seed)
    print(f"harvested {len(texts)} distinct boilerplate passages "
          f"({TARGET_WORDS} words each) from real filings")
    for t in texts[:4]:
        print(f"  {t['source']:28s} digits {t['digit_density']:.4f}  "
              f"\"{' '.join(t['text'].split()[:11])}...\"")
    print(f"calls: {len(texts)*3}")
    if not apply:
        print("\nDRY RUN — nothing called. Re-run with --apply.")
        return {"dry_run": True, "n_texts": len(texts)}

    from agents.prompts import AGENTS
    from agents.runner import run_agent
    from analysis.disagreement import compute_disagreement

    out_dir = config.OUTPUT_DIR / "experiments" / "placebo"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "boilerplate_variants.csv"
    rows, done = [], set()
    if done_path.exists():
        prev = pd.read_csv(done_path)
        rows = prev.to_dict("records")
        done = set(prev.source)
        print(f"resuming: {len(done)} passage(s) already scored")

    for i, t in enumerate(texts, 1):
        if t["source"] in done:
            continue
        print(f"[{i}/{len(texts)}] {t['source']}", flush=True)
        outs = {r: run_agent(AGENTS[r]["prompt"], t["text"], AGENTS[r]["name"],
                             config) for r in ROLES}
        if any(outs[r].get("failed") for r in ROLES):
            continue
        d = compute_disagreement(outs, config,
                                 cache_path=config.OUTPUT_DIR / "embedding_cache.json")
        rows.append({"source": t["source"], "digit_density": t["digit_density"],
                     "embedding_disagreement": d.get("embedding_disagreement"),
                     "numeric_disagreement": d.get("numeric_disagreement")})
        pd.DataFrame(rows).to_csv(done_path, index=False)

    return summarise(pd.DataFrame(rows), config, out_dir, seed)


def summarise(df: pd.DataFrame, config: Config, out_dir: Path, seed: int) -> dict:
    ch = pd.read_csv(config.OUTPUT_DIR / "chunk_level_results.csv")
    real = ch[(ch.section == "mda")
              & (~ch.any_failed.astype(str).str.lower().isin(["true", "1"]))
              & ch.embedding_disagreement.notna()].embedding_disagreement

    b = df.embedding_disagreement.dropna().to_numpy()
    cb = bootstrap_ci(b, statistic=np.mean, n_boot=10000, seed=seed,
                      method="percentile")
    cr = bootstrap_ci(real.to_numpy(), statistic=np.mean, n_boot=10000, seed=seed,
                      method="percentile")

    # One synthetic passage, from E4, for comparison.
    e4 = None
    p = out_dir / "results.json"
    if p.exists():
        e4 = ((json.loads(p.read_text()).get("conditions") or {})
              .get("boilerplate") or {}).get("mean")

    res = {
        "seed": seed, "n_boilerplate_texts": int(len(b)),
        "boilerplate_real_passages": {
            "mean": float(cb.observed), "ci": [float(cb.low), float(cb.high)],
            "sd_across_texts": float(np.std(b, ddof=1)) if len(b) > 1 else None,
            "min": float(b.min()), "max": float(b.max()),
        },
        "real_mda_windows": {
            "mean": float(cr.observed), "ci": [float(cr.low), float(cr.high)],
            "n": int(len(real)),
        },
        "e4_single_synthetic_passage": e4,
        "boilerplate_exceeds_real": bool(cb.observed > cr.observed),
        "difference": float(cb.observed - cr.observed),
    }
    (out_dir / "boilerplate_variants.json").write_text(
        json.dumps(res, indent=1, default=str), encoding="utf-8")
    txt = report(res)
    (out_dir / "boilerplate_variants.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    return res


def report(r: dict) -> str:
    bp, rl = r["boilerplate_real_passages"], r["real_mda_windows"]
    L = ["E4b — Boilerplate, measured across DISTINCT real passages", "=" * 74, ""]
    L.append(f"  real MD&A windows        {rl['mean']:.4f}  "
             f"[{rl['ci'][0]:.4f}, {rl['ci'][1]:.4f}]   n = {rl['n']}")
    L.append(f"  real boilerplate ({r['n_boilerplate_texts']} texts) "
             f"{bp['mean']:.4f}  [{bp['ci'][0]:.4f}, {bp['ci'][1]:.4f}]")
    if bp.get("sd_across_texts") is not None:
        L.append(f"    spread across texts: sd {bp['sd_across_texts']:.4f}, "
                 f"range {bp['min']:.4f} to {bp['max']:.4f}")
    if r.get("e4_single_synthetic_passage"):
        L.append(f"  (E4's single synthetic passage: "
                 f"{r['e4_single_synthetic_passage']:.4f})")
    L.append("")
    L.append(f"  boilerplate exceeds real MD&A: {r['boilerplate_exceeds_real']}  "
             f"(difference {r['difference']:+.4f})")
    L.append("")
    L.append("  Passages are forward-looking-statement disclaimers taken from the")
    L.append("  sample's own filings, screened to be near-free of figures, so this is")
    L.append("  genuine filing prose rather than text written for the experiment. With")
    L.append("  several distinct passages the spread ACROSS texts is estimated, which")
    L.append("  the single-passage version of this test could not do.")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-texts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--agent-sleep", type=float, default=0.5)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    config = Config()
    config.API_PROVIDER = a.provider
    config.apply_provider()
    config.AGENT_SLEEP_SECONDS = a.agent_sleep
    if a.backend:
        config.PROVIDERS[config.API_PROVIDER]["extra_body"] = {
            "provider": {"order": [a.backend], "allow_fallbacks": False,
                         "quantizations": ["bf16"]}}
    run(config, a.n_texts, a.seed, a.apply)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
