"""
E3 — Encoder robustness.

The primary measure is the mean pairwise cosine distance between three personas'
reasoning embeddings, computed with all-MiniLM-L6-v2. A referee will ask whether
the result is an artifact of that one encoder. This recomputes the whole measure
with two independent encoders and compares.

Costs NOTHING in model calls: it re-encodes the persona reasoning text that the
run persisted to chunk_level_results.csv. (Before that text was persisted, this
experiment was impossible without re-running all ~2,000 scoring calls — the
embedding cache stores hash -> vector with no text.)

It does download encoder weights from Hugging Face on first run.

Deliberately self-contained: it does NOT import analysis/embeddings.py, because
that module hardcodes the primary encoder and is in use by the scoring pipeline.
The encoding recipe here mirrors it exactly — mean-pool over the attention mask,
L2-normalise, max_length=256, 1 - cosine, mean over the three persona pairs — so
the only thing that varies between arms is the encoder.

Run:  python3 -m experiments.encoder_robustness.run
"""
from __future__ import annotations

import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Must precede any transformers import: TensorFlow's abseil library takes a
# global mutex on import that never releases under this Python/PyTorch pair.
sys.modules.setdefault("tensorflow", None)

from analysis.trend_tests import (
    DEFAULT_SEED,
    TIER_ORDER,
    aggregate_to_firm,
    jonckheere_terpstra_labeled,
)
from config import Config

logger = logging.getLogger(__name__)

PRIMARY_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
ALTERNATIVE_ENCODERS = (
    "sentence-transformers/all-mpnet-base-v2",   # 768-d, different architecture
    "BAAI/bge-small-en-v1.5",                    # 384-d, different training corpus
)
MAX_LENGTH = 256          # same truncation as the primary measure
ROLES = ("value", "sentiment", "fundamentals")


def _load(model_name: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModel.from_pretrained(model_name)
    mod.eval()
    return tok, mod, torch


def encode(texts: list[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """Mean-pooled, L2-normalised sentence embeddings. Mirrors the primary recipe."""
    tok, mod, torch = _load(model_name)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=MAX_LENGTH, return_tensors="pt")
            hidden = mod(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.cpu().numpy())
    return np.vstack(out)


def _mean_pairwise_distance(vectors: dict[str, np.ndarray], idx: int) -> float:
    ds = []
    for a, b in combinations(ROLES, 2):
        va, vb = vectors[a][idx], vectors[b][idx]
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom <= 0:
            continue
        ds.append(float(np.clip(1.0 - float(np.dot(va, vb)) / denom, 0.0, 2.0)))
    return float(np.mean(ds)) if ds else float("nan")


def recompute(chunk_df: pd.DataFrame, model_name: str) -> pd.Series:
    """Window-level disagreement under `model_name`, aligned to chunk_df's index."""
    cols = [f"{r}_reasoning" for r in ROLES]
    missing = [c for c in cols if c not in chunk_df.columns]
    if missing:
        raise KeyError(
            f"{missing} not in the window-level output. Per-window reasoning must be "
            f"persisted for this experiment to run without new model calls."
        )

    usable = chunk_df[cols].notna().all(axis=1)
    sub = chunk_df[usable]
    logger.info("Encoding %d windows x 3 personas with %s", len(sub), model_name)

    vectors = {r: encode(sub[f"{r}_reasoning"].astype(str).tolist(), model_name)
               for r in ROLES}
    vals = [_mean_pairwise_distance(vectors, i) for i in range(len(sub))]
    return pd.Series(vals, index=sub.index, name=model_name)


def _tier_summary(chunk_df: pd.DataFrame, measure_col: str,
                  section: str) -> dict:
    firm = aggregate_to_firm(chunk_df, section=section, measure=measure_col)
    means = {t: float(firm.loc[firm["tier"] == t, "firm_mean"].mean())
             for t in TIER_ORDER}
    groups = [firm.loc[firm["tier"] == t, "firm_mean"].to_numpy() for t in TIER_ORDER]
    jt = jonckheere_terpstra_labeled(
        firm["firm_mean"].to_numpy(), firm["tier"].to_numpy(),
        group_order=TIER_ORDER, alternative="increasing",
    )
    ordered = means[TIER_ORDER[0]] < means[TIER_ORDER[1]] < means[TIER_ORDER[2]]
    return {
        "tier_means": means,
        "monotone_small_gt_mid_gt_large": bool(ordered),
        "jt_statistic": float(jt.statistic),
        "jt_z": float(jt.z),
        "jt_p_one_sided": float(jt.p_value),
        "n_firms": int(len(firm)),
        "firm_means": firm[["ticker", "tier", "firm_mean"]].to_dict("records"),
        "_group_sizes": [len(g) for g in groups],
    }


def run(config: Config, chunk_path: Path | None = None, section: str = "mda",
        seed: int = DEFAULT_SEED) -> dict:
    chunk_path = chunk_path or (config.OUTPUT_DIR / "chunk_level_results.csv")
    if not chunk_path.exists():
        raise FileNotFoundError(f"{chunk_path} not found — run the scoring pipeline first.")
    chunk_df = pd.read_csv(chunk_path)

    out_dir = config.OUTPUT_DIR / "experiments" / "encoder_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "section": section,
        "seed": seed,
        "primary_encoder": PRIMARY_ENCODER,
        "max_length": MAX_LENGTH,
        "n_windows": int(len(chunk_df)),
        "note": "Re-encodes persisted persona reasoning. No model/API calls.",
        "arms": {},
    }

    # Primary arm, as the pipeline computed it.
    results["arms"][PRIMARY_ENCODER] = _tier_summary(
        chunk_df, "embedding_disagreement", section
    )

    enriched = chunk_df.copy()
    for name in ALTERNATIVE_ENCODERS:
        col = f"disag__{name.split('/')[-1]}"
        try:
            enriched[col] = recompute(chunk_df, name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Encoder %s failed: %s", name, exc)
            results["arms"][name] = {"error": str(exc)}
            continue
        results["arms"][name] = _tier_summary(enriched, col, section)

        ok = enriched[col].notna() & enriched["embedding_disagreement"].notna()
        if ok.sum() >= 3:
            rho, p = spearmanr(enriched.loc[ok, col],
                               enriched.loc[ok, "embedding_disagreement"])
            results["arms"][name]["window_rank_corr_with_primary"] = {
                "spearman_rho": float(rho), "p": float(p), "n_windows": int(ok.sum()),
            }

    # Firm-level agreement between arms, which is what the paper's claim rests on.
    firm_tables = {}
    for name, arm in results["arms"].items():
        if "firm_means" in arm:
            f = pd.DataFrame(arm["firm_means"]).set_index("ticker")["firm_mean"]
            firm_tables[name] = f
    if len(firm_tables) >= 2:
        fm = pd.DataFrame(firm_tables)
        results["firm_level_rank_correlations"] = {}
        for a, b in combinations(fm.columns, 2):
            rho, p = spearmanr(fm[a], fm[b])
            results["firm_level_rank_correlations"][f"{a} vs {b}"] = {
                "spearman_rho": float(rho), "p": float(p), "n_firms": int(len(fm)),
            }
        fm.to_csv(out_dir / "firm_means_by_encoder.csv")

    enriched.to_csv(out_dir / "window_level_by_encoder.csv", index=False)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=1, default=str), encoding="utf-8"
    )
    return results


def write_report(config: Config, **kw) -> str:
    r = run(config, **kw)
    L: list[str] = []
    L.append("E3 — Encoder robustness")
    L.append("=" * 78)
    L.append("")
    L.append(f"Section: {r['section']}   Windows: {r['n_windows']:,}   "
             f"Truncation: {r['max_length']} tokens (same as the primary measure)")
    L.append("No model or API calls: persona reasoning was re-encoded from cached text.")
    L.append("")
    L.append("Tier means and the ordered-trend test under each encoder")
    L.append("-" * 78)
    L.append(f"{'encoder':34s} {'large':>8s} {'mid':>8s} {'small':>8s} "
             f"{'monotone':>9s} {'JT p':>8s}")
    for name, arm in r["arms"].items():
        short = name.split("/")[-1]
        if "error" in arm:
            L.append(f"{short:34s}  FAILED: {arm['error'][:40]}")
            continue
        m = arm["tier_means"]
        L.append(f"{short:34s} {m['large']:8.4f} {m['mid']:8.4f} {m['small']:8.4f} "
                 f"{str(arm['monotone_small_gt_mid_gt_large']):>9s} "
                 f"{arm['jt_p_one_sided']:8.4f}")
    L.append("")
    for name, arm in r["arms"].items():
        wc = arm.get("window_rank_corr_with_primary") if isinstance(arm, dict) else None
        if wc:
            L.append(f"  window-level Spearman vs primary, {name.split('/')[-1]}: "
                     f"rho = {wc['spearman_rho']:+.3f} (p = {wc['p']:.2g}, "
                     f"n = {wc['n_windows']:,})")
    L.append("")
    if "firm_level_rank_correlations" in r:
        L.append("Firm-level rank correlations between encoders")
        L.append("-" * 78)
        for pair, v in r["firm_level_rank_correlations"].items():
            a, b = (p.split("/")[-1] for p in pair.split(" vs "))
            L.append(f"  {a} vs {b}: rho = {v['spearman_rho']:+.3f} "
                     f"(p = {v['p']:.2g}, n = {v['n_firms']} firms)")
        L.append("")
    L.append("How to read this")
    L.append("-" * 78)
    L.append("  What must hold for the paper's claim to be encoder-independent is that the")
    L.append("  TIER ORDERING and the trend test agree across encoders. High window-level")
    L.append("  rank correlation is reassuring but not sufficient, and a moderate one is")
    L.append("  not fatal: different encoders can disagree on individual windows while")
    L.append("  agreeing on firm-level aggregates, which is the level the hypothesis is")
    L.append("  about. If the ordering flips under another encoder, say so plainly — that")
    L.append("  would mean the result is an encoder artifact.")
    L.append("")
    L.append("  Labelled EXPLORATORY in ANALYSIS_PLAN.md section 6.")
    L.append("")
    text = "\n".join(L)
    (config.OUTPUT_DIR / "experiments" / "encoder_robustness" / "report.txt").write_text(
        text, encoding="utf-8"
    )
    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(write_report(Config()))
