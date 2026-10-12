"""
Generate the paper's figures as vector PDFs.

Every figure is built from output/ by code — none is drawn by hand, and none
embeds a number that is not read from a generated artifact.

DESIGN DECISIONS, recorded because a referee prints in greyscale:

  Size tier is an ORDERED category, so it is encoded with a single-hue
  sequential ramp (lightness carries size) rather than three categorical hues.
  That is what makes the figures survive a greyscale print: the three steps have
  strictly monotone relative luminance (0.516 / 0.182 / 0.039) with large gaps.

  The ramp was validated rather than eyeballed. Adjacent-pair separation under
  simulated colour-vision deficiency is dE 22.6 (deuteranopia) and 22.8
  (tritanopia), well above the 8 threshold, and 22.9 under normal vision.

  The lightest step sits below 3:1 contrast on a white page. That is permitted
  only with relief, so every series carries relief and then some: a distinct
  marker (o / s / ^), a distinct linestyle, a direct label, a thin dark edge on
  light fills, and the same values repeated in a table in the paper body.

  No dual-axis charts. Grid and axes are recessive. Marks are thin.

Run:  python3 -m paper.make_figures
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.modules.setdefault("tensorflow", None)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from config import Config

logger = logging.getLogger(__name__)

# Single-hue sequential ramp; lightness encodes firm size. Validated for CVD.
TIER_COLOR = {"large": "#12395f", "mid": "#3b7ab5", "small": "#9cc3e3"}
TIER_MARKER = {"large": "o", "mid": "s", "small": "^"}
TIER_LINESTYLE = {"large": "-", "mid": "--", "small": ":"}
TIER_LABEL = {"large": "Large-cap", "mid": "Mid-cap", "small": "Small-cap"}
TIER_ORDER = ("large", "mid", "small")

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"
EDGE = "#2b2b2b"

SKIPPED: list[str] = []


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "pdf.fonttype": 42,        # embed TrueType, not Type-3 (ACM requirement)
        "ps.fonttype": 42,
        "font.size": 8,
        "font.family": "serif",    # matches acmart body text
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
    })


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    logger.info("wrote %s.pdf", name)


def _load_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


# ── Figure 1: coverage under fixed budgets, by tier ──────────────────────────

def fig_coverage(config: Config, out_dir: Path) -> bool:
    """
    The confound, stated as the reader's own practice: under a fixed word budget,
    what fraction of a filing's narrative does the model actually read, by size?
    """
    path = config.OUTPUT_DIR / "coverage_diagnostic.csv"
    if not path.exists():
        SKIPPED.append("fig1_coverage (needs coverage_diagnostic.csv)")
        return False
    cov = pd.read_csv(path)

    # Budgets common in the literature, plus this project's retired cap.
    budgets = [512 * 0.75, 2000, 4096 * 0.75, 8192 * 0.75, 32768 * 0.75]
    labels = ["512 tok", "2,000 w", "4k tok", "8k tok", "32k tok"]

    cap_idx = budgets.index(2000)

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    # Mark the retired cap this project actually used.
    ax.axvline(cap_idx, color=MUTED, linewidth=0.6, linestyle="--", zorder=0)
    ax.annotate("retired cap", (cap_idx, 101), ha="center", va="bottom",
                fontsize=6.5, color=MUTED)
    for tier in TIER_ORDER:
        sub = cov[cov["tier"] == tier]
        if sub.empty:
            continue
        fracs = [float(np.median(np.minimum(1.0, b / sub["section_words_total"]))) * 100
                 for b in budgets]
        ax.plot(range(len(budgets)), fracs,
                color=TIER_COLOR[tier], marker=TIER_MARKER[tier],
                linestyle=TIER_LINESTYLE[tier], label=TIER_LABEL[tier],
                markeredgecolor=EDGE, markeredgewidth=0.4, clip_on=False)
        # Relief for the light step: a value label at the retired 2,000-word
        # cap, which is the budget the paper's own claim is about. Labels go
        # HERE and not at the right-hand end, where every tier converges on
        # 100% and three labels would collide.
        ax.annotate(f"{fracs[cap_idx]:.0f}%", (cap_idx, fracs[cap_idx]),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7, color=INK,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.85))

    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Fixed reading budget")
    ax.set_ylabel("Median share of narrative read")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.2, len(budgets) - 1 + 0.2)
    ax.legend(loc="lower right", ncol=1, borderpad=0.2)
    _save(fig, out_dir, "fig1_coverage")
    return True


# ── Figure 2: the truncation curve ───────────────────────────────────────────

def fig_truncation(config: Config, out_dir: Path) -> bool:
    """Tier gradient as a function of how many windows the model was allowed."""
    exp = config.OUTPUT_DIR / "experiments" / "truncation_contrast"
    csv = exp / "truncation_contrast_gradients.csv"
    if not csv.exists():
        SKIPPED.append("fig2_truncation (needs experiments/truncation_contrast/)")
        return False
    d = pd.read_csv(csv)
    # The primary measure, inclusive budget rule (every filing contributes
    # min(k, its windows)); the strict variant drops short filings, and those
    # drops are not tier-neutral.
    d = d[(d.measure == "embedding_disagreement") & (d.variant == "inclusive")].copy()
    d = d.sort_values("k_numeric")
    kcol, gcol = "k", "gradient_small_minus_large"
    if d.empty:
        SKIPPED.append("fig2_truncation (no embedding/inclusive rows)")
        return False

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(len(d))
    lo = d["ci_low"] if "ci_low" in d else None
    hi = d["ci_high"] if "ci_high" in d else None
    if lo is not None and hi is not None:
        ax.fill_between(x, lo, hi, color=TIER_COLOR["small"], alpha=0.45,
                        linewidth=0.4, edgecolor=EDGE, label="95% CI")
    ax.plot(x, d[gcol], color=TIER_COLOR["large"], marker="o",
            markeredgecolor=EDGE, markeredgewidth=0.4,
            label="Gradient (small $-$ large)", clip_on=False)
    ax.axhline(0, color=MUTED, linewidth=0.6, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in d[kcol]])
    ax.set_xlabel("Windows of MD\\&A read per filing")
    ax.set_ylabel("Tier gradient in disagreement")
    ax.legend(loc="best")
    _save(fig, out_dir, "fig2_truncation")
    return True


# ── Figure 3: validity panel ─────────────────────────────────────────────────

def fig_validity(config: Config, out_dir: Path) -> bool:
    """Noise floor, persona-free baseline and reliability, side by side."""
    exp = config.OUTPUT_DIR / "experiments"
    rel = _load_json(exp / "reliability" / "results.json")
    if rel is None:
        SKIPPED.append("fig3_validity (needs experiments/reliability/results.json)")
        return False
    emb = (rel or {}).get("embedding_disagreement", {})
    sh = (emb.get("split_half") or {})
    tr = (emb.get("test_retest") or {})

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))

    ax = axes[0]
    per = exp / "reliability" / "split_half_embedding_disagreement.csv"
    if per.exists():
        p = pd.read_csv(per)
        for tier in TIER_ORDER:
            s = p[p["tier"] == tier]
            if s.empty:
                continue
            ax.scatter(s["half_even"], s["half_odd"], s=16,
                       facecolor=TIER_COLOR[tier], edgecolor=EDGE, linewidth=0.4,
                       marker=TIER_MARKER[tier], label=TIER_LABEL[tier])
        lim = [min(p[["half_even", "half_odd"]].min()) * 0.97,
               max(p[["half_even", "half_odd"]].max()) * 1.03]
        ax.plot(lim, lim, color=MUTED, linewidth=0.6, linestyle="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        r = sh.get("pearson_half_half")
        sb = sh.get("spearman_brown_from_pearson")
        sub = f"$r$ = {r:+.3f}" if r is not None else ""
        if sb is not None:
            sub += f", Spearman--Brown {sb:+.3f}"
        ax.set_title(f"Split-half within filing\n{sub}", fontsize=8)
    ax.set_xlabel("Even-indexed windows")
    ax.set_ylabel("Odd-indexed windows")
    ax.legend(loc="upper left", fontsize=6.5)

    ax = axes[1]
    per = exp / "reliability" / "test_retest_embedding_disagreement.csv"
    if per.exists():
        p = pd.read_csv(per)
        for tier in TIER_ORDER:
            s = p[p["tier"] == tier]
            if s.empty:
                continue
            ax.scatter(s["10-K"], s["10-Q"], s=16,
                       facecolor=TIER_COLOR[tier], edgecolor=EDGE, linewidth=0.4,
                       marker=TIER_MARKER[tier], label=TIER_LABEL[tier])
        lim = [min(p[["10-K", "10-Q"]].min()) * 0.97,
               max(p[["10-K", "10-Q"]].max()) * 1.03]
        ax.plot(lim, lim, color=MUTED, linewidth=0.6, linestyle="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        r = tr.get("pearson_10K_10Q")
        ax.set_title("Same firm, 10-K vs 10-Q"
                     + (f"\n$r$ = {r:+.3f}" if r is not None else ""), fontsize=8)
    ax.set_xlabel("10-K disagreement")
    ax.set_ylabel("10-Q disagreement")

    fig.tight_layout()
    _save(fig, out_dir, "fig3_validity")
    return True


# ── Figure 4: W3 against the W6 null ─────────────────────────────────────────

def fig_window_robustness(config: Config, out_dir: Path) -> bool:
    """
    The advisor's concern, made quantitative: the equal-k resampled gradient
    against the gradient that pure averaging can manufacture with no tier effect.
    """
    bat = _load_json(config.OUTPUT_DIR / "experiments" / "window_battery" / "report.json")
    if bat is None:
        SKIPPED.append("fig4_window_robustness (needs experiments/window_battery/results.json)")
        return False
    w3 = ((bat.get("W3") or {}).get("by_k") or {}).get("k=2", {})
    w6 = bat.get("W6") or {}
    draws = w3.get("gradient_distribution")
    null = w6.get("null_gradient")
    if not draws or not null:
        SKIPPED.append("fig4_window_robustness (distributions not in results.json)")
        return False

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    bins = 40
    ax.hist(null, bins=bins, color="#e3e3e3", edgecolor=EDGE, linewidth=0.3,
            label="Null: window counts held,\nno tier effect (W6)")
    ax.hist(draws, bins=bins, color=TIER_COLOR["mid"], alpha=0.85,
            edgecolor=EDGE, linewidth=0.3,
            label="Equal-$k$ resampled, $k$=2 (W3)")
    obs = (bat.get("primary") or {}).get("gradient_small_minus_large")
    if obs is not None:
        ax.axvline(obs, color=TIER_COLOR["large"], linewidth=1.4)
        ax.annotate("observed", (obs, ax.get_ylim()[1] * 0.92),
                    textcoords="offset points", xytext=(4, 0),
                    fontsize=7, color=INK)
    ax.axvline(0, color=MUTED, linewidth=0.6)
    ax.set_xlabel("Tier gradient (small $-$ large)")
    ax.set_ylabel("Resamples")
    ax.legend(loc="upper left", fontsize=6.5)
    _save(fig, out_dir, "fig4_window_robustness")
    return True


def main() -> int:
    _style()
    config = Config()
    out_dir = Path(config.BASE_DIR) / "paper" / "figures"
    made = []
    for fn in (fig_coverage, fig_truncation, fig_validity, fig_window_robustness):
        try:
            if fn(config, out_dir):
                made.append(fn.__name__)
        except Exception as exc:  # noqa: BLE001
            SKIPPED.append(f"{fn.__name__} (error: {exc})")
            logger.exception("figure %s failed", fn.__name__)

    print(f"wrote {len(made)} figure(s) to {out_dir}")
    for m in made:
        print(f"  + {m}")
    if SKIPPED:
        print(f"\n{len(SKIPPED)} figure(s) not yet buildable:")
        for s in SKIPPED:
            print(f"  - {s}")
        print("\nThese build once the corresponding analysis has run.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
