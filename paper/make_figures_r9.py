"""
Round 9 / AI4F v2 figures, generated from output/ with no model calls.

  fig_r9_scale      share of narrative read under fixed budgets, 785 10-Ks (S2)
  fig_r9_placebo    real vs shuffled vs boilerplate, two model families (E4, E4b, E7)
  fig_r9_gradients  the size gradient under every scoring/aggregation variant, against
                    the W6 no-signal band (replaces fig4, whose script plotted the keys
                    of a summary dict rather than a distribution)

Run:  python3 -m paper.make_figures_r9
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper.make_figures import _style, _save, INK, MUTED, GRID, TIER_COLOR

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FIG = ROOT / "paper" / "figures"
DARK, MIDC, LIGHT = TIER_COLOR["large"], TIER_COLOR["mid"], TIER_COLOR["small"]
ACCENT = "#b8562e"  # single warm accent for the one condition the reader should find


def _j(rel):
    return json.loads((OUT / rel).read_text(encoding="utf-8"))


def fig_scale():
    sa = _j("experiments/scale_audit/results.json")
    keys = ["512 tokens", "2,000 words", "4k tokens", "8k tokens", "32k tokens"]
    labels = ["512 tok", "2,000 w", "4k tok", "8k tok", "32k tok"]
    b = [sa["budgets"][k] for k in keys]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.15), gridspec_kw={"wspace": 0.32})
    x = range(len(keys))
    a1.plot(x, [100 * v["median_bottom_decile"] for v in b], ls=":", marker="^", color=LIGHT,
            mec=DARK, mew=0.5, label="Smallest decile (float)")
    a1.plot(x, [100 * v["median_overall"] for v in b], ls="--", marker="s", color=MIDC,
            label="All filings")
    a1.plot(x, [100 * v["median_top_decile"] for v in b], ls="-", marker="o", color=DARK,
            label="Largest decile (float)")
    a1.set_xticks(list(x), labels)
    a1.set_ylabel("Median share of narrative read (%)")
    a1.set_ylim(0, 105)
    a1.legend(loc="upper left")
    a1.set_title(f"Coverage by budget, n = {sa['n_usable']} 10-Ks")
    a2.bar(list(x), [100 * v["share_truncated"] for v in b], color=MIDC, edgecolor=DARK, lw=0.5,
           width=0.6)
    for i, v in enumerate(b):
        a2.text(i, 100 * v["share_truncated"] + 2, f"{100 * v['share_truncated']:.0f}%",
                ha="center", va="bottom", fontsize=7, color=INK)
    a2.set_xticks(list(x), labels)
    a2.set_ylim(0, 112)
    a2.set_ylabel("Filings truncated (%)")
    a2.set_title("Share of filings the budget cuts")
    a2.grid(axis="x", visible=False)
    _save(fig, FIG, "fig_r9_scale")


def fig_placebo():
    pl = _j("experiments/placebo/results.json")
    xm = _j("experiments/cross_model/results.json")
    bv = _j("experiments/placebo/boilerplate_variants.json")
    conds = ["real", "shuffled", "boilerplate"]
    names = ["Real MD&A", "Sentence-\nshuffled", "Boilerplate\n(synthetic)"]
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    off = {"gpt": -0.14, "llama": 0.14}
    for key, src, col, mk, lab in [
            ("gpt", pl["conditions"], DARK, "o", f"gpt-oss-20b (n = {pl['n_windows']})"),
            ("llama", xm["conditions"], MIDC, "s", f"Llama-3.3-70B (n = {xm['n_windows']})")]:
        xs = [i + off[key] for i in range(3)]
        ms = [src[c]["mean"] for c in conds]
        lo = [src[c]["mean"] - src[c]["ci"][0] for c in conds]
        hi = [src[c]["ci"][1] - src[c]["mean"] for c in conds]
        ax.errorbar(xs, ms, yerr=[lo, hi], fmt=mk, color=col, mec=col, ms=4.5, capsize=2,
                    lw=1.0, label=lab)
        ax.plot(xs, ms, color=col, lw=0.6, alpha=0.6)
    bp = bv["boilerplate_real_passages"]
    ax.errorbar([3], [bp["mean"]], yerr=[[bp["mean"] - bp["ci"][0]], [bp["ci"][1] - bp["mean"]]],
                fmt="D", color=ACCENT, mec=ACCENT, ms=4.5, capsize=2, lw=1.0,
                label=f"gpt-oss-20b, {bv['n_boilerplate_texts']} real disclaimers")
    ax.set_xticks([0, 1, 2, 3], names + ["Boilerplate\n(from filings)"])
    ax.set_ylabel("Persona disagreement\n(mean pairwise cosine distance)")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0.22, 0.50)
    ax.grid(axis="x", visible=False)
    _save(fig, FIG, "fig_r9_placebo")


def fig_gradients():
    wb = _j("experiments/window_battery/report.json")
    ag = _j("experiments/decomposition/aggregation_control.json")
    cc = _j("experiments/decomposition/coverage_contrast.json")
    tr = _j("experiments/truncation_replication/results.json") if (
        OUT / "experiments/truncation_replication/results.json").exists() else None
    rows = {}
    with (OUT / "truncation_contrast/truncation_contrast_gradients.csv").open() as f:
        for r in csv.DictReader(f):
            if r["measure"] == "embedding_disagreement" and r["variant"] == "inclusive":
                rows[r["k"]] = r
    w3 = wb["W3"]["by_k"]["k=2"]["gradient_distribution"]["percentiles"]
    w6 = wb["W6"]["null_gradient"]["percentiles"]
    s1b = None
    if tr is not None:
        for path in (("truncated", "gradient"), ("arms", "truncated", "gradient"),
                     ("truncated_gradient",)):
            d = tr
            for k in path:
                d = d.get(k) if isinstance(d, dict) else None
            if isinstance(d, (int, float)):
                s1b = d
                break
    if s1b is None:
        s1b = _j("experiments/decomposition/results.json")["cells"]["truncated_single_unit_S1b"]["gradient"]

    items = [  # label, point, lo, hi, group
        ("Complete, windows (pre-reg.)", float(rows["all"]["gradient_small_minus_large"]),
         float(rows["all"]["ci_lo"]), float(rows["all"]["ci_hi"]), "main"),
        ("Complete, 1 random window", ag["complete_one_random_window"]["mean"],
         *ag["complete_one_random_window"]["ci"], "agg"),
        ("Equal-$k$, $k=2$ (W3)", w3["p50"], w3["p2.5"], w3["p97.5"], "agg"),
        ("First 2 windows (S1a)", float(rows["2"]["gradient_small_minus_large"]),
         float(rows["2"]["ci_lo"]), float(rows["2"]["ci_hi"]), "trunc"),
        ("First window (S1a)", float(rows["1"]["gradient_small_minus_large"]),
         float(rows["1"]["ci_lo"]), float(rows["1"]["ci_hi"]), "trunc"),
        ("2,000 w, 1 read, A (S1b)", s1b, None, None, "unit"),
        ("2,000 w, 1 read, B", cc["truncated_single_unit"]["gradient"], None, None, "unit"),
        ("Complete, 1 read, B", cc["complete_single_unit"]["gradient"], None, None, "unit"),
    ]
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    ax.axvspan(w6["p2.5"], w6["p97.5"], color=GRID, alpha=0.7, lw=0,
               label="No-signal null, central 95% (W6)")
    ax.axvline(0, color=MUTED, lw=0.6)
    col = {"main": DARK, "agg": MIDC, "trunc": MIDC, "unit": ACCENT}
    n = len(items)
    for i, (lab, pt, lo, hi, g) in enumerate(items):
        y = n - 1 - i
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=col[g], lw=1.0)
        ax.plot([pt], [y], marker="o" if lo is not None else "D", color=col[g], ms=4.2,
                mfc=col[g] if g != "unit" else "white", mew=1.0)
    ax.set_yticks(range(n), [it[0] for it in items][::-1], fontsize=7.5)
    ax.set_xlabel("Size gradient in persona disagreement (small $-$ large)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=7)
    _save(fig, FIG, "fig_r9_gradients")


def main():
    _style()
    fig_scale()
    fig_placebo()
    fig_gradients()
    print("wrote fig_r9_scale, fig_r9_placebo, fig_r9_gradients")


if __name__ == "__main__":
    main()
