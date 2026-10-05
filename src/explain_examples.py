"""
Day 4c: explanation figures + tables for the rule-picked demo examples (D29). Inference only.

For each example in outputs/tables/demo_examples.csv: E1 risk score and band, exact per-feature logit
contributions (coef x transformed value = SHAP LinearExplainer values, D27), group totals, and one figure
(group totals + the 8 largest individual contributions). The payload is drawn as escaped plain text.

Run:  python -m src.explain_examples
Writes: outputs/shap/<cell>.png, outputs/shap/<cell>_contributions.csv, outputs/shap/examples_summary.csv
"""
from __future__ import annotations

import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src import explainability as ex

TOP_K = 8
RED, BLUE = "#b5392b", "#2f6db5"


def slug(cell: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", cell.lower()).strip("_")


def make_figure(payload: str, e: ex.Explanation, truth: str, path) -> None:
    contrib = pd.Series(e.feature_contributions)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).head(TOP_K).index).sort_values()
    groups = pd.Series(e.group_contributions).sort_values()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    a1.barh(groups.index, groups.values, color=[RED if v > 0 else BLUE for v in groups.values])
    a1.set_title("Effect by feature group", fontsize=10)
    a2.barh([ex.LABELS[f][0] for f in top.index], top.values, color=[RED if v > 0 else BLUE for v in top.values])
    a2.set_title(f"Largest single features (top {TOP_K})", fontsize=10)
    for a in (a1, a2):
        a.axvline(0, color="#444", lw=0.8)
        a.set_xlabel("lowers score  <-  logit effect  ->  raises score", fontsize=8)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    shown = ex.escape_for_display(payload, 80)
    fig.suptitle(f"E1 risk score {e.score:.2f} ({e.band}); true label: {truth}\n{shown}", fontsize=10)
    fig.text(0.5, 0.01, "Single features can offset each other; group totals are the more stable summary. Model explanation, not proof of malice.", ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ex_df = pd.read_csv(config.TABLES_DIR / "demo_examples.csv").dropna(subset=["payload"])
    b = ex.load_e1()
    config.SHAP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in ex_df.itertuples(index=False):
        assert ex.classify_payload(r.payload) == "url", r.cell
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
            e = ex.explain_url(r.payload, b)
        s = slug(r.cell)
        truth = "benign" if r.cell.startswith("benign") else "malicious"
        make_figure(r.payload, e, truth, config.SHAP_DIR / f"{s}.png")
        pd.DataFrame({"feature": list(e.feature_contributions), "group": [ex.GROUP_OF[f] for f in e.feature_contributions],
                      "logit_contribution": list(e.feature_contributions.values())}).sort_values(
            "logit_contribution", key=abs, ascending=False).to_csv(config.SHAP_DIR / f"{s}_contributions.csv", index=False)
        rows.append({"cell": r.cell, "score": round(e.score, 3), "band": e.band, "intercept": round(e.intercept, 3),
                     **{f"group: {k}": round(v, 2) for k, v in e.group_contributions.items()}})
    summ = pd.DataFrame(rows)
    summ.to_csv(config.SHAP_DIR / "examples_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(summ.to_string(index=False))
    print(f"\nSaved figures and tables to {config.SHAP_DIR}")


if __name__ == "__main__":
    main()
