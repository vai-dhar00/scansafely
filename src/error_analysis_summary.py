"""
Day 5: qualitative review of selected errors (D32). Descriptive only; changes no model, threshold or result.

SELECTION RULE (fixed, stated on the poster): at each model's LOCKED primary threshold, the K=10 false positives
with the HIGHEST scores and the K=10 false negatives with the LOWEST scores. These are the most confident errors,
NOT a random sample, so category counts must never be read as error rates.

Two steps (URLs are inert text - never open them):
  python -m src.error_analysis_summary --model E1_url_only        -> writes the worksheet to fill in
  (fill `your_category` and `evidence_note` in the CSV, <= 6 distinct categories)
  python -m src.error_analysis_summary --model E1_url_only --summarise
Models: E1_url_only (the operational demo model) or E3_fusion (the secondary model; the original export).
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import config

K = 10
MAX_CATEGORIES = 6
SHORT = {"E1_url_only": "E1", "E3_fusion": "E3"}


def select_errors(preds: pd.DataFrame, urls: pd.DataFrame, name: str, k: int = K) -> pd.DataFrame:
    """Most-confident false positives / false negatives at the locked threshold (uses SAVED predictions)."""
    d = preds.merge(urls[["original_qr_id", "url"]], on="original_qr_id", validate="one_to_one")
    score, pred = f"{name}_score", f"{name}_pred"
    fp = d[(d["label"] == config.LABEL_BENIGN) & (d[pred] == 1)].sort_values([score, "original_qr_id"],
                                                                            ascending=[False, True]).head(k)
    fn = d[(d["label"] == config.LABEL_MALICIOUS) & (d[pred] == 0)].sort_values([score, "original_qr_id"]).head(k)
    out = pd.concat([fp.assign(error_type="false_positive"), fn.assign(error_type="false_negative")])
    return pd.DataFrame({
        "model": name, "row_id": out["original_qr_id"], "error_type": out["error_type"],
        "true_label": out["label"].map({config.LABEL_BENIGN: "benign", config.LABEL_MALICIOUS: "suspicious/malicious"}),
        "prediction": out[pred].map({0: "benign (below threshold)", 1: "malicious (at/above threshold)"}),
        "risk_score": out[score].round(3), "url_length": out["url_length"],
        "url_TEXT_ONLY_do_not_open": out["url"], "your_category": "", "evidence_note": ""}).reset_index(drop=True)


def summarise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the filled worksheet; return (summary rows, category counts by error type)."""
    cats = df["your_category"].fillna("").astype(str).str.strip().str.lower()
    notes = df["evidence_note"].fillna("").astype(str).str.strip()
    if (cats == "").any() or (notes == "").any():
        raise SystemExit(f"{int((cats == '').sum())} rows lack a category and {int((notes == '').sum())} lack a note.")
    if cats.nunique() > MAX_CATEGORIES:
        raise SystemExit(f"{cats.nunique()} distinct categories; merge to at most {MAX_CATEGORIES}.")
    out = df.assign(category=cats, note=notes).drop(columns=["your_category", "evidence_note"])
    counts = out.groupby(["error_type", "category"]).size().rename("n_rows").reset_index()
    return out, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="E1_url_only", choices=list(SHORT))
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    short = SHORT[a.model]
    sheet = config.TABLES_DIR / f"error_analysis_{short}_worksheet.csv"
    if not a.summarise:
        preds = pd.read_csv(config.TABLES_DIR / "test_predictions.csv")
        urls = pd.read_csv(config.URLS_CSV, keep_default_na=False)
        ws = select_errors(preds, urls, a.model)
        old = config.TABLES_DIR / "error_analysis_E3.csv"          # reuse labels already given to the same rows
        if old.exists():
            prev = pd.read_csv(old, keep_default_na=False)
            if "your_category" in prev:
                m = prev.set_index("original_qr_id")["your_category"].to_dict()
                ws["your_category"] = ws["row_id"].map(m).fillna("")
        ws.to_csv(sheet, index=False)
        n_fp = int(((preds["label"] == 0) & (preds[f"{a.model}_pred"] == 1)).sum())
        n_fn = int(((preds["label"] == 1) & (preds[f"{a.model}_pred"] == 0)).sum())
        print(f"{a.model}: {n_fp} false positives and {n_fn} false negatives in total on the 300 test rows; "
              f"worksheet holds the {K} most confident of each.")
        if old.exists():
            e3 = set(pd.read_csv(old)["original_qr_id"])
            print(f"Overlap with the existing E3 export: {len(e3 & set(ws['row_id']))} of {len(e3)} rows; "
                  f"{int((ws['your_category'] != '').sum())} rows pre-filled with your earlier category.")
        print(f"\nFill `your_category` and `evidence_note` in {sheet.relative_to(config.PROJECT_ROOT)} "
              f"(URLs are text only), then run again with --summarise.")
        return
    summary, counts = summarise(pd.read_csv(sheet, keep_default_na=False))
    summary.to_csv(config.TABLES_DIR / "error_analysis_summary.csv", index=False)
    counts.to_csv(config.TABLES_DIR / "error_analysis_category_counts.csv", index=False)
    print(counts.to_string(index=False))
    print("\nManual qualitative review of selected (most confident) errors - not error rates.")


if __name__ == "__main__":
    main()
