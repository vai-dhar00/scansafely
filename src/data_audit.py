"""
URL-shape audit: do benign and malicious URLs differ in trivial structural ways?

Why: if benign URLs are mostly bare homepages ("https://site.com") and malicious
URLs have long paths, ANY model can separate them by length alone. The QR-only
model would then look strong just because QR density follows payload length.
We measure this BEFORE modelling and use it to choose the primary dataset.

Outputs:
  outputs/tables/url_shape_audit.csv
  outputs/figures/url_length_by_class.png

Offline only: URLs are parsed as strings with urllib.parse (no network).
"""
from __future__ import annotations

from urllib.parse import urlsplit

import matplotlib

matplotlib.use("Agg")  # render to file; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config
from src.data_loading import DatasetError, clean_urls, load_raw_dataset

CLASS_COLORS = {config.LABEL_BENIGN: "#2a78d6", config.LABEL_MALICIOUS: "#eb6834"}


def _shape_flags(url: str) -> dict[str, int]:
    """Simple structural flags for the audit (not the model's feature extractor)."""
    parts = urlsplit(url if "://" in url else "//" + url)
    path = parts.path or ""
    return {
        "has_scheme": int("://" in url),
        "is_https": int(parts.scheme.lower() == "https"),
        "has_path": int(path not in ("", "/")),
        "has_query": int(bool(parts.query)),
        "bare_domain": int(path in ("", "/") and not parts.query and not parts.fragment),
    }


def audit_dataset(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-class summary rows, the cleaned URL frame with lengths)."""
    df, _ = clean_urls(load_raw_dataset(name))
    df = df.copy()
    df["length"] = df["url"].str.len()
    flags = pd.DataFrame([_shape_flags(u) for u in df["url"]], index=df.index)
    df = pd.concat([df, flags], axis=1)

    # How well does URL length ALONE separate the classes? 0.5 = not at all, 1.0 = perfectly.
    auc = roc_auc_score(df["label"], df["length"])
    length_only_auc = max(auc, 1 - auc)

    rows = []
    for label, grp in df.groupby("label"):
        rows.append({
            "dataset": name,
            "class": config.LABEL_NAMES[label],
            "n": len(grp),
            "length_median": grp["length"].median(),
            "length_p25": grp["length"].quantile(0.25),
            "length_p75": grp["length"].quantile(0.75),
            "pct_https": round(100 * grp["is_https"].mean(), 1),
            "pct_has_path": round(100 * grp["has_path"].mean(), 1),
            "pct_has_query": round(100 * grp["has_query"].mean(), 1),
            "pct_bare_domain": round(100 * grp["bare_domain"].mean(), 1),
            "pct_missing_scheme": round(100 * (1 - grp["has_scheme"].mean()), 1),
            "length_only_auc": round(length_only_auc, 3),
        })
    return pd.DataFrame(rows), df


def plot_length_distributions(frames: dict[str, pd.DataFrame]) -> None:
    """One panel per dataset: URL length distribution for each class (log x-axis)."""
    fig, axes = plt.subplots(1, len(frames), figsize=(6.5 * len(frames), 4.2),
                             sharex=True, squeeze=False)
    bins = np.logspace(np.log10(5), np.log10(2000), 50)
    for ax, (name, df) in zip(axes[0], frames.items()):
        for label in (config.LABEL_BENIGN, config.LABEL_MALICIOUS):
            lengths = df.loc[df["label"] == label, "length"]
            ax.hist(lengths, bins=bins, density=True, histtype="step", linewidth=2,
                    color=CLASS_COLORS[label],
                    label=f"{config.LABEL_NAMES[label]} (median {lengths.median():.0f})")
        ax.set_xscale("log")
        ax.set_title(name, fontsize=14)
        ax.set_xlabel("URL length (characters, log scale)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=10)
    fig.suptitle("URL length by class, before sampling", fontsize=15)
    fig.tight_layout()
    out = config.FIGURES_DIR / "url_length_by_class.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)


def run_audit() -> pd.DataFrame:
    config.ensure_directories()
    summaries, frames = [], {}
    for name in config.RAW_DATASETS:
        try:
            summary, df = audit_dataset(name)
        except DatasetError as err:
            print(f"[skip] {name}: {err}\n")
            continue
        summaries.append(summary)
        frames[name] = df
    if not summaries:
        raise SystemExit("No datasets found in data/raw/. See the Phase 2 download steps.")

    table = pd.concat(summaries, ignore_index=True)
    table.to_csv(config.TABLES_DIR / "url_shape_audit.csv", index=False)
    plot_length_distributions(frames)
    return table


if __name__ == "__main__":
    result = run_audit()
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(result.to_string(index=False))
    print("\nSaved: outputs/tables/url_shape_audit.csv, outputs/figures/url_length_by_class.png")
