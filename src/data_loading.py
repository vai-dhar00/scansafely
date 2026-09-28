"""
Load, clean and sample raw URL datasets into the dataset of record (urls.csv).

SAFETY: URLs are handled as plain text only. Nothing here opens, requests or
resolves a URL. Do not click URLs in the CSV files: some are real phishing links.

Pipeline:  raw CSV -> standardise columns/labels -> clean -> deduplicate
           -> balanced seeded sample -> assign original_qr_id -> urls.csv
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import tldextract

from src import config

# OFFLINE public-suffix parser. suffix_list_urls=() forces the snapshot bundled with
# the package; the default constructor would DOWNLOAD the list from the internet.
_TLD = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
_DEFAULT_PORTS = {"http": 80, "https": 443}


def canonicalize_url(url: str) -> str:
    """Deterministic canonical form used ONLY for duplicate checks and grouping.
    The original string is kept separately and is what gets encoded in the QR.
    Lowercases scheme + host, drops default ports, empty path -> '/'. Query and
    fragment are left untouched. No network access."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = parts.port if parts.port and parts.port != _DEFAULT_PORTS.get(scheme) else None
    netloc = host + (f":{port}" if port else "")
    if parts.username:  # keep user-info (e.g. 'user@host'), a known phishing trick
        netloc = parts.username + (f":{parts.password}" if parts.password else "") + "@" + netloc
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, parts.fragment))


def registered_domain(url: str) -> str:
    """e.g. 'a.b.example.co.uk' -> 'example.co.uk'. IP hosts are returned as-is."""
    host = (urlsplit(url.strip()).hostname or "").lower()
    ext = _TLD(host)
    return ext.top_domain_under_public_suffix or host


def url_template(url: str) -> str:
    """Host-free path/query 'shape' for near-duplicate audits (e.g. the same phishing
    kit on different domains): digit runs of 3+ -> <N>, query values -> <V>."""
    parts = urlsplit(canonicalize_url(url))
    path = re.sub(r"\d{3,}", "<N>", parts.path.lower())
    query = "&".join(sorted(p.split("=", 1)[0] + "=<V>" for p in parts.query.split("&") if p))
    return path + ("?" + query if query else "")


class DatasetError(Exception):
    """Raised with a human-readable message when a raw file is missing or malformed."""


def load_raw_dataset(name: str) -> pd.DataFrame:
    """Read one raw dataset and return columns: url (str), label (0/1), source (str)."""
    if name not in config.RAW_DATASETS:
        raise DatasetError(f"Unknown dataset '{name}'. Options: {list(config.RAW_DATASETS)}")
    spec = config.RAW_DATASETS[name]
    path: Path = config.RAW_DIR / spec["file"]
    if not path.exists():
        raise DatasetError(
            f"Missing file: {path.relative_to(config.PROJECT_ROOT)}\n"
            f"Download the '{name}' dataset and save it with exactly this name."
        )

    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in (spec["url_column"], spec["label_column"]) if c not in df.columns]
    if missing:
        raise DatasetError(
            f"{spec['file']} is missing column(s) {missing}.\n"
            f"Columns found (first 15): {list(df.columns)[:15]}\n"
            f"Fix 'url_column'/'label_column' for '{name}' in src/config.py."
        )

    raw_labels = df[spec["label_column"]]
    if raw_labels.dtype == object:
        raw_labels = raw_labels.astype(str).str.strip().str.lower()
    mapped = raw_labels.map(spec["label_map"])
    unmapped = sorted(set(raw_labels[mapped.isna()].astype(str)))[:10]
    if unmapped:
        raise DatasetError(f"{spec['file']}: label values not in label_map: {unmapped}")

    return pd.DataFrame({
        "url": df[spec["url_column"]].astype(str),
        "label": mapped.astype(int),
        "source": name,
    })


def clean_urls(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove unusable rows and duplicates. Returns the cleaned frame and a report of what was removed."""
    report: dict[str, int] = {"rows_in": len(df)}
    df = df.copy()
    df["url"] = df["url"].str.strip()

    bad = df["url"].isin(["", "nan", "None"]) | df["url"].str.contains(r"\s", regex=True)
    report["dropped_empty_or_whitespace"] = int(bad.sum())
    df = df[~bad]

    # A QR payload over ~2,000 chars is impractical to scan; very long rows are usually data errors.
    too_long = df["url"].str.len() > 2000
    report["dropped_over_2000_chars"] = int(too_long.sum())
    df = df[~too_long]

    # Duplicates and label conflicts are checked on the CANONICAL form, so
    # 'HTTP://Example.com' and 'http://example.com/' count as the same URL.
    df["canonical_url"] = df["url"].map(canonicalize_url)

    # Same URL with BOTH labels = contradictory ground truth -> remove every copy.
    labels_per_url = df.groupby("canonical_url")["label"].nunique()
    conflicting = labels_per_url[labels_per_url > 1].index
    report["dropped_conflicting_label_rows"] = int(df["canonical_url"].isin(conflicting).sum())
    df = df[~df["canonical_url"].isin(conflicting)]

    before = len(df)
    df = df.drop_duplicates(subset="url", keep="first")
    report["dropped_exact_duplicates"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset="canonical_url", keep="first")
    report["dropped_canonical_duplicates"] = before - len(df)

    report["rows_out"] = len(df)
    report["benign_out"] = int((df["label"] == config.LABEL_BENIGN).sum())
    report["malicious_out"] = int((df["label"] == config.LABEL_MALICIOUS).sum())
    return df.reset_index(drop=True), report


def sample_balanced(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    """Draw the same number of URLs from each class, reproducibly."""
    parts = []
    for label in (config.LABEL_BENIGN, config.LABEL_MALICIOUS):
        pool = df[df["label"] == label]
        if len(pool) < n_per_class:
            raise DatasetError(
                f"Only {len(pool)} {config.LABEL_NAMES[label]} URLs after cleaning; "
                f"N_PER_CLASS={n_per_class}. Lower N_PER_CLASS in config.py."
            )
        parts.append(pool.sample(n=n_per_class, random_state=seed))
    # Shuffle so the file is not ordered by class.
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def build_urls_csv(dataset: str | None = None) -> pd.DataFrame:
    """Create data/processed/urls.csv from the primary dataset and log what was done."""
    dataset = dataset or config.PRIMARY_DATASET
    config.ensure_directories()

    raw = load_raw_dataset(dataset)
    cleaned, report = clean_urls(raw)
    sample = sample_balanced(cleaned, config.N_PER_CLASS, config.RANDOM_SEED)

    sample.insert(0, "original_qr_id", [f"Q{i:05d}" for i in range(len(sample))])
    sample["registered_domain"] = sample["url"].map(registered_domain)  # Day 2 split group key
    sample["url_template"] = sample["url"].map(url_template)            # near-duplicate audit
    sample["date_collected"] = date.today().isoformat()  # date the raw file was processed
    sample["split"] = ""                                 # assigned on Day 2, grouped + stratified
    sample.to_csv(config.URLS_CSV, index=False)

    # Domain concentration: large domain groups make grouped splits lumpy (checked on Day 2).
    per_domain = sample.groupby("registered_domain").size().sort_values(ascending=False)
    report["sample_unique_domains"] = int(per_domain.size)
    report["sample_largest_domain_group"] = int(per_domain.iloc[0])
    report["sample_domains_with_5plus_urls"] = int((per_domain >= 5).sum())

    log = {
        "step": "build_urls_csv",
        "dataset": dataset,
        "citation": config.RAW_DATASETS[dataset]["citation"],
        "random_seed": config.RANDOM_SEED,
        "n_per_class": config.N_PER_CLASS,
        "cleaning_report": report,
        "output": str(config.URLS_CSV.relative_to(config.PROJECT_ROOT)),
        "date": date.today().isoformat(),
    }
    (config.TABLES_DIR / "data_cleaning_report.json").write_text(json.dumps(log, indent=2))
    return sample


if __name__ == "__main__":
    try:
        out = build_urls_csv()
    except DatasetError as err:
        raise SystemExit(f"ERROR: {err}")
    print(f"Wrote {config.URLS_CSV.relative_to(config.PROJECT_ROOT)} with {len(out)} rows")
    print(out["label"].map(config.LABEL_NAMES).value_counts().to_string())
    rep = json.loads((config.TABLES_DIR / "data_cleaning_report.json").read_text())["cleaning_report"]
    print(f"Canonical duplicates removed: {rep['dropped_canonical_duplicates']}")
    print(f"Unique registered domains in sample: {rep['sample_unique_domains']} "
          f"(largest group: {rep['sample_largest_domain_group']} URLs; "
          f"domains with 5+ URLs: {rep['sample_domains_with_5plus_urls']})")
    print("Cleaning report: outputs/tables/data_cleaning_report.json")
