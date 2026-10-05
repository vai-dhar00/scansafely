# ScanSafely

**An Explainable Multimodal Framework for Pre-Navigation Quishing Detection**
*Undergraduate IT research proof-of-concept. Not a production security tool.*

## Research question

Can an explainable multimodal detector combine QR-code structural information and decoded-URL lexical characteristics to identify potentially risky QR codes before navigation, while remaining useful under realistic QR-image distortions?

## Contribution (scoped claim)

We evaluate whether combining locally decoded URL lexical signals with QR-image characteristics improves pre-navigation risk screening. We also measure the trade-off between QR decodability and classification performance under controlled image distortions.

We do **not** claim to be the first such system. We do not claim complete protection or production readiness.

## Safety boundaries (non-negotiable)

1. Decoded URLs are **inert text strings**. No code in this repository opens, visits, requests, previews, resolves (DNS), or runs WHOIS on any URL.
2. No network calls anywhere in `src/` or `app.py` (`config.ALLOW_NETWORK = False`).
3. The demo shows the payload as escaped, **non-clickable** text. It never uses links, `st.link_button`, redirects or `webbrowser`.
4. Malicious URL lists come only from reputable published datasets. They are downloaded as files and handled offline.
5. Risk labels (LOW RISK / SUSPICIOUS / HIGH RISK) are **prototype policy thresholds** on a binary model's probability. They are not ground-truth classes.

## Reproducibility rules

- A single seed is used everywhere: `RANDOM_SEED = 42` in `src/config.py`.
- Splits are grouped by `original_qr_id` and stratified by label (70/15/15). All distorted variants stay in their original's split.
- **The test set is frozen before any model tuning.** It is never used for tuning or model selection.
- Distortions are generated only from test-split images.
- All results, including failures and decode errors, are logged to `outputs/`.
- Exact package versions are recorded in `requirements-lock.txt`.

## Planned workflow

| Day | Work |
|---|---|
| 1 | Environment setup, schema, data collection (offline), QR generation, decoder, feature extractors, first `features.csv` |
| 2 | Grouped seeded split, test freeze, E1 URL-only / E2 QR-only / E3 multimodal models, first evaluation table, payload-length confound check |
| 3 | Test-only distortions, decode-rate and post-decode classification robustness |
| 4 | SHAP explanations, Streamlit demo, figures |
| 5 | Cross-check every number, poster, limitations, pitch |

## Experiments

- **E1:** URL lexical features only
- **E2:** QR/image features only
- **E3:** URL and QR/image features combined (multimodal)

All three use the same classifier family for a fair comparison. Primary metrics are F1 and malicious-class recall, plus false-positive counts. We also report decode success rate for each distortion and classification performance on the decoded cases.

## Known limitation to test, not hide

If every QR image comes from the same generator, the QR structure (version and module density) is mostly determined by payload length. The QR-only model may therefore act as a URL-length proxy rather than detecting anything visually "malicious" about a QR code. We measure this correlation directly and report it.

## Project structure

```
scansafely/
├── README.md, requirements.txt, .gitignore, check_setup.py, app.py
├── docs/data_dictionary.md
├── data/{raw, processed, qr_images, distorted_test_images}/
├── models/
├── outputs/{figures, tables, shap, demo_screenshots}/
├── src/   config.py + one module per pipeline stage
├── notebooks/
└── tests/
```

## Setup

See Phase 1 instructions. Then verify with `python check_setup.py`.

## Data sources

*To be completed on Day 1, Phase 2:* name, URL of the dataset's publication page, licence, download date, and number of rows used.
