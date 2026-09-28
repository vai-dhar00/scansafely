# ScanSafely - Data Dictionary

Two CSV files, both in `data/processed/`.

- **`urls.csv`**: one row per original URL string. This is the dataset of record.
- **`features.csv`**: one row per QR image (clean original or distorted variant). All features are computed offline.

Every row in `features.csv` links back to `urls.csv` through `original_qr_id`. **The split is assigned per `original_qr_id`, never per image**, so a clean QR and all of its distorted variants always share a split.

---

## 1. `urls.csv` (dataset of record)

| Column | Type | Example | Description |
|---|---|---|---|
| `original_qr_id` | str | `Q000123` | Unique ID per URL string. Primary key. |
| `url` | str | `https://example.com/a` | URL, stored as **inert text only**. Never opened, requested or resolved. |
| `label` | int | `0` / `1` | 0 = benign, 1 = suspicious/malicious (label assigned by the source). |
| `source` | str | `benign_list_A` | Short name of the dataset or list the URL came from (details in README). |
| `date_collected` | str | `2026-09-28` | ISO date when the list was downloaded. Labels can go stale over time. |
| `split` | str | `train` / `val` / `test` | Assigned once on Day 2 with `RANDOM_SEED`, grouped and stratified. The test split is frozen after this. |

## 2. `features.csv` (one row per image)

### Identification and experiment metadata

| Column | Type | Description |
|---|---|---|
| `qr_id` | str | Unique image ID, e.g. `Q000123__clean` or `Q000123__gaussian_blur__moderate`. |
| `original_qr_id` | str | Links to `urls.csv`. Used for grouped splitting. |
| `label` | int | Copied from `urls.csv`. |
| `split` | str | Copied from `urls.csv`. Never recomputed per image. |
| `image_path` | str | Path **relative to the project root**, so it works on every teammate's machine. |
| `distortion_type` | str | `none`, `gaussian_blur`, `rotation`, `jpeg`, `low_resolution`, `perspective`, `occlusion`. |
| `distortion_severity` | str | `none`, or a level from `config.DISTORTIONS` (e.g. `moderate`, `q30`, `+10`). |

### Decoder output (added to the suggested schema)

These columns make decode failures visible instead of being silently dropped.

| Column | Type | Description |
|---|---|---|
| `decode_success` | int (0/1) | 1 if the local decoder returned a non-empty payload. |
| `decoded_payload` | str | Decoded text (inert). Empty if decoding failed. |
| `decoded_matches_original` | int (0/1) | 1 if `decoded_payload == url` exactly. |
| `decoder_name` | str | e.g. `opencv`. Recorded so results are reproducible. |

### URL lexical features (computed from the **decoded** payload, as in the real pipeline; NaN if decoding failed)

| Column | Type | Description |
|---|---|---|
| `url_length` | int | Total characters. |
| `url_hostname_length` | int | Characters in the hostname. |
| `url_dot_count` | int | Number of `.` in the whole URL. |
| `url_subdomain_count` | int | Hostname labels minus 2 (floored at 0). A simple approximation: it does not use the Public Suffix List, so `a.co.uk` is overcounted. This is documented as a limitation. |
| `url_hyphen_count` | int | Number of `-`. |
| `url_digit_count` | int | Number of digits. |
| `url_digit_ratio` | float | Digits divided by length. |
| `url_special_char_count` | int | Characters that are not letters, digits or `.`/`/`/`:`. |
| `url_at_count`, `url_question_count`, `url_equals_count`, `url_ampersand_count`, `url_percent_count` | int | Counts of `@ ? = & %`. |
| `url_path_depth` | int | Non-empty path segments. |
| `url_query_param_count` | int | Number of query parameters. |
| `url_has_https` | int (0/1) | Scheme is `https`. |
| `url_has_ip` | int (0/1) | Hostname is an IPv4 address. |
| `url_has_punycode` | int (0/1) | Contains `xn--`. |
| `url_entropy` | float | Shannon entropy (bits per character) of the URL string. |
| `url_suspicious_token_count` | int | Number of distinct tokens from `config.SUSPICIOUS_TOKENS` found. |
| `url_has_token_<word>` | int (0/1) | Flags for selected tokens (e.g. `login`, `verify`, `account`). |

### QR / image features (computed from the image pixels)

| Column | Type | Description |
|---|---|---|
| `qr_width`, `qr_height` | int | Image size in pixels. |
| `qr_aspect_ratio` | float | Width divided by height. |
| `qr_gray_mean`, `qr_gray_std` | float | Grayscale mean and standard deviation. |
| `qr_dark_ratio` | float | Fraction of pixels below `DARK_PIXEL_THRESHOLD` (128). |
| `qr_contrast` | float | Max minus min grayscale value, divided by 255. |
| `qr_blur_score` | float | Variance of the Laplacian (higher means sharper). |
| `qr_edge_density` | float | Fraction of Canny edge pixels. |
| `qr_estimated_module_density` | float | Estimated modules per side (method documented in `qr_features.py`). |
| `qr_version` | int or NaN | **Only filled if reliably obtained.** For clean generated images, we know the version from the generator. For distorted images it is left NaN rather than guessed. |
| `qr_quiet_zone_ratio` | float | Estimated blank border width divided by image width, if feasible. |

### Robustness-only columns (in `outputs/tables/robustness_results.csv`)

`predicted_label`, `predicted_probability`, `risk_label`, `inference_time_ms`, `classification_possible` (= `decode_success`), and `model_name`.

---

## Leakage controls built into this schema

1. **Grouped split.** `split` comes from `urls.csv`, keyed by `original_qr_id`, so variants cannot cross splits.
2. **Duplicate URLs.** Exact duplicates are removed *before* IDs are assigned. Near-duplicates (same URL, different tracking parameters) are a known remaining risk.
3. **Identical QR settings for every URL.** Error-correction level, box size and border come from `config.py`, so rendering settings cannot encode the label.
4. **Label-free features.** No feature column uses `source`, `label` or `date_collected`.
5. **Payload-length confound (key limitation).** When all images come from one generator, QR density and version are largely determined by payload length. `url_length` sits in `features.csv` so that on Day 2 we can measure how strongly the QR-only features track `url_length`, and test whether QR-only performance holds up once length is controlled for.
