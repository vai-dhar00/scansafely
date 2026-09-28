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
| `canonical_url` | str | `http://example.com/` | Lowercased scheme/host, default port removed, empty path -> `/`. Used ONLY for duplicate/conflict checks; the QR encodes the original `url`. |
| `registered_domain` | str | `example.co.uk` | Offline public-suffix parse (tldextract, bundled list). Split group key on Day 2. IP hosts kept as-is. |
| `url_template` | str | `/account/verify?id=<V>` | Host-free path/query shape (digit runs -> `<N>`, query values -> `<V>`) for the near-duplicate audit. |
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
| `decode_outcome` | str | `exact` / `mismatch` / `failure`. A mismatch is NOT a success; classification metrics use `exact` only. |
| `decoder_name` | str | e.g. `opencv`. Recorded so results are reproducible. |

### URL lexical features (computed from the **decoded** payload, as in the real pipeline; NaN if decoding failed)

| Column | Type | Description |
|---|---|---|
| `url_length` | int | Total characters. |
| `url_hostname_length` | int | Characters in the hostname. |
| `url_path_length`, `url_query_length` | int | Characters in the path / query string. |
| `url_dot_count` | int | Number of `.` in the whole URL. |
| `url_subdomain_count` | int | Labels in the subdomain part, using the public suffix list bundled with tldextract (offline). `login.secure.example.co.uk` gives 2. 0 for IP hosts. |
| `url_hyphen_count` | int | Number of `-`. |
| `url_digit_count` | int | Number of digits. |
| `url_digit_ratio` | float | Digits divided by length. |
| `url_special_char_count` | int | Characters that are not letters, digits or `.`/`/`/`:`. |
| `url_at_count`, `url_question_count`, `url_equals_count`, `url_ampersand_count`, `url_percent_count` | int | Counts of `@ ? = & %`. |
| `url_path_depth` | int | Non-empty path segments. |
| `url_query_param_count` | int | Number of query parameters. |
| `url_has_https` | int (0/1) | Scheme is `https`. |
| `url_has_ip` | int (0/1) | Hostname is an IPv4 or IPv6 address. |
| `url_has_punycode` | int (0/1) | Contains `xn--`. |
| `url_has_nonstandard_port` | int (0/1) | Explicit port other than 80/443. |
| `url_has_double_slash_path` | int (0/1) | `//` inside the path (a redirection trick). |
| `url_entropy` | float | Shannon entropy (bits per character) of the URL string. |
| `url_suspicious_token_count` | int | Number of distinct tokens from `config.SUSPICIOUS_TOKENS` found. |
| `url_has_token_<word>` | int (0/1) | Flags for `login`, `verify`, `account`, `secure`, `update`, `bank`. |

Full ordered list: `URL_FEATURES` in `src/url_features.py` (30 features).

### QR / image features (pixels only, D10/D13; full list: `QR_FEATURES` in `src/qr_features.py`)

All pixel statistics are computed on a standardised 512x512 grayscale copy, so they do not depend on upload resolution.

| Column | Role | Description |
|---|---|---|
| `qr_aspect_ratio` | model | Original width / height. |
| `qr_gray_mean`, `qr_gray_std` | model | Grayscale mean and standard deviation (0-1). |
| `qr_dark_ratio` | model | Fraction of pixels below `DARK_PIXEL_THRESHOLD` (128). |
| `qr_contrast` | model | (99th - 1st percentile grey level) / 255. Robust to single noisy pixels. |
| `qr_blur_score` | model | Variance of the Laplacian (higher = sharper). log1p before scaling. |
| `qr_edge_density` | model | Fraction of Canny edge pixels. |
| `qr_dark_component_density` | model | Dark connected blobs (Otsu threshold) per 10,000 px. log1p before scaling. |
| `qr_quiet_zone_ratio` | model | Smallest blank margin around the dark content / image side. |
| `qr_decoded_modules_per_side` | model | Size of the rectified code grid returned by the decoder (= 17 + 4 x version), read from pixels. NaN if decoding fails. log1p before scaling. |
| `qr_width`, `qr_height` | descriptive | Original pixel size. Not a model input (depends on upload resolution). |
| `qr_version_generator` | descriptive | Version chosen by the generator. Never a model input (D10). |

### Robustness-only columns (in `outputs/tables/robustness_results.csv`)

`predicted_label`, `predicted_probability`, `risk_label`, `inference_time_ms`, `classification_possible` (= `decode_success`), and `model_name`.

---

## Leakage controls built into this schema

1. **Grouped split.** `split` comes from `urls.csv`, keyed by `original_qr_id`, so variants cannot cross splits.
2. **Duplicate URLs.** Exact duplicates are removed *before* IDs are assigned. Near-duplicates (same URL, different tracking parameters) are a known remaining risk.
3. **Identical QR settings for every URL.** Error-correction level, box size and border come from `config.py`, so rendering settings cannot encode the label.
4. **Label-free features.** No feature column uses `source`, `label` or `date_collected`.
5. **Payload-length confound (key limitation).** When all images come from one generator, QR density and version are largely determined by payload length. `url_length` sits in `features.csv` so that on Day 2 we can measure how strongly the QR-only features track `url_length`, and test whether QR-only performance holds up once length is controlled for.
