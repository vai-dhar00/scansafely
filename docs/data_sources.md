# Data Sources

**Safety handling.** The raw files contain real phishing URLs. They are handled as text only.
- Never click, Cmd+click, copy into a browser, or preview a URL from these files. VS Code and Excel turn URLs into clickable links, so view the data through pandas instead.
- `data/raw/` is excluded from git (see `.gitignore`), so the raw lists are not re-published.
- Many phishing URLs in these datasets are years old. Their labels describe them *at collection time*, and the sites may now be offline or have changed owner. This is a stated limitation.

## 1. PhiUSIIL Phishing URL Dataset
- **Publication:** Prasad, A. & Chandra, S. (2024). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. *Computers & Security*, 136, 103545. [verify the full reference before the poster]
- **Where:** UCI Machine Learning Repository, dataset 967
- **Licence:** CC BY 4.0
- **Size (per UCI page):** 235,795 URLs, of which 134,850 legitimate and 100,945 phishing
- **Label encoding:** `label` 1 = legitimate, 0 = phishing (**inverted** relative to ours; handled in `config.py`)
- **Saved as:** `data/raw/phiusiil.csv`
- **Downloaded on:** 28/09/26

## 2. Web Page Phishing Detection (Hannousse & Yahiouche)
- **Publication:** Hannousse, A. & Yahiouche, S. (2021). Towards benchmark datasets for machine learning based website phishing detection: An experimental study. *Engineering Applications of Artificial Intelligence*. [verify volume/pages]
- **Where:** Mendeley Data, doi:10.17632/c2gw7fy2j4.3 (version 3, 2021-06-25). URLs were collected in May 2020.
- **Licence:** CC BY 4.0
- **Size (per Mendeley page):** 11,430 URLs, 50% phishing / 50% legitimate
- **Label encoding:** `status` = `legitimate` / `phishing`
- **Saved as:** `data/raw/hannousse.csv`
- **Downloaded on:** 28/09/26
- **Note:** we use only the `url` and `status` columns. The dataset's 87 pre-computed features are **not** used. Some of them came from live web and third-party lookups, which our offline design excludes, and we compute our own features.
