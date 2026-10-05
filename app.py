"""
ScanSafely demo (Day 4b). Run:  streamlit run app.py

SAFETY (D25/D28): offline, in-memory, no network calls, no logging of payloads or images, uploaded filename
never used, decoded payload shown ONLY as plain text via st.text (no links, no markdown, no copy button).
E1 (URL-only) is the only operational model; there is no QR-image-quality judgement and no E3 score.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from src import config
from src import demo_logic as dl
from src import explainability as ex

st.set_page_config(page_title="ScanSafely", page_icon=None, layout="centered")


@st.cache_resource(show_spinner=False)
def get_model() -> dict:
    return ex.load_e1()          # refuses to run if the model file differs from its locked hash


st.title("ScanSafely")
st.caption("Evaluating URL and QR-Image Signals for Pre-Navigation Quishing Risk Screening")
st.info(f"**Prototype notice.** {config.DISCLAIMER} Everything runs locally: the QR code is decoded on this "
        "machine, the link is never opened, and nothing is uploaded, saved or logged.")

try:
    bundle = get_model()
except Exception as exc:  # noqa: BLE001
    st.error("The locked E1 model could not be loaded or verified, so the demo cannot run.")
    st.caption(type(exc).__name__)
    st.stop()

upload = st.file_uploader("Upload an image containing a QR code (PNG, JPG or WEBP, up to 5 MB)",
                          type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=False)

if upload is not None:
    try:
        res = dl.analyse_upload(upload.getvalue(), bundle)     # the filename is never read
    except dl.UploadRejected as rej:
        st.error(str(rej))
        st.stop()

    w, h = res["image_size"]
    st.markdown(f"**Analysis status:** {res['status']}")
    if res["kind"] == "undecodable":
        st.warning(res["message"])
        st.caption(f"Image size: {w} x {h} pixels.")
    else:
        dec = ("standard decoder" if res["status"] != ex.STATUS_FALLBACK else "fallback decoder")
        st.caption(f"QR payload decoded locally using the {dec}. Image size: {w} x {h} pixels.")
        st.markdown("**Decoded payload** (plain text, not opened):")
        st.text(res["payload_display"])

        if res["explanation"] is None:
            st.warning(res["message"])
        else:
            e = res["explanation"]
            box = {"LOW RISK": st.info, "SUSPICIOUS": st.warning, "HIGH RISK": st.error}[e.band]
            box(f"**Risk assessment: {e.band}**")
            c1, c2 = st.columns(2)
            c1.metric("Model risk score", f"{e.score:.2f}")
            c2.markdown(f"Prototype bands: LOW RISK below {config.RISK_THRESHOLDS['suspicious']:.2f}; "
                        f"SUSPICIOUS {config.RISK_THRESHOLDS['suspicious']:.2f} to below "
                        f"{config.RISK_THRESHOLDS['high_risk']:.2f}; HIGH RISK {config.RISK_THRESHOLDS['high_risk']:.2f} or above.")
            st.caption(res["score_note"])
            st.markdown(f"**Recommendation:** {e.recommendation}")

            st.subheader("Why this result?")
            st.markdown("**What pushed the score up or down, by feature group** "
                        "(positive = raises the risk score; log-odds units)")
            names, vals = zip(*sorted(e.group_contributions.items(), key=lambda kv: kv[1]))
            fig, ax = plt.subplots(figsize=(6.4, 2.6))
            ax.barh(names, vals, color=["#b5392b" if v > 0 else "#2f6db5" for v in vals])
            ax.axvline(0, color="#444", lw=0.8)
            ax.set_xlabel("lowers risk score  <-  effect on score  ->  raises risk score")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(res["correlation_note"])

            st.markdown("**Largest individual contributions** (individual features can offset each other, "
                        "so treat these as less reliable than the group totals)")
            if e.raising:
                st.markdown("Raising the score:\n" + "\n".join(f"- {t}" for t in e.raising))
            if e.lowering:
                st.markdown("Lowering the score:\n" + "\n".join(f"- {t}" for t in e.lowering))
            if not (e.raising or e.lowering):
                st.markdown("No single feature stood out.")
            st.caption(res["caveat"])

    st.divider()
    st.caption(config.DISCLAIMER)

with st.expander("Research comparison (study findings, not a live prediction)"):
    st.markdown(
        "A QR-image fusion model was evaluated during this study. It did not improve clean held-out "
        "discrimination over URL features and showed higher false-positive sensitivity when image features "
        "changed under tested distortions. It is therefore not used for the prototype recommendation.")
    for name, cap in [("model_comparison.png", "Model comparison on the frozen test set"),
                      ("robustness_e1_vs_e3.png", "URL-only vs fusion under controlled distortions")]:
        path = config.FIGURES_DIR / name
        if path.exists():
            st.image(str(path), caption=cap)

with st.expander("How this works and what it cannot do"):
    st.markdown(
        "- The QR image is decoded locally; the decoded text is treated as inert text and is never opened.\n"
        "- Only `http://` and `https://` payloads up to 2,000 characters are scored, by a URL-only logistic "
        "regression trained on a 2020 labelled URL collection. Other payloads are shown as text only.\n"
        "- The risk score is a prototype model output, not a probability that a destination is malicious.\n"
        "- Explanations show what the model learned from this dataset; some learned associations may be "
        "dataset-specific and are not general phishing indicators.\n"
        "- A LOW RISK result is not a guarantee of safety, and a HIGH RISK result is not proof of malice.")
