import io

import numpy as np
import pytest
import qrcode
from PIL import Image

from src import demo_logic as dl
from src import explainability as ex
from tests.test_explainability import _synthetic_bundle

pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")

@pytest.fixture(scope="module")
def bundle():
    return _synthetic_bundle()[0]


def _png(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO(); img.save(buf, format=fmt); return buf.getvalue()


def _qr(text: str) -> Image.Image:
    q = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=4)
    q.add_data(text); q.make(fit=True)
    return q.make_image(fill_color="black", back_color="white").convert("RGB")


@pytest.mark.parametrize("data,msg", [
    (b"", "empty"), (b"not an image at all", "readable"),
    (_png(Image.new("RGB", (64, 64)), "GIF"), "Only PNG"),
    (_png(Image.new("L", (10, 200))), "too small"),
    (_png(Image.new("L", (7000, 100))), "too large"),
    (_png(Image.new("L", (5000, 4000))), "too large"),
    (b"\x00" * (dl.MAX_UPLOAD_BYTES + 1), "larger than"),
])
def test_rejections(data, msg):
    with pytest.raises(dl.UploadRejected, match=msg):
        dl.validate_and_load(data)


def test_http_url_end_to_end(bundle):
    url = "https://www.example.com/a/b?x=1"
    out = dl.analyse_upload(_png(_qr(url)), bundle)
    assert out["kind"] == "url" and out["payload_display"] == url
    assert out["status"] == ex.STATUS_COMPLETE and out["explanation"].band in ("LOW RISK", "SUSPICIOUS", "HIGH RISK")
    assert out["image_size"][0] > 100


def test_jpeg_and_webp_accepted(bundle):
    for fmt in ("JPEG", "WEBP"):
        out = dl.analyse_upload(_png(_qr("https://example.com/x"), fmt), bundle)
        assert out["kind"] == "url", fmt


def test_non_url_payload_has_no_score(bundle):
    out = dl.analyse_upload(_png(_qr("WIFI:S:home;T:WPA;P:secret;;")), bundle)
    assert out["kind"] == "non_url" and out["explanation"] is None and out["status"] == ex.STATUS_NO_URL
    assert "unavailable" in out["message"]


def test_blank_image_is_undecodable_with_no_label(bundle):
    out = dl.analyse_upload(_png(Image.new("L", (300, 300), 255)), bundle)
    assert out["kind"] == "undecodable" and out["explanation"] is None and out["status"] == ex.STATUS_UNDECODABLE
    assert "band" not in out and out["message"] == ex.UNDECODABLE_MESSAGE


def test_transparent_png_composited_on_white(bundle):
    rgba = _qr("https://example.com/t").convert("RGBA")
    arr = np.array(rgba); arr[(arr[..., :3] == 255).all(-1), 3] = 0     # white -> transparent
    out = dl.analyse_upload(_png(Image.fromarray(arr, "RGBA")), bundle)
    assert out["kind"] == "url"


def test_demo_modules_have_no_network_or_disk_writes():
    import pathlib
    import re
    src = pathlib.Path(dl.__file__).read_text()
    for bad in ("import requests", "urllib.request", "import webbrowser", "import socket", ".save(", "tempfile"):
        assert bad not in src, bad
    assert not re.search(r"(?<![.\w])open\(", src), "builtin open() (file access) found"  # Image.open(BytesIO) is fine
