"""Validated image download: temp file -> size check -> strict decode -> sha256 -> atomic move."""
from __future__ import annotations
import datetime as dt, os, tempfile, time
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

from ..provenance.hashing import sha256_file


def fetch_validated(url: str, dest: str | Path, retries: int = 3, timeout: int = 60, throttle_s: float = 0.0) -> dict:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return _record(dest, url, "ok", "cached")
    last = None
    for attempt in range(1, retries + 1):
        tmp = None
        try:
            time.sleep(throttle_s)
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout) as r:
                expected = r.headers.get("Content-Length")
                fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
                with os.fdopen(fd, "wb") as f:
                    f.write(r.read())
            if expected and os.path.getsize(tmp) != int(expected):
                raise IOError("size mismatch (truncated download)")
            with Image.open(tmp) as im:   # strict decode: verify() then a full load()
                im.verify()
            with Image.open(tmp) as im:
                im.load()
            os.replace(tmp, dest)
            return _record(dest, url, "ok", "downloaded", attempt)
        except Exception as e:  # noqa: BLE001
            last = str(e)
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
    return {"image_path": str(dest), "availability_status": "failed", "error": last,
            "retrieval_source": url, "retrieval_timestamp": _now()}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _record(path: Path, url: str, status: str, how: str, attempt: int = 0) -> dict:
    with Image.open(path) as im:
        w, h, fmt = im.width, im.height, im.format
    return {"image_path": str(path), "image_sha256": sha256_file(path), "image_width": w, "image_height": h,
            "image_format": fmt, "availability_status": status, "retrieval_source": url,
            "retrieval_timestamp": _now(), "how": how, "attempt": attempt}
