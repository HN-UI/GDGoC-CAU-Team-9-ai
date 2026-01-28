# app/utils/image_io.py

import mimetypes
from typing import Tuple, Optional
from urllib.parse import urlparse

import requests


def guess_mime(name: str, default: str = "image/jpeg") -> str:
    """
    파일명/URL 확장자로 MIME 추정
    """
    mime, _ = mimetypes.guess_type(name)
    return mime or default


def is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def load_image_from_file(path: str, mime_type: Optional[str] = None) -> Tuple[bytes, str]:
    """
    로컬 파일 -> (bytes, mime)
    """
    if mime_type is None:
        mime_type = guess_mime(path)

    with open(path, "rb") as f:
        data = f.read()

    return data, mime_type


def load_image_from_url(url: str, timeout_sec: int = 15) -> Tuple[bytes, str]:
    """
    URL -> (bytes, mime)
    - 헤더 Content-Type 우선
    - 없거나 이상하면 확장자로 추정
    """
    r = requests.get(url, timeout=timeout_sec)
    r.raise_for_status()

    content_type = (r.headers.get("Content-Type") or "").split(";")[0].strip()
    if content_type.startswith("image/"):
        mime = content_type
    else:
        mime = guess_mime(url)

    return r.content, mime


def load_image(source: str, timeout_sec: int = 15) -> Tuple[bytes, str]:
    """
    source가 URL이면 다운로드, 아니면 로컬 파일로 읽기.
    """
    if is_url(source):
        return load_image_from_url(source, timeout_sec=timeout_sec)
    return load_image_from_file(source)
