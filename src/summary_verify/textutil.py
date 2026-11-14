"""Text helpers copied verbatim from the service's minutes module (minutes.py), and only these:
MAX_TITLE_CHARS, _skill_body, meeting_title, _MEDIA_ID, _INVISIBLE and clean_text.

article_summary.py uses clean_text and _skill_body, and imports this module under the name it used for the
original (`mm`). meeting_title is what the service's summary endpoint did to the file name before passing it
on as the title; it is kept so that a caller can do the same. The rest of minutes.py (the meeting minutes'
own prompt, schema and validation) is not part of this export.
"""

import re
from pathlib import Path

MAX_TITLE_CHARS = 100


def _skill_body(path: Path) -> str:
    """SKILL.md without its front matter (name and description are for people browsing the skills)."""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        text = text.split("\n---", 1)[1].split("\n", 1)[1]
    return text.strip()


def meeting_title(file_name: str | None) -> str | None:
    """An upload's file name as the meeting title: no extension, one line, nothing that could end the tag."""
    if not file_name:
        return None
    stem = re.sub(r"\.(?=[A-Za-z0-9]{0,4}[A-Za-z])[A-Za-z0-9]{1,5}$", "", clean_text(file_name))  # not ".11"
    stem = re.sub(r"[\x00-\x1f\x7f<>]", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()[:MAX_TITLE_CHARS]
    if _MEDIA_ID.fullmatch(stem):  # WeChat's placeholder name (MEDIA_ID.ext): no title at all
        return None
    return stem or None


_MEDIA_ID = re.compile(r"(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z0-9_\-]{32,}")

_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\x00"))


def clean_text(text: str) -> str:
    """No NUL (PostgreSQL rejects it), zero-width characters or lone surrogates in stored text."""
    return text.translate(_INVISIBLE).encode("utf-8", "replace").decode("utf-8")
