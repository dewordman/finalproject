# src/text_build.py
"""Text preparation shared by training, experiments, bundle and inference.

Revision 2 keeps the Chapter Three principle (light normalisation, evidence stays
visible) but adds *declared, optional* clean-up steps that remove source and era
cues found by the audit:

* ``mask_terms`` can now be DELETED (default) instead of replaced by a token;
* dates, times and four-digit years can be removed;
* URLs and e-mail addresses can be removed (the Enron-Spam corpus is distributed
  with both already removed, so keeping them in the other sources would let a model
  recognise the source from the mere presence of a link);
* raw e-mail header remnants and HTML / quoted-printable debris can be removed.

Every step is controlled by an options dictionary so that the SAME dictionary can
be stored in the final bundle and re-used at prediction time.
"""
from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

#: Behaviour when ``options`` is ``None``: identical to the pre-revision code
#: (replace masked terms by the token "maskedterm"; no extra clean-up).
LEGACY_OPTIONS: Dict[str, Any] = {
    "mask_mode": "token",
    "strip_temporal_cues": False,
    "strip_urls": False,
    "strip_email_addresses": False,
    "strip_header_remnants": False,
    "decode_html_in_csv_sources": False,
}

OPTION_KEYS = tuple(LEGACY_OPTIONS)
MASK_TOKEN = "maskedterm"


def get_text_options(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Read the text-preparation options from ``cfg['text']``.

    A key that is ABSENT falls back to the legacy value (so old configurations
    still load). A key that is present but ``null`` raises ``ValueError``.
    """
    section = cfg.get("text")
    if not isinstance(section, dict):
        raise ValueError("Configuration is missing the 'text' section")
    options: Dict[str, Any] = {}
    for key, legacy in LEGACY_OPTIONS.items():
        if key in section:
            value = section[key]
            if value is None:
                raise ValueError(f"text.{key} must not be null")
        else:
            value = legacy
        options[key] = value
    if options["mask_mode"] not in ("delete", "token"):
        raise ValueError("text.mask_mode must be 'delete' or 'token'")
    for key in OPTION_KEYS:
        if key != "mask_mode" and not isinstance(options[key], bool):
            raise ValueError(f"text.{key} must be true or false")
    return options


def _resolve_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    resolved = dict(LEGACY_OPTIONS)
    if options:
        unknown = set(options) - set(OPTION_KEYS)
        if unknown:
            raise ValueError(f"Unknown text options: {sorted(unknown)}")
        resolved.update(options)
    return resolved


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def html_to_text(html_str: str) -> str:
    """Convert an HTML string to visible text using the lxml parser."""
    soup = BeautifulSoup(html_str, "lxml")
    return soup.get_text(separator=" ")


_HTML_MARKERS = ("<html", "<body", "<div", "<br", "<table", "<font", "<p>")
_QP_HEX = re.compile(r"=([0-9A-Fa-f]{2})")


def _looks_like_html(text: str) -> bool:
    lowered = text.lower()
    return sum(1 for marker in _HTML_MARKERS if marker in lowered) >= 2


def _decode_quoted_printable(text: str) -> str:
    """Decode =XX sequences and soft line breaks when the text clearly contains them."""
    if len(_QP_HEX.findall(text)) < 3:
        return text
    joined = re.sub(r"=\r?\n", "", text)
    return _QP_HEX.sub(lambda m: bytes.fromhex(m.group(1)).decode("latin-1"), joined)


def decode_html_and_qp(text: str) -> str:
    """Convert HTML-looking text to visible text and decode quoted-printable debris."""
    try:
        if _looks_like_html(text):
            text = html_to_text(text)
        return _decode_quoted_printable(text)
    except Exception:  # never fail on malformed input
        return text


# ---------------------------------------------------------------------------
# Header remnants
# ---------------------------------------------------------------------------

_HEADER_START = re.compile(
    r"^(?:from |return-path:|received:|x-[a-z0-9\-]+:|delivered-to:|message-id:|"
    r"content-type:|mime-version:|content-transfer-encoding:)", re.I)
_HEADER_FIELD = re.compile(r"^[A-Za-z][A-Za-z0-9\-]*:")
_HEADER_LINES = re.compile(
    r"^[ \t]*(?:x-spam-[^\n]*|x-authentication-warning:[^\n]*|x-mailer:[^\n]*)$",
    re.I | re.M)


def strip_header_remnants(text: str) -> Tuple[str, bool]:
    """Remove a leading raw header block and X-Spam/X-Mailer style lines.

    Returns ``(cleaned_text, removed_anything)``.
    """
    if not isinstance(text, str) or not text:
        return ("" if not isinstance(text, str) else text), False
    removed = False
    lines = text.splitlines()
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start < len(lines) and _HEADER_START.match(lines[start].strip()):
        i = start
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                break
            if (_HEADER_FIELD.match(line) or line.startswith((" ", "\t"))
                    or line.startswith("From ")):
                i += 1
                continue
            break
        while i < len(lines) and not lines[i].strip():
            i += 1
        lines = lines[i:]
        removed = True
    cleaned = "\n".join(lines)
    cleaned2 = _HEADER_LINES.sub(" ", cleaned)
    if cleaned2 != cleaned:
        removed = True
    return cleaned2, removed


# ---------------------------------------------------------------------------
# E-mail addresses and temporal cues
# ---------------------------------------------------------------------------

_EMAIL = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9\-]+(?:\.[a-z0-9\-]+)+", re.I)

_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
          r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
          r"dec(?:ember)?)")
_WEEKDAY = (r"(?:mon(?:day)?|tue(?:s(?:day)?)?|wed(?:nesday)?|thu(?:r(?:s(?:day)?)?)?|"
            r"fri(?:day)?|sat(?:urday)?|sun(?:day)?)")
_ZONE = r"(?:gmt|utc|est|edt|cst|cdt|mst|mdt|pst|pdt|wat|cet|cest|bst)"
_TIME = r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?"
_TAIL = rf"(?:\s+{_TIME})?(?:\s*(?:[+\-]\d{{4}}|{_ZONE}\b))?"

_TEMPORAL_PATTERNS = [
    # Mon, 5 Nov 2007 10:11:12 +0100 / 5th of November 2007
    re.compile(rf"(?:\b{_WEEKDAY}\.?,?\s+)?\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}\b\.?,?\s+\d{{2,4}}\b{_TAIL}", re.I),
    # Nov 5, 2007 10:11 / Monday, November 5th 2007
    re.compile(rf"(?:\b{_WEEKDAY}\.?,?\s+)?\b{_MONTH}\b\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b{_TAIL}", re.I),
    # Nov 5 / Nov 2007 / sep 2002
    re.compile(rf"\b{_MONTH}\b\.?\s+(?:\d{{1,2}}(?:st|nd|rd|th)?\b|\d{{4}}\b)", re.I),
    # weekday followed by a comma ("Mon, ...")
    re.compile(rf"\b{_WEEKDAY}\b\s*,", re.I),
    # ISO and numeric dates
    re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),
    re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b"),
    # clock times and numeric zone offsets
    re.compile(rf"\b{_TIME}(?=\W|$)", re.I),
    re.compile(r"(?<=\s)[+\-]\d{4}\b"),
    # any remaining bare four-digit year 1990-2039
    re.compile(r"\b(?:199\d|20[0-3]\d)\b"),
]


_URL = re.compile(r"(?:https?|ftp)://[^\s<>\"'\)\]]+|\bwww\.[^\s<>\"'\)\]]+", re.I)


def strip_urls(text: str) -> str:
    """Delete URLs (scheme-prefixed or starting with ``www.``); replaced by one space."""
    return _URL.sub(" ", text)


def strip_email_addresses(text: str) -> str:
    """Delete e-mail addresses (replaced by one space)."""
    return _EMAIL.sub(" ", text)


def strip_temporal_cues(text: str) -> str:
    """Delete dates, clock times, zone offsets and bare years 1990-2039.

    Other numbers are left alone, and the words "may" / "march" are only
    removed when they are part of a date pattern.
    """
    for pattern in _TEMPORAL_PATTERNS:
        text = pattern.sub(" ", text)
    return text


# ---------------------------------------------------------------------------
# Term masking
# ---------------------------------------------------------------------------

def _term_pattern(term: str) -> str:
    """Whole-word pattern; multi-word terms tolerate any whitespace between words."""
    escaped = re.escape(term.lower().strip())
    return escaped.replace(r"\ ", r"\s+")


def delete_terms(text: str, terms: Optional[List[str]], mode: str = "delete") -> str:
    """Remove (``mode='delete'``) or replace (``mode='token'``) whole-word terms.

    In delete mode a single-word term also removes the dotted host it belongs to
    (``monkey`` removes ``monkey.org`` and ``www.monkey.org`` completely), so that
    no fragment such as ``org`` survives.
    """
    if not terms:
        return text
    if mode not in ("delete", "token"):
        raise ValueError("mode must be 'delete' or 'token'")
    for term in terms:
        if not isinstance(term, str) or not term.strip():
            continue
        core = _term_pattern(term)
        if mode == "token":
            text = re.sub(r"\b" + core + r"\b", MASK_TOKEN, text, flags=re.I)
        elif re.fullmatch(r"[\w\-]+", term.strip()):
            pattern = rf"(?<![\w.])(?:[\w\-]+\.)*{core}(?:\.[\w\-]+)*(?!\w)"
            text = re.sub(pattern, " ", text, flags=re.I)
        else:
            text = re.sub(rf"(?<!\w){core}(?!\w)", " ", text, flags=re.I)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalise(
    text: str,
    mask_terms: Optional[List[str]] = None,
    mask_dates: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """Decode entities, lower-case, optionally clean, mask terms, collapse whitespace."""
    opts = _resolve_options(options)
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = html.unescape(text).lower()
    if opts["strip_urls"]:
        text = strip_urls(text)
    if opts["strip_email_addresses"]:
        text = strip_email_addresses(text)
    if mask_dates or opts["strip_temporal_cues"]:
        text = strip_temporal_cues(text)
    text = delete_terms(text, mask_terms, opts["mask_mode"])
    return re.sub(r"\s+", " ", text).strip()


def prepare_body(body: str, options: Optional[Dict[str, Any]] = None) -> str:
    """Body-only clean-up applied BEFORE normalisation."""
    opts = _resolve_options(options)
    if not isinstance(body, str):
        body = "" if body is None else str(body)
    if opts["strip_header_remnants"]:
        body, _ = strip_header_remnants(body)
    if opts["decode_html_in_csv_sources"]:
        body = decode_html_and_qp(body)
    return body


def build_model_text(
    subject: str,
    body: str,
    mask_terms: Optional[List[str]] = None,
    mask_dates: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """Build ``"subject: <subject> body: <body>"``.

    ``options=None`` reproduces the pre-revision behaviour (plus ``mask_dates``).
    The same call, with the same ``mask_terms``, ``mask_dates`` and ``options``,
    must be used for training, experiments and prediction.
    """
    body = prepare_body(body, options)
    norm_sub = normalise(subject, mask_terms, mask_dates, options)
    norm_body = normalise(body, mask_terms, mask_dates, options)
    return f"subject: {norm_sub} body: {norm_body}"


def text_settings_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Everything needed to rebuild model text identically later (stored in the bundle)."""
    section = cfg["text"]
    if section.get("mask_terms") is None or section.get("mask_dates") is None:
        raise ValueError("text.mask_terms and text.mask_dates must be set")
    return {
        "mask_terms": list(section["mask_terms"]),
        "mask_dates": bool(section["mask_dates"]),
        "text_options": get_text_options(cfg),
    }
