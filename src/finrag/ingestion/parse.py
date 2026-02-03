from __future__ import annotations

import re
from copy import deepcopy
from typing import List, Tuple

from bs4 import BeautifulSoup
from langchain_core.documents import Document

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

# "garbage" if a line has very low alphabetic content
_ALPHA_RATIO_MIN = 0.25

# drop lines that are basically symbol soup
_MANY_SYMBOLS_RE = re.compile(r"[^A-Za-z0-9\s]{12,}")

# common SEC "Item" headings (10-K)
_ITEM_RE = re.compile(
    r"(?im)^\s*item\s+"
    r"(1a|1b|1c|1|2|3|4|5|6|7a|7|8|9a|9b|9|10|11|12|13|14|15)\s*"
    r"(\.|:|\-|\s)\s*"
)


def _alpha_ratio(s: str) -> float:
    if not s:
        return 0.0
    alpha = sum(ch.isalpha() for ch in s)
    return alpha / max(len(s), 1)


def _looks_like_xbrl_fact_line(s: str) -> bool:
    s_low = s.lower()
    # lots of schema URLs / namespaces / fact identifiers
    if "fasb.org/us-gaap" in s_low:
        return True
    if "http://www.xbrl.org" in s_low:
        return True
    if s_low.startswith(("xmlns", "xbrl", "ix:", "schema", "link:", "xlink:")):
        return True
    return False


def _clean_lines(text: str) -> str:
    out: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue

        # drop URL-heavy lines
        if _URL_RE.search(s) and len(s) > 120:
            continue

        # drop XBRL namespace / fact lines
        if _looks_like_xbrl_fact_line(s):
            continue

        # drop long digit-heavy lines (IDs, units, etc.)
        if len(s) > 250:
            digit_ratio = sum(ch.isdigit() for ch in s) / max(len(s), 1)
            if digit_ratio > 0.25:
                continue

        # drop symbol soup
        if _MANY_SYMBOLS_RE.search(s):
            continue

        # drop lines with too little alphabetic content
        if len(s) > 40 and _alpha_ratio(s) < _ALPHA_RATIO_MIN:
            continue

        out.append(s)

    # de-dup repeated lines (SEC filings can repeat headers)
    dedup: List[str] = []
    seen = set()
    for s in out:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(s)

    return "\n".join(dedup)


def _extract_human_text_from_sec_html(html: str) -> str:
    """
    Extract readable text from SEC iXBRL HTML (often XHTML/XML-ish),
    while aggressively removing XBRL junk + hidden elements.
    """
    import warnings

    # Prefer XML parser (SEC iXBRL is frequently XHTML-ish)
    # Fallback to HTML parser if XML parsing fails.
    try:
        soup = BeautifulSoup(html, "lxml-xml")
    except Exception:
        # This can raise XMLParsedAsHTMLWarning sometimes; not fatal.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            soup = BeautifulSoup(html, "lxml")

    # Remove obvious non-content
    for t in soup(["script", "style", "noscript", "svg"]):
        try:
            t.decompose()
        except Exception:
            pass

    # Drop tables (iXBRL fact tables spam tokens)
    for t in soup.find_all("table"):
        try:
            t.decompose()
        except Exception:
            pass

    # IMPORTANT: iterate on a snapshot list so decomposing doesn't break iteration
    all_tags = list(soup.find_all(True))
    for tag in all_tags:
        if tag is None:
            continue
        # After mutations, bs4 can leave tags with attrs=None
        if getattr(tag, "attrs", None) is None:
            continue

        # Safe style extraction (attrs.get, not tag.get)
        style = tag.attrs.get("style", "")
        if isinstance(style, list):
            style = " ".join(style)
        style = str(style).lower()

        if "display:none" in style or "visibility:hidden" in style:
            try:
                tag.decompose()
            except Exception:
                pass
            continue

        name = (getattr(tag, "name", "") or "").lower()

        # Unwrap inline-XBRL tags but keep visible text if any
        if name.startswith("ix:"):
            try:
                tag.unwrap()
            except Exception:
                pass

    body = soup.find("body") or soup
    text = body.get_text("\n", strip=True)

    text = _clean_lines(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _split_by_items(text: str) -> List[Tuple[str, str]]:
    """
    Split SEC 10-K-like text into sections keyed by Item headings.
    Returns list of (section_name, section_text).
    """
    matches = list(_ITEM_RE.finditer(text))
    if not matches:
        return [("Filing", text)]

    sections: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        item = m.group(1).upper()  # e.g., 1A, 7
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()

        # skip tiny chunks
        if len(chunk) < 400:
            continue

        sections.append((f"Item {item}", chunk))

    return sections or [("Filing", text)]


def parse_and_clean(docs: List[Document]) -> List[Document]:
    cleaned: List[Document] = []

    for d in docs:
        fn = (d.metadata.get("file_name") or "").lower()
        text = d.page_content or ""

        # SEC iXBRL HTML
        if fn.endswith((".htm", ".html")):
            extracted = _extract_human_text_from_sec_html(text)

            # Split into Item sections and attach metadata["section"]
            for section_name, section_text in _split_by_items(extracted):
                meta = deepcopy(d.metadata) if d.metadata else {}
                meta["section"] = section_name
                # HTML isn't paginated; keep page=1 if present
                meta.setdefault("page", 1)

                cleaned.append(
                    Document(page_content=section_text, metadata=meta))
            continue

        # PDFs or plain text (minimal normalization here)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        d.page_content = text
        cleaned.append(d)

    return cleaned
