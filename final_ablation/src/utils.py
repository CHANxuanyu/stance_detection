import re
from typing import Optional

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")


def normalize_text(text: Optional[str], replace_urls_mentions: bool = True) -> str:
    if text is None:
        return ""
    text = str(text)
    if replace_urls_mentions:
        text = URL_RE.sub("$URL$", text)
        text = MENTION_RE.sub("$MENTION$", text)
    return text
