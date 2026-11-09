"""Label documentation versions and resolve source-relative notebook links."""

import logging
import os
import posixpath
import re
from html import escape, unescape
from urllib.parse import unquote, urlsplit, urlunsplit

from mkdocs.utils import get_relative_url

log = logging.getLogger("mkdocs.hooks.notebook_links")


def on_config(config):
    config.extra["docs_version"] = os.environ.get("MIKE_DOCS_VERSION", "dev")


def on_page_content(html, *, page, config, files):
    if not page.file.src_uri.endswith(".ipynb"):
        return html

    def resolve(match):
        href = unescape(match.group(1))
        url = urlsplit(href)
        if url.scheme or url.netloc or not url.path.endswith((".md", ".ipynb")):
            return match.group(0)
        source = posixpath.normpath(
            posixpath.join(posixpath.dirname(page.file.src_uri), unquote(url.path))
        )
        target = files.get_file_from_path(source)
        if target is None:
            log.warning("%s links to missing source %s", page.file.src_uri, href)
            return match.group(0)
        relative = get_relative_url(target.url, page.url)
        rewritten = urlunsplit(("", "", relative, url.query, url.fragment))
        return f'href="{escape(rewritten, quote=True)}"'

    return re.sub(r'href="([^"]*)"', resolve, html)
