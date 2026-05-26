"""Hooks for the documentation."""

from __future__ import annotations

import pathlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

from mkdocs.structure.files import File, Files

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.pages import Page

# ---------------------------------------------------------------------------
# Monkeypatch mkdocs-jupyter's ``should_include`` so that ``.md`` files
# that are *not* in the ``include`` glob list are rejected **before**
# ``jupytext.read()`` is called.  Without this, jupytext invokes Pandoc
# (``--from markdown --to ipynb``) for every ``.md`` file it encounters
# and Pandoc emits dozens of "unclosed Div" warnings for files that
# contain HTML or mkdocstrings ``:::`` directives.
# ---------------------------------------------------------------------------
try:
    from mkdocs_jupyter.plugin import Plugin as _JupyterPlugin

    _orig_should_include = _JupyterPlugin.should_include

    def _patched_should_include(self, file):  # type: ignore[override]
        ext = pathlib.PurePath(file.abs_src_path).suffix
        if ext == ".md":
            srcpath = pathlib.PurePath(file.abs_src_path)
            if not any(srcpath.match(p) for p in self.config["include"]):
                return False
        return _orig_should_include(self, file)

    _JupyterPlugin.should_include = _patched_should_include
except ImportError:
    pass

_ROOT = Path(__file__).parent.parent

readme = _ROOT / "README.md"
changelog = _ROOT / "CHANGELOG.md"
contributing = _ROOT / "CONTRIBUTING.md"
design = _ROOT / "DESIGN.md"
license = _ROOT / "LICENSE"


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Add root-level markdown files to the documentation site.

    The README.md is injected as ``index.md`` so that the docs
    landing page and the GitHub README stay in sync automatically.
    """
    # Remove docs/index.md if it exists so README.md takes its place.
    files = Files([f for f in files if f.src_path != "index.md"])

    # Inject README.md as the docs index page.
    idx = File(
        path="index.md",
        src_dir=str(readme.parent),
        dest_dir=str(config.site_dir),
        use_directory_urls=config.use_directory_urls,
    )
    idx.abs_src_path = str(readme)
    files.append(idx)

    for path in (changelog, contributing, design):
        files.append(
            File(
                path=path.name,
                src_dir=str(path.parent),
                dest_dir=str(config.site_dir),
                use_directory_urls=config.use_directory_urls,
            )
        )
    lic = File(
        path="LICENSE.md",
        src_dir=str(license.parent),
        dest_dir=str(config.site_dir),
        use_directory_urls=config.use_directory_urls,
    )
    lic.abs_src_path = str(license)
    files.append(lic)
    return files


# Regex matching src="docs/..." in HTML img tags and ![...](docs/...) in markdown.
_DOCS_PREFIX = re.compile(r'((?:src="|]\())(docs/)')


def on_page_markdown(markdown: str, page: Page, **_kwargs: object) -> str:
    """Rewrite ``docs/`` image paths in README so they resolve in mkdocs.

    On GitHub, paths like ``docs/examples/images/foo.png`` are relative to
    the repo root.  When the README is served as ``index.md`` inside the
    docs directory, the ``docs/`` prefix must be stripped.
    """
    if page.file.abs_src_path == str(readme):
        markdown = _DOCS_PREFIX.sub(r"\1", markdown)
    return markdown


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHFABCDEFGHJKSTsu]")
_ABS_PREFIX = str(_ROOT) + "/"


def on_page_content(html: str, page: Page, **_kwargs: object) -> str:
    """Post-process rendered HTML for notebook pages.

    1. Fix relative image paths for directory URLs (``../images/`` →
       ``../../images/``).
    2. Strip ANSI escape codes that leak into ``<pre>`` output blocks
       when notebooks are executed in a terminal-aware environment.
    3. Remove absolute repo paths so build artefacts are portable.
    """
    if page.file.src_path.startswith("examples/notebooks/"):
        html = html.replace('src="../images/', 'src="../../images/')
        html = _ANSI_RE.sub("", html)
        html = html.replace(_ABS_PREFIX, "")
    return html
