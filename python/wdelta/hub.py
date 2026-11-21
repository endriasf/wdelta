"""Minimal Hugging Face Hub access: uses huggingface_hub when installed, plain HTTPS otherwise."""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import urllib.request
from pathlib import Path

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def _req(url: str) -> urllib.request.Request:
    r = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        r.add_header("Authorization", f"Bearer {tok}")
    return r


def _api(repo: str, revision: str | None = None) -> dict:
    url = f"{HF}/api/models/{repo}" + (f"/revision/{revision}" if revision else "")
    with urllib.request.urlopen(_req(url), timeout=120) as r:
        return json.loads(r.read())


def commit_of(repo: str) -> str:
    """Current commit hash of a repo's main branch."""
    return _api(repo)["sha"]


def download(repo: str, dest: Path, patterns: list[str] | None = None, revision: str | None = None) -> Path:
    """Download the files of `repo` (optionally filtered by glob patterns) into `dest`."""
    dest = Path(dest)
    marker = dest / ".wdelta-download-complete"
    if marker.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, revision=revision, local_dir=dest, allow_patterns=patterns,
                          token=os.environ.get("HF_TOKEN"))
    except ImportError:
        files = [s["rfilename"] for s in _api(repo, revision).get("siblings", [])]
        if patterns:
            files = [f for f in files if any(fnmatch.fnmatch(f, p) for p in patterns)]
        for f in files:
            out = dest / f
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            url = f"{HF}/{repo}/resolve/{revision or 'main'}/{f}"
            with urllib.request.urlopen(_req(url), timeout=300) as r, open(f"{out}.part", "wb") as w:
                shutil.copyfileobj(r, w, length=1 << 22)
            os.replace(f"{out}.part", out)
    marker.write_text(revision or "main")
    return dest
