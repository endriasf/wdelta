"""
wdelta — lossless delta compression for fine-tuned models.

A packed model is a folder holding one ``.wdz`` file per safetensors shard, every other
file of the model copied as-is (config, tokenizer, ``.bin`` files, ...) and a
``wdelta.json`` manifest naming the base model (Hugging Face repo id + exact commit) and
the SHA-256 of every file. Unpacking downloads that base if needed and restores every
file byte-for-byte. Hidden files and folders (names starting with ".") are not packed.

    import wdelta
    wdelta.pack("my-org/my-finetune", base="meta-llama/Llama-3.1-8B-Instruct", out="packed/")
    path = wdelta.from_pretrained("packed/")          # unpacks into a cache, returns the folder
    model = AutoModelForCausalLM.from_pretrained(path)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import hub

__version__ = "0.3.1"
FORMAT = "wdelta-packed-v1"
MANIFEST = "wdelta.json"
CACHE = Path(os.environ.get("WDELTA_CACHE", Path.home() / ".cache" / "wdelta"))
WEIGHT_EXT = ".safetensors"


# ------------------------------------------------------------------ binary

def _binary() -> str:
    """Locate the wdelta executable: $WDELTA_BIN, the copy bundled in the package, or PATH."""
    exe = "wdelta.exe" if os.name == "nt" else "wdelta"
    cands = [os.environ.get("WDELTA_BIN"), str(Path(__file__).parent / "bin" / exe), shutil.which("wdelta")]
    for c in cands:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError("wdelta executable not found; build it with `cargo build --release` and set WDELTA_BIN")


def _run(*args: str, env: dict | None = None) -> str:
    p = subprocess.run([_binary(), *args], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout).strip())
    return p.stdout


# ------------------------------------------------------------------ helpers

def _resolve(spec: str, kind: str, patterns=None, revision: str | None = None) -> tuple[Path, str | None, str | None]:
    """Return (local_dir, repo_id, commit) for a local folder or a Hugging Face repo id."""
    p = Path(spec)
    if p.is_dir():
        return p, None, None
    repo = spec[3:] if spec.startswith("hf:") else spec
    commit = revision or hub.commit_of(repo)
    local = hub.download(repo, CACHE / kind / repo.replace("/", "__") / commit, patterns=patterns, revision=commit)
    return local, repo, commit


def _model_files(src: Path) -> list[Path]:
    """Every file of a model folder, as paths relative to it. Hidden files and folders (names
    starting with ".", such as download caches and .gitattributes) are left out."""
    return sorted(f.relative_to(src) for f in src.rglob("*")
                  if f.is_file() and not any(part.startswith(".") for part in f.relative_to(src).parts))


def _base_dir_for(b_dir: Path, rel: Path) -> str:
    """The base folder (relative to b_dir) whose tensors a shard at `rel` is coded against:
    the matching subfolder of the base if it holds safetensors files, else the base's top level."""
    sub = rel.parent
    if sub != Path(".") and any((b_dir / sub).glob("*" + WEIGHT_EXT)):
        return sub.as_posix()
    return ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ API

def pack(finetune: str, base: str, out: str, base_id: str | None = None, base_revision: str | None = None) -> dict:
    """Compress a fine-tuned model against its base.

    finetune, base: a local folder or a Hugging Face repo id ("org/name").
    base_id: label recorded for the base when `base` is a local folder (normally its repo id),
             so that `unpack` can download it later.
    Returns the manifest.
    """
    ft_dir, ft_repo, ft_commit = _resolve(finetune, "src")
    b_dir, b_repo, b_commit = _resolve(base, "bases", patterns=["*.safetensors", "*.json"], revision=base_revision)
    label = base_id or b_repo or str(b_dir)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, WDELTA_BASE_ID=label)
    files = _model_files(ft_dir)
    shards, total_in, total_out = [], 0, 0
    for rel in files:
        if rel.suffix != WEIGHT_EXT or rel.name == MANIFEST:
            continue
        f = ft_dir / rel
        wdz = out_dir / rel.with_name(rel.name + ".wdz")
        wdz.parent.mkdir(parents=True, exist_ok=True)
        bsub = _base_dir_for(b_dir, rel)
        _run("encode", str(b_dir / bsub), str(f), str(wdz), env=env)
        shards.append({"file": rel.as_posix(), "wdz": wdz.relative_to(out_dir).as_posix(), "base_dir": bsub,
                       "bytes": f.stat().st_size, "wdz_bytes": wdz.stat().st_size, "sha256": _sha256(f)})
        total_in += f.stat().st_size
        total_out += wdz.stat().st_size
    if not shards:
        raise ValueError(f"no {WEIGHT_EXT} files in {ft_dir}")
    # every other file is copied as-is, with its hash, so unpack restores the complete folder
    coded = {s["file"] for s in shards}
    outputs = {s["wdz"] for s in shards}
    extras = []
    for rel in files:
        if rel.as_posix() in coded or rel.name == MANIFEST:
            continue
        if rel.as_posix() in outputs:
            raise ValueError(f"{rel}: name clashes with a compressed shard; rename it before packing")
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ft_dir / rel, out_dir / rel)
        extras.append({"file": rel.as_posix(), "bytes": (ft_dir / rel).stat().st_size, "sha256": _sha256(ft_dir / rel)})
    manifest = {
        "format": FORMAT, "wdelta_version": __version__,
        "base": {"repo": label, "revision": b_commit or base_revision},
        "source": {"repo": ft_repo, "revision": ft_commit},
        "shards": shards, "extra_files": [e["file"] for e in extras], "extras": extras,
        "bytes": total_in, "packed_bytes": total_out, "extra_bytes": sum(e["bytes"] for e in extras),
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


def info(packed: str) -> dict:
    """Read a packed model's manifest (local folder or Hugging Face repo id)."""
    d = Path(packed)
    if not d.is_dir():
        d = hub.download(packed[3:] if packed.startswith("hf:") else packed,
                         CACHE / "packed-meta" / packed.replace("/", "__"), patterns=[MANIFEST])
    return json.loads((d / MANIFEST).read_text())


def unpack(packed: str, out: str, base: str | None = None) -> Path:
    """Restore a packed model into `out`, byte-identical to the original.

    packed: a local folder or a Hugging Face repo id holding a packed model.
    base: optional local folder with the base model; if omitted, the base recorded in the
          manifest is downloaded (at the recorded commit) into the wdelta cache.
    """
    p_dir, _, _ = _resolve(packed, "packed") if not Path(packed).is_dir() else (Path(packed), None, None)
    m = json.loads((p_dir / MANIFEST).read_text())
    if m.get("format") != FORMAT:
        raise ValueError(f"{p_dir}: not a {FORMAT} folder")
    if base:
        b_dir = Path(base)
    else:
        repo, rev = m["base"]["repo"], m["base"]["revision"]
        if Path(repo).is_dir():
            b_dir = Path(repo)
        else:
            b_dir = hub.download(repo, CACHE / "bases" / repo.replace("/", "__") / (rev or "main"),
                                 patterns=["*.safetensors", "*.json"], revision=rev)
    out_dir = Path(out).resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".wdelta-", dir=out_dir.parent))  # same filesystem, so the final rename is atomic
    try:
        for s in m["shards"]:
            dst = tmp / s["file"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            _run("decode", str(b_dir / s.get("base_dir", "")), str(p_dir / s["wdz"]), str(dst))  # checks SHA-256 internally
        for e in m.get("extras") or [{"file": f} for f in m.get("extra_files", [])]:
            src, dst = p_dir / e["file"], tmp / e["file"]
            if not src.is_file():
                raise FileNotFoundError(f"{src}: listed in {MANIFEST} but missing from the packed model")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if "sha256" in e and _sha256(dst) != e["sha256"]:
                raise ValueError(f"{e['file']}: SHA-256 does not match the one recorded at pack time")
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(tmp, out_dir)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return out_dir


def from_pretrained(packed: str, base: str | None = None, cache_dir: str | None = None) -> str:
    """Return a local folder with the unpacked model, ready for `transformers`' from_pretrained.

    The unpacked copy is cached (keyed by the manifest), so later calls return immediately.
    """
    m = info(packed)
    key = hashlib.sha256(json.dumps(m["shards"], sort_keys=True).encode()).hexdigest()[:16]
    target = Path(cache_dir or CACHE / "unpacked") / key
    done = target / ".wdelta-complete"
    if not done.exists():
        unpack(packed, str(target), base=base)
        done.write_text(m["base"]["repo"] or "")
    return str(target)
