"""Tests for the wdelta Python package. Offline by default; set WDELTA_NET_TESTS=1 to also
run the Hugging Face round trip (downloads about 3 GB).

    WDELTA_BIN=target/release/wdelta pytest python/tests -q
"""
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

import wdelta


def write_st(path, tensors):
    hdr, off, blobs = {"__metadata__": {"format": "pt"}}, 0, []
    for name, (dtype, arr) in tensors.items():
        b = arr.tobytes()
        hdr[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def bf16(x):
    return (np.asarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def models(tmp_path, monkeypatch):
    """A two-shard base and fine-tune with config/tokenizer files, plus an isolated cache."""
    monkeypatch.setenv("WDELTA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(wdelta, "CACHE", tmp_path / "cache")
    rng = np.random.default_rng(0)
    base, ft = tmp_path / "base", tmp_path / "ft"
    base.mkdir(); ft.mkdir()
    for shard in (1, 2):
        w = rng.standard_normal((256, 512)).astype(np.float32) * 0.02
        dw = rng.standard_normal(w.shape).astype(np.float32) * 1e-4 * (rng.random(w.shape) < 0.6)
        same = rng.standard_normal(1000).astype(np.float32)
        name = f"model-0000{shard}-of-00002.safetensors"
        write_st(base / name, {f"l{shard}.w": ("BF16", bf16(w)), f"l{shard}.same": ("BF16", bf16(same))})
        extra = {"head.w": ("BF16", bf16(rng.standard_normal(5000)))} if shard == 2 else {}
        write_st(ft / name, {f"l{shard}.w": ("BF16", bf16(w + dw)), f"l{shard}.same": ("BF16", bf16(same)),
                              f"l{shard}.ids": ("I64", np.arange(10, dtype=np.int64)), **extra})
    for d in (base, ft):
        (d / "config.json").write_text(json.dumps({"model_type": "test"}))
    (ft / "tokenizer.json").write_text('{"version": "1.0"}')
    (ft / "sub").mkdir()
    (ft / "sub" / "notes.txt").write_text("nested file")
    return base, ft, tmp_path


def test_pack_unpack_round_trip_is_byte_identical(models):
    base, ft, tmp = models
    m = wdelta.pack(str(ft), str(base), str(tmp / "packed"), base_id="test/base")
    assert m["base"]["repo"] == "test/base" and len(m["shards"]) == 2
    assert m["packed_bytes"] < m["bytes"]
    out = wdelta.unpack(str(tmp / "packed"), str(tmp / "restored"), base=str(base))
    for f in ft.rglob("*"):
        if f.is_file():
            assert sha(f) == sha(out / f.relative_to(ft)), f"{f.name} differs"


def test_manifest_records_shard_hashes(models):
    base, ft, tmp = models
    m = wdelta.pack(str(ft), str(base), str(tmp / "packed"))
    for s in m["shards"]:
        assert s["sha256"] == sha(ft / s["file"])
        assert (tmp / "packed" / s["wdz"]).exists()
    assert wdelta.info(str(tmp / "packed"))["shards"] == m["shards"]


def test_unpack_finds_local_base_recorded_at_pack_time(models):
    base, ft, tmp = models
    wdelta.pack(str(ft), str(base), str(tmp / "packed"))  # label = local base path
    out = wdelta.unpack(str(tmp / "packed"), str(tmp / "restored"))
    assert sha(out / "model-00001-of-00002.safetensors") == sha(ft / "model-00001-of-00002.safetensors")


def test_from_pretrained_caches_and_returns_a_loadable_folder(models):
    base, ft, tmp = models
    wdelta.pack(str(ft), str(base), str(tmp / "packed"))
    p1 = wdelta.from_pretrained(str(tmp / "packed"), base=str(base))
    assert (Path(p1) / "config.json").exists() and (Path(p1) / "model-00002-of-00002.safetensors").exists()
    mtime = (Path(p1) / "model-00001-of-00002.safetensors").stat().st_mtime
    p2 = wdelta.from_pretrained(str(tmp / "packed"), base=str(base))
    assert p1 == p2 and (Path(p2) / "model-00001-of-00002.safetensors").stat().st_mtime == mtime


def test_wrong_base_is_rejected_and_leaves_no_output(models):
    base, ft, tmp = models
    wdelta.pack(str(ft), str(base), str(tmp / "packed"))
    with pytest.raises(RuntimeError, match="base mismatch|missing from the supplied base"):
        wdelta.unpack(str(tmp / "packed"), str(tmp / "restored"), base=str(ft))
    assert not (tmp / "restored").exists()
    assert not any(p.name.startswith(".wdelta-") for p in tmp.iterdir())


def test_damaged_packed_file_is_rejected(models):
    base, ft, tmp = models
    m = wdelta.pack(str(ft), str(base), str(tmp / "packed"))
    wdz = tmp / "packed" / m["shards"][0]["wdz"]
    data = bytearray(wdz.read_bytes())
    data[len(data) // 2] ^= 0x10
    wdz.write_bytes(bytes(data))
    with pytest.raises(RuntimeError):
        wdelta.unpack(str(tmp / "packed"), str(tmp / "restored"), base=str(base))
    assert not (tmp / "restored").exists()


def test_command_line(models, capsys):
    from wdelta.__main__ import main
    base, ft, tmp = models
    main(["pack", str(ft), "--base", str(base), "--base-id", "test/base", "-o", str(tmp / "packed")])
    main(["unpack", str(tmp / "packed"), "-o", str(tmp / "restored"), "--base", str(base)])
    main(["info", str(tmp / "packed")])
    out = capsys.readouterr().out
    assert "packed 2 shards" in out and "unpacked to" in out and '"test/base"' in out
    assert sha(tmp / "restored" / "model-00002-of-00002.safetensors") == sha(ft / "model-00002-of-00002.safetensors")


@pytest.mark.skipif(not os.environ.get("WDELTA_NET_TESTS"), reason="set WDELTA_NET_TESTS=1 to run Hub tests")
def test_hub_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(wdelta, "CACHE", tmp_path / "cache")
    wdelta.pack("Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B-Base", str(tmp_path / "packed"))
    monkeypatch.setattr(wdelta, "CACHE", tmp_path / "cache2")  # fresh cache: base must be downloaded
    out = wdelta.unpack(str(tmp_path / "packed"), str(tmp_path / "restored"))
    m = wdelta.info(str(tmp_path / "packed"))
    assert sha(out / "model.safetensors") == m["shards"][0]["sha256"]
