"""Command line: python -m wdelta pack|unpack|info ..."""
import argparse
import json
import sys

from . import info, pack, unpack


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wdelta", description="Lossless delta compression for fine-tuned models.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="compress a fine-tune against its base")
    p.add_argument("finetune", help="local folder or Hugging Face repo id")
    p.add_argument("--base", required=True, help="local folder or Hugging Face repo id of the base model")
    p.add_argument("--base-id", help="repo id to record when --base is a local folder")
    p.add_argument("-o", "--out", required=True, help="output folder")
    u = sub.add_parser("unpack", help="restore a packed model (downloads the base if needed)")
    u.add_argument("packed", help="local folder or Hugging Face repo id")
    u.add_argument("-o", "--out", required=True, help="output folder")
    u.add_argument("--base", help="local base model folder (otherwise downloaded)")
    i = sub.add_parser("info", help="show a packed model's manifest")
    i.add_argument("packed")
    a = ap.parse_args(argv)
    if a.cmd == "pack":
        m = pack(a.finetune, a.base, a.out, base_id=a.base_id)
        print(f"packed {len(m['shards'])} shards: {m['bytes'] / 1e9:.2f} GB -> {m['packed_bytes'] / 1e9:.2f} GB "
              f"({100 * m['packed_bytes'] / m['bytes']:.1f}%), base {m['base']['repo']}@{(m['base']['revision'] or '')[:10]}")
        extras = m.get("extras", [])
        if extras:
            print(f"copied {len(extras)} other files uncompressed: {m['extra_bytes'] / 1e9:.2f} GB")
            for e in sorted(extras, key=lambda e: -e["bytes"])[:5]:
                if e["bytes"] >= 100 << 20:
                    print(f"  {e['file']}: {e['bytes'] / 1e9:.2f} GB (not compressed; only safetensors weights are)")
    elif a.cmd == "unpack":
        print(f"unpacked to {unpack(a.packed, a.out, base=a.base)}")
    else:
        json.dump(info(a.packed), sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
