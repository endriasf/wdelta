"""End-to-end check: a model restored by wdelta.from_pretrained behaves identically in transformers.

usage: python test_transformers_e2e.py PACKED_DIR ORIGINAL_DIR
"""
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import wdelta

packed, original = sys.argv[1], sys.argv[2]
t0 = time.time()
path = wdelta.from_pretrained(packed)
print(f"from_pretrained -> {path} ({time.time() - t0:.1f}s)")

kw = dict(dtype=torch.bfloat16)
a = AutoModelForCausalLM.from_pretrained(original, **kw).eval()
b = AutoModelForCausalLM.from_pretrained(path, **kw).eval()

sa, sb = a.state_dict(), b.state_dict()
assert sa.keys() == sb.keys()
same = sum(torch.equal(sa[k], sb[k]) for k in sa)
print(f"state_dict: {same}/{len(sa)} tensors bit-identical")
assert same == len(sa)

tok = AutoTokenizer.from_pretrained(path)
prompt = "The National Archives preserves records because"
ids = tok(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    la, lb = a(ids).logits, b(ids).logits
    ga = a.generate(ids, max_new_tokens=20, do_sample=False)
    gb = b.generate(ids, max_new_tokens=20, do_sample=False)
print(f"logits identical: {torch.equal(la, lb)}   greedy generation identical: {torch.equal(ga, gb)}")
print("restored model says:", repr(tok.decode(gb[0][ids.shape[1]:], skip_special_tokens=True)))
assert torch.equal(la, lb) and torch.equal(ga, gb)
print("PASS")
