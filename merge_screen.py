#!/usr/bin/env python
"""Merge sharded screen outputs (screen_shard_*.json) into one ranked screen.json."""
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
shards = sorted(out.glob("screen_shard_*.json"))
if not shards:
    raise SystemExit(f"no screen shards in {out}")
merged = json.loads(shards[0].read_text())
results = []
for f in shards:
    results.extend(json.loads(f.read_text())["results"])
results.sort(key=lambda r: -r["score"])
merged["results"] = results
merged["merged_shards"] = len(shards)
(out / "screen.json").write_text(json.dumps(merged, indent=2))
with (out / "screened.fasta").open("w") as fh:
    for r in results:
        fh.write(f">{merged['proposal_set']}_{r['design']:03d}_iptm{r['iptm']:.3f}\n{r['sequence']}\n")
print(f"merged {len(shards)} shards -> {len(results)} candidates, best ipTM {results[0]['iptm']}")
