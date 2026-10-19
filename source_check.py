"""Compare the TRPO source each run was trained with."""
import json, glob, collections
from pathlib import Path

seen = collections.defaultdict(list)
for p in sorted(glob.glob("runs/study-v3/T-*/contract.json")):
    try:
        h = json.loads(Path(p).read_text()).get("source_hashes", {})
    except (OSError, ValueError):
        continue
    key = tuple(sorted(h.items()))
    seen[key].append(Path(p).parent.name)

if not seen:
    print("no run contracts found")
else:
    for i, (key, runs) in enumerate(seen.items(), 1):
        print(f"source set {i}: {', '.join(runs)}")
        for name, digest in key:
            print(f"    {name:<18} {digest[:12]}")
    algo = {dict(k).get("maskable_trpo.py") for k in seen}
    print()
    if len(algo) == 1:
        print("ALGORITHM IDENTICAL across all runs (maskable_trpo.py unchanged)")
    else:
        print("WARNING: maskable_trpo.py differs between runs — the algorithm changed mid-study")
