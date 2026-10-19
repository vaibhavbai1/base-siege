"""Final training metrics per learner, sampled acting, at the full budget."""
import json, glob
from pathlib import Path

rows = []
for p in sorted(glob.glob("runs/study-v3/T-*/development.json")):
    run = Path(p).parent.name
    recs = json.loads(Path(p).read_text()).get("records", [])
    s = [r for r in recs if r.get("mode") == "sampled"]
    if not s:
        continue
    last = max(s, key=lambda r: r["timesteps"])
    rows.append((run, last))

if not rows:
    print("No sampled development records found.")
else:
    print(f"{'run':<6} {'transitions':>12} {'destruction':>12} {'full clear':>11} {'town hall':>10} {'return':>8}")
    for run, r in rows:
        print(f"{run:<6} {r['timesteps']:>12,} {r['mean_destruction']*100:>11.1f}% "
              f"{r.get('full_clear_rate',0)*100:>10.1f}% {r.get('town_hall_rate',0)*100:>9.1f}% "
              f"{r.get('mean_return',0):>8.2f}")
    n = len(rows)
    print(f"\nmean across {n} learners: "
          f"destruction {sum(r['mean_destruction'] for _, r in rows)/n*100:.1f}%, "
          f"full clear {sum(r.get('full_clear_rate',0) for _, r in rows)/n*100:.1f}%, "
          f"town hall {sum(r.get('town_hall_rate',0) for _, r in rows)/n*100:.1f}%")
