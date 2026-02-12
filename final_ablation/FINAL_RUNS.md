# Final Runs (Ablation)

Folder: `/home/chan/projects/stance_detection/7_final_ablation`

## 1) Best Original Model (Entropy Gate)

```bash
./run_best_entropy.sh
```

Output:
- `outputs/final_best_entropy_wp055_tau12_seed44/metrics_full.json`

## 2) Baseline Control (Concat)

```bash
./run_baseline_concat.sh
```

Output:
- `outputs/final_baseline_concat_wp055_seed44/metrics_full.json`

## 3) Compare quickly

```bash
python - <<'PY'
import json
from pathlib import Path
base = Path('outputs/final_baseline_concat_wp055_seed44/metrics_full.json')
best = Path('outputs/final_best_entropy_wp055_tau12_seed44/metrics_full.json')
for name, p in [('baseline_concat', base), ('best_entropy', best)]:
    d = json.loads(p.read_text())
    print(name, {k: round(d[k],4) for k in ['macro_f1','weighted_f2','f2_support','f2_deny','accuracy']})
PY
```
