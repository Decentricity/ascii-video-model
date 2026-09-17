#!/usr/bin/env bash
# Generate diverse Phase B carousel rollouts + GIFs for neurascii.github.io
set -euo pipefail
cd /home/decentricity/neurascii
source .venv/bin/activate

CKPT=runs/videos_all/ckpt_best.pt
if [[ ! -f "$CKPT" ]]; then
  CKPT=runs/videos_all/ckpt_last.pt
fi
echo "Using checkpoint: $CKPT"

mkdir -p samples/rollouts
mkdir -p /home/decentricity/neurascii.github.io/assets/gifs

# Pick 8 distinct classes (one clip each) spanning diverse prefixes
mapfile -t SEEDS < <(python - <<'PY'
from pathlib import Path
root = Path("data/processed/videos_all")
by_class = {}
for p in sorted(root.glob("*.avm.npz")):
    name = p.name
    if "__" not in name:
        continue
    cls = name.split("__", 1)[0]
    if cls not in by_class:
        by_class[cls] = p
# Prefer diverse spread: first of each class sorted, take every Nth to diversify
classes = sorted(by_class.keys())
n = min(8, max(6, len(classes)))
n = min(n, len(classes))
if len(classes) <= 8:
    chosen = classes[:n]
else:
    # evenly sample across sorted class list
    step = len(classes) / 8
    chosen = [classes[int(i * step)] for i in range(8)]
for i, cls in enumerate(chosen, 1):
    print(f"{i}|{cls}|{by_class[cls]}")
PY
)

echo "Selected ${#SEEDS[@]} seeds:"
printf '%s\n' "${SEEDS[@]}"

CLASSES=()
for entry in "${SEEDS[@]}"; do
  IFS='|' read -r i cls path <<< "$entry"
  CLASSES+=("$cls")
  out="samples/rollouts/carousel_videos_all_$(printf '%02d' "$i")_${cls}.avm.npz"
  echo "=== Generate $i $cls ==="
  python -m neurascii.generate --checkpoint "$CKPT" --seed-avm "$path" --out "$out" --steps 100
  gif="/home/decentricity/neurascii.github.io/assets/gifs/videos_all_$(printf '%02d' "$i")_${cls}.gif"
  echo "=== Render $gif ==="
  python -m neurascii.render_gif "$out" -o "$gif" --fps 6 --max-frames 120 --max-width 720 --cell 8
done

echo "CLASSES_USED=${CLASSES[*]}"
echo "NUM_GIFS=${#CLASSES[@]}"
ls -la /home/decentricity/neurascii.github.io/assets/gifs/videos_all_*.gif
