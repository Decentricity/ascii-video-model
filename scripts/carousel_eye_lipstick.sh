#!/usr/bin/env bash
# Generate 10-GIF carousels for ApplyEyeMakeup-only and ApplyLipstick-only POCs
set -euo pipefail
cd /home/decentricity/neurascii
source .venv/bin/activate

PAGES=/home/decentricity/neurascii.github.io
mkdir -p samples/rollouts
mkdir -p "$PAGES/assets/gifs"

pick_ckpt() {
  local name="$1"
  shift
  local c
  for c in "$@"; do
    if [[ -f "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  echo "MISSING_CKPT:$name" >&2
  return 1
}

EYE_CKPT=$(pick_ckpt eye \
  runs/poc_eye_best/ckpt_best.pt \
  runs/poc_eye/ckpt_best.pt \
  runs/poc_eye_best/ckpt_last.pt \
  runs/poc_eye/ckpt_last.pt) || { echo "Eye carousel stopped: no checkpoint"; EYE_CKPT=""; }

LIP_CKPT=$(pick_ckpt lipstick \
  runs/poc_lipstick/ckpt_best.pt \
  runs/poc_lipstick/ckpt_last.pt) || { echo "Lipstick carousel stopped: no checkpoint"; LIP_CKPT=""; }

echo "EYE_CKPT=${EYE_CKPT:-NONE}"
echo "LIP_CKPT=${LIP_CKPT:-NONE}"

# Evenly sample up to 10 distinct clips; if fewer than 10, pad with start offsets
select_seeds() {
  local processed_dir="$1"
  python - "$processed_dir" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = root / "manifest.json"
entries = []
if manifest.is_file():
    data = json.loads(manifest.read_text())
    for e in data:
        p = Path(e["avm"])
        if not p.is_file():
            p = root / Path(e["avm"]).name
        if p.is_file():
            entries.append((e.get("stem") or p.stem, p))
else:
    for p in sorted(root.glob("*.avm.npz")):
        entries.append((p.stem, p))
if not entries:
    raise SystemExit(f"no seeds in {root}")
n = len(entries)
want = 10
chosen = []
if n >= want:
    step = n / want
    idxs = [int(i * step) for i in range(want)]
    # unique-ify if collisions
    seen = set()
    for i in idxs:
        while i in seen:
            i = (i + 1) % n
        seen.add(i)
        chosen.append((entries[i][0], entries[i][1], 0))
else:
    for stem, path in entries:
        chosen.append((stem, path, 0))
    starts = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    i = 0
    while len(chosen) < want:
        stem, path, _ = entries[i % n]
        start = starts[(len(chosen) - n) % len(starts)]
        chosen.append((f"{stem}@start{start}", path, start))
        i += 1
for i, (stem, path, start) in enumerate(chosen, 1):
    print(f"{i}|{stem}|{path}|{start}")
PY
}

run_carousel() {
  local label="$1"
  local ckpt="$2"
  local processed="$3"
  local out_prefix="$4"
  local gif_prefix="$5"
  local -a seeds
  mapfile -t seeds < <(select_seeds "$processed")
  echo "=== $label: ${#seeds[@]} seeds, ckpt=$ckpt ==="
  local entry i stem path start out gif
  local count=0
  for entry in "${seeds[@]}"; do
    IFS='|' read -r i stem path start <<< "$entry"
    out="samples/rollouts/${out_prefix}_$(printf '%02d' "$i").avm.npz"
    gif="$PAGES/assets/gifs/${gif_prefix}_$(printf '%02d' "$i").gif"
    echo "--- $label $i $stem start=$start ---"
    python -m neurascii.generate \
      --checkpoint "$ckpt" \
      --seed-avm "$path" \
      --out "$out" \
      --steps 100 \
      --start "$start"
    python -m neurascii.render_gif "$out" -o "$gif" \
      --fps 6 --max-frames 120 --max-width 720 --cell 8
    count=$((count + 1))
  done
  echo "${label}_GIFS=$count"
  # write captions sidecar for index.html authoring
  local caps="$PAGES/assets/gifs/${gif_prefix}_captions.txt"
  : > "$caps"
  for entry in "${seeds[@]}"; do
    IFS='|' read -r i stem path start <<< "$entry"
    echo "$(printf '%02d' "$i")|$stem" >> "$caps"
  done
}

if [[ -n "$EYE_CKPT" ]]; then
  run_carousel eye "$EYE_CKPT" data/processed/poc_eye carousel_eye eye
else
  echo "Skipping eye carousel"
fi

if [[ -n "$LIP_CKPT" ]]; then
  run_carousel lipstick "$LIP_CKPT" data/processed/poc_lipstick carousel_lipstick lipstick
else
  echo "Skipping lipstick carousel"
fi

echo DONE
ls -la "$PAGES/assets/gifs"/eye_*.gif "$PAGES/assets/gifs"/lipstick_*.gif 2>&1 | head -40
