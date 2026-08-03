#!/usr/bin/env bash
# Export a .blend and render PNG previews for every MALIBU3D ROI config.
#
# Each ROI runs in its own process so memory is released between tiles, and so
# one failure does not take the batch down.
#
#   bash scripts/render_all_rois.sh [resolution_x] [resolution_y] [samples]
set -u

PY="${HOME}/miniconda3/envs/blendify/bin/python"
RESX="${1:-1200}"
RESY="${2:-800}"
SAMPLES="${3:-32}"

cd "$(dirname "$0")/.." || exit 1

configs=$(ls configs/figures/malibu3d_*.yaml | grep -v '_graphs.yaml')
total=$(echo "$configs" | wc -l)
index=0
failed=()

echo "Rendering $total ROIs at ${RESX}x${RESY}, $SAMPLES samples"
echo

for config in $configs; do
    index=$((index + 1))
    roi=$(basename "$config" .yaml)
    printf '[%2d/%2d] %s\n' "$index" "$total" "$roi"

    if ! "$PY" examples/00_custom.py --config "$config" --image --export \
            --set render.n_samples="$SAMPLES" \
                  "render.resolution=[$RESX,$RESY]" \
            > "/tmp/${roi}.log" 2>&1; then
        echo "        FAILED — see /tmp/${roi}.log"
        failed+=("$roi")
        continue
    fi
    grep -E "points,|layer opacity" "/tmp/${roi}.log" | head -2 | sed 's/^/        /'
    grep -c "^Saved:" "/tmp/${roi}.log" | sed 's/^/        images: /'
done

echo
if [ ${#failed[@]} -eq 0 ]; then
    echo "All $total ROIs done."
else
    echo "Failed: ${failed[*]}"
fi
