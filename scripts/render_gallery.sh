#!/bin/bash
# Render every pattern-gallery tile from holo_foil.blend (sequential, CPU by default), then compose
# docs/images/pattern_gallery.jpg and the asset previews in build/previews/.        scripts/render_gallery.sh [spp] [W H] [CPU|GPU]
set -e
cd "$(dirname "$0")/.."
SPP=${1:-64}; W=${2:-384}; H=${3:-512}; DEV=${4:-CPU}
BLENDER=${BLENDER:-blender}
TILES="linear sheen radial scratched glitter cosmos cracked stars beams dotmatrix image card_holo card_reverse card_layer"
mkdir -p build/gallery
for t in $TILES; do
  "$BLENDER" -b holo_foil.blend -t 6 --python scripts/render_gallery.py -- --tile $t --mode map \
      --res $W $H --out build/gallery/map_$t 2>&1 | grep -E "^TILE|Error|Traceback" | cut -c1-160
  "$BLENDER" -b holo_foil.blend -t 6 --python scripts/render_gallery.py -- --tile $t --mode render \
      --res $W $H --spp $SPP --device $DEV --out build/gallery/render_$t 2>&1 | grep -E "^TILE|Error|Traceback" | cut -c1-160
done
python3 scripts/compose_gallery.py --tiledir build/gallery --width 320
echo "re-run scripts/build_holo_foil.py to embed the asset previews"
