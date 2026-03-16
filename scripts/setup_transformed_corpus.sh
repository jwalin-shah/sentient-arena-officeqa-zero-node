#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFICEQA_DIR="$ROOT/external/officeqa"
TARGET_DIR="$OFFICEQA_DIR/treasury_bulletins_parsed/transformed"

mkdir -p "$ROOT/external"

if [ ! -d "$OFFICEQA_DIR/.git" ]; then
  git clone --filter=blob:none --sparse https://github.com/databricks/officeqa.git "$OFFICEQA_DIR"
fi

cd "$OFFICEQA_DIR"
git sparse-checkout init --no-cone
git sparse-checkout set \
  README.md \
  officeqa_pro.csv \
  officeqa_full.csv \
  reward.py \
  treasury_bulletins_parsed/unzip.py \
  treasury_bulletins_parsed/transformed/treasury_bulletins_transformed.zip
git checkout

mkdir -p "$TARGET_DIR"
ZIP_PATH="$OFFICEQA_DIR/treasury_bulletins_parsed/transformed/treasury_bulletins_transformed.zip"

if [ -f "$ZIP_PATH" ]; then
  cd "$OFFICEQA_DIR/treasury_bulletins_parsed/transformed"
  python3 - << 'PY'
import zipfile
from pathlib import Path
z = Path("treasury_bulletins_transformed.zip")
if not z.exists():
    raise SystemExit(1)
with zipfile.ZipFile(z, "r") as f:
    f.extractall(".")
print("Extracted transformed corpus")
PY
fi

echo "Corpus setup complete. Transformed files under:"
echo "$TARGET_DIR"
