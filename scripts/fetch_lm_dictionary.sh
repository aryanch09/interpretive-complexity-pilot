#!/usr/bin/env bash
# Fetch the Loughran-McDonald Master Dictionary.
#
# Free for use in academic research (sraf.nd.edu). Redistribution is not
# explicitly granted, so the file is gitignored and fetched rather than vendored.
# Cite Loughran and McDonald (2011) when using it.
set -euo pipefail

FID=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/dictionaries"
DEST="$DEST_DIR/Loughran-McDonald_MasterDictionary_1993-2025.csv"

mkdir -p "$DEST_DIR"
if [ -f "$DEST" ]; then
  echo "Already present: $DEST"; exit 0
fi

echo "Fetching Master Dictionary ..."
curl -sL "https://drive.google.com/uc?export=download&id=$FID" -o "$DEST"

if ! head -1 "$DEST" | grep -q "^Word,"; then
  echo "ERROR: downloaded file does not look like the dictionary." >&2
  echo "Google Drive may be serving a confirmation page. Download manually from" >&2
  echo "  https://sraf.nd.edu/loughranmcdonald-master-dictionary/" >&2
  echo "and save it to $DEST" >&2
  rm -f "$DEST"; exit 1
fi
echo "Saved $DEST ($(wc -l < "$DEST") rows)"
