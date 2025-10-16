#!/usr/bin/env bash
# writes a minimal PTX file so include_str! finds something; returns success
# Finds the output path after '-o'
OUT=""
next_is_out=0
for arg in "$@"; do
  if [ "$next_is_out" = "1" ]; then
    OUT="$arg"
    break
  fi
  if [ "$arg" = "-o" ]; then
    next_is_out=1
    continue
  fi
  if [[ "$arg" == -o* ]]; then
    OUT="${arg:2}"
    break
  fi
done
if [ -z "$OUT" ]; then
  echo "nvcc_placeholder: no -o path provided" 1>&2
  exit 1
fi
mkdir -p "$(dirname "$OUT")" || true
cat > "$OUT" << 'PTX'
.version 7.0
.target sm_80
.address_size 64
// minimal placeholder PTX
PTX
exit 0
