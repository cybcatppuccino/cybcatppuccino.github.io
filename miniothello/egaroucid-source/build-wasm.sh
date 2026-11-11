#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
: "${CLANG:=clang}"
"$CLANG" --target=wasm32 -O3 -DNDEBUG -nostdlib -fno-builtin \
  -Wl,--no-entry -Wl,--export-memory \
  -Wl,--initial-memory=50331648 -Wl,--max-memory=134217728 \
  -Wl,-z,stack-size=1048576 -Wl,--strip-all \
  -o assets/egaroucid-6x6.wasm egaroucid-source/egaroucid-6x6.c
printf 'Built %s\n' assets/egaroucid-6x6.wasm
