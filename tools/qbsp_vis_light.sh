#!/usr/bin/env bash
set -euo pipefail
map="$1"
qbsp "$map"
bsp="${map%.map}.bsp"
vis "$bsp"
light "$bsp"
echo "Built: $bsp"