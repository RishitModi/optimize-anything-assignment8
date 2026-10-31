#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
XV6_DIR="$(cd "$SCRIPT_DIR/../xv6-public" && pwd)"
PATCHES_DIR="$SCRIPT_DIR/../patches"

cd "$XV6_DIR"
git add -N .

mkdir -p "$PATCHES_DIR"
git diff > "$PATCHES_DIR/lottery_scheduler.diff"
git diff --stat

