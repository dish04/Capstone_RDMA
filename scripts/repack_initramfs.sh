#!/bin/bash
# ==============================================================================
# Repack initramfs archive from initfs staging folder
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INITFS_DIR="$WORKSPACE_ROOT/initfs"
OUTPUT_IMG="$WORKSPACE_ROOT/capstone/initramfs.cpio.gz"
OUTPUT_ALT="$WORKSPACE_ROOT/capstone/rdma/initrd.img"

echo "Repacking initramfs from: $INITFS_DIR"
chmod +x "$INITFS_DIR/init"

cd "$INITFS_DIR"
find . -print0 | cpio --null -ov --format=newc | gzip -9 > "$OUTPUT_IMG"
cp "$OUTPUT_IMG" "$OUTPUT_ALT"

echo "Successfully generated:"
echo "  - $OUTPUT_IMG ($(du -h "$OUTPUT_IMG" | cut -f1))"
echo "  - $OUTPUT_ALT"
