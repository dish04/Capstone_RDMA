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

echo "Ensuring directory structure and symlinks in: $INITFS_DIR"
chmod +x "$INITFS_DIR/init"
mkdir -p "$INITFS_DIR"/{proc,sys,dev,sys/kernel/config,dev/infiniband,dev/shm,etc/libibverbs.d,mnt/weights,tmp,root,sbin,bin,usr,lib,lib64}

# Ensure essential busybox applet symlinks exist in initfs/bin
for app in cat grep cut mkdir mknod chmod chown ping ls ps kill rm cp mv head tail sed awk find umount clear echo env date touch dmesg stat which test true false sync; do
    if [ ! -e "$INITFS_DIR/bin/$app" ]; then
        ln -sf busybox "$INITFS_DIR/bin/$app"
    fi
done

cd "$INITFS_DIR"
find . -print0 | cpio --null -ov --format=newc | gzip -9 > "$OUTPUT_IMG"
cp -f "$OUTPUT_IMG" "$OUTPUT_ALT"

echo "Successfully generated:"
echo "  - $OUTPUT_IMG ($(du -h "$OUTPUT_IMG" | cut -f1))"
echo "  - $OUTPUT_ALT"

