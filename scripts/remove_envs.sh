#!/bin/bash

set -e
set -u

TARGET_DIR="/raid/haoran/miniforge3/envs"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

for folder in "$TARGET_DIR"/swedev_*; do
    if [ -d "$folder" ]; then
        echo "Deleting folder: $folder"
        rm -rf "$folder" &
    fi
done

echo "All 'swedev_' subdirectories in '$TARGET_DIR' have been removed."
