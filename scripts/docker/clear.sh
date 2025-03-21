#!/bin/bash

images_to_delete=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "^${prefix}")

if [ -z "$images_to_delete" ]; then
    echo "No images found with prefix: $prefix"
    exit 0
fi

echo "Deleting the following images:"
echo "$images_to_delete"

while IFS= read -r image; do
    docker rmi "$image"
done <<< "$images_to_delete"