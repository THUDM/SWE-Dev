#!/bin/bash

prefix=$1
images=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "^$prefix")

if [ -z "$images" ]; then
    echo "No such images"
    exit 0
fi

echo "Find images below:"
echo "$images"

for image in $images; do
    new_image=$(echo "$image" | sed "s|^${prefix}||")
    new_image="${new_image//_s_/__}"
    echo "Retagging $image -> $new_image"

    docker tag "$image" "$new_image"

    echo "Deleting original image: $image"
    docker rmi "$image"
done

echo "Retag finished"