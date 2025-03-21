#!/bin/bash

input_file="all-swebench-verified-instance-images.txt"

if [ ! -f "$input_file" ]; then
    echo "File $input_file not exists"
    exit 1
fi

count=0

local_images=$(docker images --format "{{.Repository}}:{{.Tag}}")

while IFS= read -r image_name || [ -n "$image_name" ]; do
    image_name="${image_name//_s_/__}"
    
    if echo "$local_images" | grep -q "^${image_name}$"; then
        count=$((count+1))
        echo $count $image_name
    else
        echo "No such image: $image_name"
    fi
done < "$input_file"