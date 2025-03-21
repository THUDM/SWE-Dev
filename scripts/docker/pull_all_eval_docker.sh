#!/bin/bash
set -e

LEVEL=$1
# three levels:
# - base, keyword "sweb.base"
# - env, keyword "sweb.env"
# - instance, keyword "sweb.eval"
SET=$2

if [ -z "$LEVEL" ]; then
    echo "Usage: $0 <cache_level> <set>"
    echo "cache_level: base, env, or instance"
    echo "set: lite, full"
    exit 1
fi

if [ -z "$SET" ]; then
    echo "Usage: $0 <cache_level> <set>"
    echo "cache_level: base, env, or instance"
    echo "set: lite, full, default is lite"
    SET="lite"
fi

NAMESPACE=${3:-swebench}

echo "Using namespace: $NAMESPACE"

if [ "$SET" == "verified" ]; then
    IMAGE_FILE="$(dirname "$0")/swebench-verified-instance-images.txt"
else
    IMAGE_FILE="$(dirname "$0")/swebench-full-instance-images.txt"
fi

# Define a pattern based on the level
case $LEVEL in
    base)
        PATTERN="sweb.base"
        ;;
    env)
        PATTERN="sweb.base\|sweb.env"
        ;;
    instance)
        PATTERN="sweb.base\|sweb.env\|sweb.eval"
        ;;
    *)
        echo "Invalid cache level: $LEVEL"
        echo "Valid levels are: base, env, instance"
        exit 1
        ;;
esac

echo "Pulling docker images for [$LEVEL] level"

echo "Pattern: $PATTERN"
echo "Image file: $IMAGE_FILE"

# Read each line from the file, filter by pattern, and pull the docker image
export NAMESPACE  # Ensure environment variable is accessible to subprocesses
grep "$PATTERN" "$IMAGE_FILE" | xargs -P 64 -I {} bash -c '
    image="{}"

    echo "Processing image: $image"

    # Check if image already exists
    image_name=$(echo "$image" | sed "s/_s_/__/g")
    if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^$image_name$"; then
        echo "Image $image_name already exists, skipping pull."
    else
        echo "Pulling $NAMESPACE/$image into $image"
        docker pull $NAMESPACE/$image
    fi
'