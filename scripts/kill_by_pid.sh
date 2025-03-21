#!/bin/bash

# Check if PID range parameter is passed
if [ $# -ne 1 ]; then
    echo "Usage: $0 a-b"
    exit 1
fi

# Extract range
range=$1
start_pid=$(echo$range | cut -d'-' -f1)
end_pid=$(echo$range | cut -d'-' -f2)

# Validate range
if ! [[ $start_pid =~ ^[0-9]+$ && $end_pid =~ ^[0-9]+$ ]]; then
    echo "Error: Please provide a valid PID range (e.g., 100-105)"
    exit 1
fi

if [ $start_pid -gt$end_pid ]; then
    echo "Error: Start PID should be less than or equal to end PID"
    exit 1
fi

# Iterate over range and attempt to kill each process
for pid in $(seq$start_pid $end_pid); do
    if kill -0 $pid 2>/dev/null; then
        kill -9 $pid
        echo "Terminated process PID: $pid"
    else
        echo "Skipped process PID: $pid (does not exist or no permission)"
    fi
done

echo "Done"
