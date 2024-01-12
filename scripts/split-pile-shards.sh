#!/bin/bash

# Assuming the input file is provided as the first argument
input_file="$1"
base_file=$(basename "$input_file" .jsonl.zst)

declare -A fds

# Function to set up a file descriptor for zstd for a given category
setup_fd() {
    local category=$1
    local output_file="${category}/${base_file}.jsonl.zst"

    # Create the directory if it doesn't exist
    mkdir -p "$category"

    # Set up a file descriptor for zstd process for this category
    exec {fd}> >(zstd -z > "$output_file")
    fds[$category]=$fd
}

cleanup() {
for fd in "${fds[@]}"; do
    exec {fd}>&-
done
}

trap cleanup EXIT

C=0

# Decompress the input file and process line by line
while IFS= read -r line; do
    # Print a progress indicator
    if [ $((C++ % 1000)) -eq 0 ]; then
        echo -ne "\r$C"
    fi
    # Extract the category value
    category=$(echo "$line" | jq -r '.meta.pile_set_name' | tr [:upper:] [:lower:] | tr - _ | tr -Cd [a-z0-9_])

    # Check if we already have a pipe for this category, if not, set it up
    if [ -z "${fds[$category]}" ]; then
        setup_fd "$category"
    fi

    # Write to the appropriate pipe
    eval "echo \"\$line\" >&${fds[$category]}"
done < <(zstdcat "$input_file")


cleanup