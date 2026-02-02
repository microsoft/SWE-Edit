#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Cleanup script - removes swebench Docker containers and trajectories

set -e

# Find all containers using swebench images
SWEBENCH_CONTAINERS=$(docker ps -a --format "{{.ID}} {{.Image}}" | grep "swebench/" | awk '{print $1}' || true)

if [ -z "$SWEBENCH_CONTAINERS" ]; then
    echo "ℹ️  No swebench containers found."
else
    # Count containers
    CONTAINER_COUNT=$(echo "$SWEBENCH_CONTAINERS" | grep -c . || echo "0")

    echo "⚠️  WARNING: This will remove $CONTAINER_COUNT swebench Docker container(s)!"
    echo ""
    echo "Containers to be removed:"
    docker ps -a --format "{{.ID}} {{.Image}}" | grep "swebench/" || true
    echo ""
    read -p "Are you sure? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        echo "Cleanup cancelled."
        exit 0
    fi

    echo "🧹 Cleaning up swebench containers..."
    if ! docker rm -f $SWEBENCH_CONTAINERS 2>/dev/null; then
        echo "  ⚠️  Warning: Some containers may have failed to remove"
    fi
fi

echo "✅ Cleanup complete!"
