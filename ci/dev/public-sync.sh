#!/bin/bash

# Define source and destination directories
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../blueprints" && pwd)"

# Rsync options
RSYNC_OPTIONS=(
    --archive
    --verbose
    --delete
    --exclude 'ci/'
    --exclude 'README-e2e.md'
    --exclude 'README-sync.md'
    --exclude '.github/*'
    --exclude 'Makefile'
    --exclude '.pre-commit-config.yaml'
    --exclude '.vscode/'
    --exclude 'pyproject.toml'
    --exclude '.git/'
)

# Perform the sync
rsync "${RSYNC_OPTIONS[@]}" "$SOURCE_DIR/" "$DEST_DIR/"
