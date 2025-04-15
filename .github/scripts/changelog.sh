#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Extract the first line starting with __version__
first_version_line=$(grep '^__version__' ladim/__init__.py | head -n 1)

# Step 2: Extract the substring within quotes from the __version__ line
version_value=$(echo "$first_version_line" | sed -n "s/^__version__ = '\(.*\)'/\1/p")

# Step 3: Check if the contents of the variable exist within the text file
grep "[$version_value]" CHANGELOG.md
if grep -q "[$version_value]" CHANGELOG.md; then
    echo "The entry '[$version_value]' exists in CHANGELOG.md"
else
    echo "The entry '[$version_value]' does not exist in CHANGELOG.md"
    exit 1
fi
