#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Extract the file __init__.py from target branch and store it as comparefile.txt
git show HEAD^1:ladim/__init__.py > init.py.compare

# Step 2: Extract the first line starting with __version__ in __init__.py and comparefile.txt
first_version_line=$(grep '^__version__' ladim/__init__.py | head -n 1)
compare_version_line=$(grep '^__version__' init.py.compare | head -n 1)

# Extract version numbers from the lines
first_version_tag=$(echo "$first_version_line" | sed -E "s/__version__ *= *['\"]([^'\"]+)['\"].*/\1/")
compare_version_tag=$(echo "$compare_version_line" | sed -E "s/__version__ *= *['\"]([^'\"]+)['\"].*/\1/")

echo "Old: $compare_version_line"
echo "New: $first_version_line"

# Fail if they are equal
if [ "$first_version_line" == "$compare_version_line" ]; then
    echo "Version number not updated"
    exit 1
fi

# Compare major, minor, patch
IFS='.' read -r cmp_major cmp_minor cmp_patch <<< "$compare_version_tag"
IFS='.' read -r new_major new_minor new_patch <<< "$first_version_tag"

if [ "$cmp_major" -gt "$new_major" ]; then
    echo "Old version major ($cmp_major) is greater than new version major ($new_major)"
    exit 1
elif [ "$cmp_major" -eq "$new_major" ] && [ "$cmp_minor" -gt "$new_minor" ]; then
    echo "Old version minor ($cmp_minor) is greater than new version minor ($new_minor)"
    exit 1
elif [ "$cmp_major" -eq "$new_major" ] && [ "$cmp_minor" -eq "$new_minor" ] && [ "$cmp_patch" -gt "$new_patch" ]; then
    echo "Old version patch ($cmp_patch) is greater than new version patch ($new_patch)"
    exit 1
fi
