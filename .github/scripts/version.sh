#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

git show HEAD^1:ladim/__init__.py
git show HEAD^2:ladim/__init__.py

# Step 1: Extract the file firstfile.txt from branch master and store it as comparefile.txt
git show HEAD^1:ladim/__init__.py > init.py.compare

# Step 2: Extract the first line starting with __version__ in firstfile.txt and comparefile.txt
first_version_line=$(grep '^__version__' ladim/__init__.py | head -n 1)
compare_version_line=$(grep '^__version__' init.py.compare | head -n 1)

echo "Old: $compare_version_line"
echo "New: $first_version_line"

# Fail if they are equal
if [ "$first_version_line" == "$compare_version_line" ]; then
    echo "Version number not updated"
    exit 1
fi
