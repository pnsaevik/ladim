#!/bin/bash

# Step 1: Extract the first line starting with __version__
first_version_line=$(grep '^__version__' ladim/__init__.py | head -n 1)

# Step 2: Extract the substring within quotes from the __version__ line
version_value=$(echo "$first_version_line" | sed -n "s/^__version__ = '\(.*\)'/\1/p")

# Step 3: Check if the contents of the variable exist within the text file
if grep -q "$version_value" ladim/__init__.py; then
    echo "The version value '$version_value' exists in firstfile.txt."
else
    echo "The version value '$version_value' does not exist in firstfile.txt."
    exit 1
fi
