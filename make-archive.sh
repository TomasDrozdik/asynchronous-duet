#!/bin/bash

# Create BUILD_DIR and of duet package, docker images and duet configurations.
# Output is archived BUILD_DIR used by `run-instance.sh`

set -xeuo pipefail

# Rebuild dist
python -m build

BUILD_DIR="duetbench"
mkdir -p ${BUILD_DIR}

# Add duet pacakge
cp ./dist/*.tar.gz ${BUILD_DIR}/duet.tar.gz

# Add duet configs
for file in $(find ./benchmarks -name "duet.yml") ; do
    mkdir -p ${BUILD_DIR}/$(dirname ${file})
    cp ${file} ${BUILD_DIR}/$(dirname ${file})
done

# Compress BUILD_DIR
tar -cvzf ${BUILD_DIR}.tar.gz ${BUILD_DIR}

mv ${BUILD_DIR}.tar.gz duetbench.$(git rev-parse --abbrev-ref HEAD).tar.gz
