#!/bin/bash

# Create archive of duet package, docker images and duet configurations.
# Output is fixed name archive `duetbench.tar.gz` used by `run-instance.sh`

set -xeuo pipefail

# Rebuild dist
python -m build

ARCHIVE="duetbench"
mkdir -p ${ARCHIVE}

# Add duet pacakge
cp ./dist/*.tar.gz ${ARCHIVE}/duet.tar.gz

# Add duet configs
for file in $(find ./benchmarks -name "duet.yml") ; do
    mkdir -p ${ARCHIVE}/$(dirname ${file})
    cp ${file} ${ARCHIVE}/$(dirname ${file})
done

# Add images
IMAGES="renaissance dacapo scalabench speccpu"
for image in ${IMAGES} ; do
    if [ ! -f ${image}.tar ] ; then
        docker image save ${image} > ${image}.tar
    fi

    cp ${image}.tar ${ARCHIVE}
done

# Compress archive
tar -cvzf duetbench.tar.gz ${ARCHIVE}
