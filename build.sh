#!/bin/bash

set -e

while getopts d: flag ; do
    case "${flag}" in
        d) docker=${OPTARG};;
    esac
done

docker=${docker:-docker}

# Build barrier-agent as a base image for synchronized duets
pushd ./benchmarks/barrier && ${docker} build -t barrier-agent . && popd

# Build Renaissance, Scalabench and Dacapo
for suite in "renaissance" "scalabench" "dacapo" ; do
    pushd ./benchmarks/${suite} && ${docker} build -t ${suite} . && popd
done

# Build speccpu suite (cca 40min) with special dependencies passed in
pushd ./benchmarks/speccpu && docker build --build-arg SPECCPU_ZIP=speccpu.zip --build-arg BUILD="$(python ./build_args_from_duet_conf.py duet.yml)" -t speccpu . && popd
