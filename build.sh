#!/bin/bash

set -e

while getopts d: flag ; do
    case "${flag}" in
        d) docker=${OPTARG};;
        s) speccpu_path=${OPTARG};;
        c) speccpu_config=${OPTARG};;
    esac
done

docker=${docker:-docker}

for suite in "renaissance" "scalabench" "dacapo" ; do
    pushd ./benchmarks/${suite} && ${docker} build -t ${suite} . && popd
done

# Build speccpu suite (cca 40min)
speccpu_path=${speccpu_path:-speccpu.zip}
speccpu_config=${speccpu_config:-duet.yml};;
pushd ./benchmarks/${suite} docker build --build-arg SPECCPU_ZIP=${speccpu_path} --build-arg BUILD="$(./build_args_from_duet_conf.py ${speccpu_config})" -t speccpu . && popd
