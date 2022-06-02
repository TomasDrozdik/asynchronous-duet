#!/bin/bash

set -e

while getopts d: flag ; do
    case "${flag}" in
        d) docker=${OPTARG};;
    esac
done

docker=${docker:-docker}

for suite in "renaissance" "scalabench" "dacapo" ; do
    pushd ./benchmarks/${suite} && ${docker} build -t ${suite} . && popd
done
