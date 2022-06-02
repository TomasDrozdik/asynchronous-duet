#!/bin/bash

set -e

while getopts o:t:d: flag ; do
    case "${flag}" in
        o) outdir=${OPTARG}/;;
        t) test=1;;
        d) docker=${OPTARG};;
    esac
done

docker=${docker:-docker}

source venv/bin/activate

for suite in "renaissance" "dacapo" "scalabench" ; do
    if [ -z test ] ; then
        outpath=${outdir}results.${suite}
        duetconf=./benchmarks/${suite}/duet.yml
        log=${outdir}results.${suite}.log
    else
        outpath=${outdir}results.${suite}.test
        duetconf=./benchmarks/${suite}/test.duet.yml
        log=${outdir}results.${suite}.test.log
    fi

    duetbench --docker ${docker} --force --outdir ${outpath} ${duetconf} 2>&1 | tee ${log}
done
