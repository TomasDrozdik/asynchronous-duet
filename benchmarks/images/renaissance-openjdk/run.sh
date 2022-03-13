#!/bin/bash

usage="./run.sh <benchmark> <repetitions>"

if [ $# -ne 2 ] then
    echo $usage
    exit 1
fi

java -jar renaissance-gpl-0.13.0.jar --json results/results.json --repetitions $2 $1
