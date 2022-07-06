#!/bin/bash

set -ex

HELP="./move-images.sh [-s] [-l] [-d docker] <ssh_target> <target_directory> <image>..."

save=false
load=false
docker=docker

while getopts sld: flag ; do
    case "${flag}" in
        s) save=true ;;
        l) load=true ;;
        d) docker=${OPTARG} ;;
    esac
done

shift $((OPTIND - 1))

SSH_TARGET=${1}
TARGET_DIRECTORY=${2}

shift 2
IMAGES=${*}

if [[ -z ${IMAGES} || -z ${SSH_TARGET} || -z ${TARGET_DIRECTORY} ]] ; then
    echo "Missing positional arguments, print help and exit..."
    echo ${HELP}
    exit 1
fi

if [ "${save}" = true ] ; then
    for image in ${IMAGES} ; do
        if [ ! -f ${image}.tar ] ; then
            ${docker} image save ${image} > ${image}.tar
        fi

        scp ${image}.tar ${SSH_TARGET}:${TARGET_DIRECTORY}
    done
fi

if [ "${load}" = true ] ; then
    for image in ${IMAGES} ; do
        ssh ${SSH_TARGET} ${docker} load --input ${TARGET_DIRECTORY}/${image}.tar
    done
fi
