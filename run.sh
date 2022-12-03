#!/bin/bash

# Run duetbench on general linux worker machine with podman
#
# from: https://github.com/TomasDrozdik/asynchronous-duet
#
# Prerequisites:
# - python3, pip3, podman
# - Existing duetbench tar archive on remote server that is accesible for reading from the instance
#     - archive has fixed structure created by `make-archive.sh`
# - Existing ftp server to publish results to

set -euo pipefail

# TODO: Configure following arguments before running <------------------------
FTP_PATH="ftp://shiva.ms.mff.cuni.cz" # to upload results to
REMOTE_PATH="https://d3s.mff.cuni.cz/f/temporary/duet/drozdik/" # to fetch config from
REMOTE_CONFIG="${1:?}" # name of the config ro run
IMAGES="renaissance dacapo scalabench speccpu" # to download from remote
REPETITIONS="1" # how many times to rerun given config

# Get duetbench
wget "${REMOTE_PATH}/${REMOTE_CONFIG}" -O "${REMOTE_CONFIG}"
tar -xvf "${REMOTE_CONFIG}"
rm "${REMOTE_CONFIG}"

# Load docker images
for image in ${IMAGES} ; do
    wget "${REMOTE_PATH}/${image}.tar" -O "${image}.tar"
    podman image load <"${image}.tar"
    rm "${image}.tar"
done

# Install duetbench
cd duetbench  # fixed archive name by make-archive.sh
pip install duet.tar.gz

# Run duetbench
configs=$(find . -name "*duet.yml" | tr "\n" " ")
for i in $(seq ${REPETITIONS}) ; do
    outdir="results.drozdikt.$(hostname -f).${i}.$(date '+%Y-%m-%d--%H-%M-%S--%s')"
    log=${outdir}.log
    bash -c "duetbench --outdir ${outdir} --verbose --docker podman -- ${configs} &> ${log}"

    tarball="${outdir}.tar.gz"
    tar -cvzf "${tarball}" "${outdir}" "${log}"
    curl --upload-file "${tarball}" "${FTP_PATH}/${tarball}"
done
