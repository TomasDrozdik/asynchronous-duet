#!/bin/bash

# Initialize and run duetbench on fresh AWS Ubuntu22 AMI Instance
#
# from: https://github.com/TomasDrozdik/asynchronous-duet
#
# Prerequisites:
# - Ubuntu22 EC2 instance
# - Existing duetbench tar archive on remote server that is accesible for reading from the instance
#     - archive has fixed structure created by `make-archive.sh`
# - Existing ftp server to publish results to

set -euo pipefail

HELP="run-instance.sh <remote_path> <remote_config>"

# TODO: Replace with appropiate values to execute without arguments
REMOTE_PATH=${1:?${HELP}} # path with ${REMOTE_CONFIG} and ${image}.tar for image in ${IMAGES} see below
REMOTE_CONFIG=${2:?${HELP}} # filename of config e.g. duetbench.config-1run.tar.gz
FTP_PATH="ftp://shiva.ms.mff.cuni.cz"
IMAGES="renaissance dacapo scalabench speccpu"

# Get duetbench
wget ${REMOTE_PATH}/${REMOTE_CONFIG}
tar -xvf ${REMOTE_CONFIG}
rm ${REMOTE_CONFIG}

# Install packages
sudo apt update -y
sudo apt install -y python3 python3-pip podman
export PATH="$HOME/.local/bin:$PATH"

# Load docker images
for image in ${IMAGES} ; do
    wget -O - ${REMOTE_PATH}/${image}.tar | podman image load
done

# Install duetbench
cd duetbench  # fixed archive name by make-archive.sh
pip install duet.tar.gz

# Run duetbench
configs=$(find . -name "*duet.yml" | tr "\n" " ")
outdir="results.drozdikt.$(hostname -f).$(date '+%Y-%m-%d--%H-%M-%S--%s')"
log=${outdir}.log
bash -c "duetbench --outdir ${outdir} --verbose --docker podman -- ${configs} &> ${log}"

tar="${outdir}.tar"
tar -cvzf "${tar}" "${outdir}" "${log}"
curl --upload-file "${tar}" "${FTP_PATH}/${tar}"

sudo poweroff
