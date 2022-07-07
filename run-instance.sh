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

HELP="run-instance.sh <path_to_duetbench_archive>"

# Get duetbench
# TODO: Replace with path to remote archive
ARCHIVE_PATH=${1:?${HELP}}
ARCHIVE="duetbench.tar.gz"
# TODO(Get archive) wget ${ARCHIVE_PATH} -O ${ARCHIVE}
tar -xvf duetbench.tar.gz
rm ${ARCHIVE}

# Install packages
sudo apt update -y
sudo apt install -y python3 python3-pip podman
export PATH="$HOME/.local/bin:$PATH"

# Load docker images
cd duetbench
for image in $(find . -name "*.tar") ; do
    podman image load --input ${image}
    rm ${image}
done

# Install duetbench
pip install duet.tar.gz

# Run duetbench
configs=$(find . -name "*duet.yml" | tr "\n" " ")
outdir="results.drozdikt.$(hostname -f).$(date '+%s')"
log=${outdir}.log
bash -c "duetbench --outdir ${outdir} --verbose --docker podman -- ${configs} &> ${log}"

tar=${outdir}.tar
tar -cvzf ${tar} ${outdir} ${log}
sftp anonymous@shiva.ms.mff.cuni.cz:~ <<< "put ${tar}"

sudo poweroff
