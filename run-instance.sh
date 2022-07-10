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

HELP="run-instance.sh <path_to_duetbench_archives> <ftp_url>"

# Get duetbench
# TODO: Replace with path to remote archive
ARCHIVE_PATH=${1:?${HELP}}
FTP_PATH=${2:?${HELP}}
IMAGES="renaissance dacapo scalabench speccpu"

ARCHIVE="duetbench.tar.gz"
# TODO(Get archive) wget ${ARCHIVE_PATH} -O ${ARCHIVE}
tar -xvf duetbench.tar.gz
rm ${ARCHIVE}

# Install packages
sudo apt update -y
sudo apt install -y python3 python3-pip podman
export PATH="$HOME/.local/bin:$PATH"

# Load docker images
for image in ${IMAGES} ; do
    wget -O - ${ARCHIVE_PATH}/${image}.tar | podman image load
done

# Install duetbench
cd duetbench
pip install duet.tar.gz

# Run duetbench
configs=$(find . -name "*duet.yml" | tr "\n" " ")
outdir="results.drozdikt.$(hostname -f).$(date '+%s')"
log=${outdir}.log
bash -c "duetbench --outdir ${outdir} --verbose --docker podman -- ${configs} &> ${log}"

tar=${outdir}.tar
tar -cvzf ${tar} ${outdir} ${log}
ftp ${FTP_PATH} <<< "put ${tar}"

sudo poweroff
