#!/usr/bin/env bash

set -ex

usage() {
cat << EOF
usage: ./duetbench -A <docker_image> -B <docker_image> --docker <command> --rm

Mandatory:
-A     specify docker image to run as duet A
-B     specify docker image to run as duet B

Optional:
-d     specify command to run instead of docker, e.g. podman
-h     print this message and exit
EOF
    exit 2
}

# Parse CLI arguments

image_a=""
image_b=""
docker_cmd="docker"
rm=0

[ $? -ne 0 ] && usage

while getopts "A:B:d:h" o ; do
    case "${o}" in
        h)
            usage
            ;;
        A) # Specify docker image for duetA
            image_a=${OPTARG}
            container_a=duetA-${image_a}
            #check_image_exists ${image_a}
            ;;
        B) # Specify docker image for duetB
            image_b=${OPTARG}
            container_b=duetB-${image_b}
            #check_image_exists ${image_b}
            ;;
        d)
            docker_cmd=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done

start_containers () {
    echo "Start containers"
    ${docker_cmd} run --name ${container_a} -it --rm -d ${image_a} "/bin/bash"
    ${docker_cmd} run --name ${container_b} -it --rm -d ${image_b} "/bin/bash"
}

exec_setup_stage () {
    echo "Execute setup stage"

    for container in ${container_a} ${container_b} ; do
        ${docker_cmd} exec ${container} "./setup.sh"
    done
}

exec_run_stage() {
    echo "Execute run stage"

    for container in ${container_a} ${container_b} ; do
        ${docker_cmd} exec ${container} "./run.sh" 2>&1 | sed -e "s/^/[${container}]     /g" &
    done
    wait
}

exec_process_stage () {
    echo "Execute process stage"

    for container in ${container_a} ${container_b} ; do
        ${docker_cmd} exec ${container} "./process.sh"
    done
}

copy_results () {
    echo "Copy results"

    date=$(date "+%Y%m%d-%H%M%S")
    for container in ${container_a} ${container_b} ; do
        results_dir=results.${date}.${container}
        rm -rf ${results_dir}
        mkdir ${results_dir}
        ${docker_cmd} cp ${container}:/duet/results/ ${results_dir}
        mv ${results_dir}/results/* ${results_dir} && rmdir ${results_dir}/results
    done
}

stop_containers () {
    echo "Stop containers"

    for container in ${container_a} ${container_b} ; do
        ${docker_cmd} stop ${container}
    done
}

main () {
cat <<EOF
DuetBenchmark:
A: ${image_a} -> ${container_a}
B: ${image_b} -> ${container_b}
docker_cmd: ${docker_cmd}
rm: ${rm}
EOF

    start_containers
    exec_setup_stage
    exec_run_stage
    exec_process_stage
    copy_results
    stop_containers
}
main