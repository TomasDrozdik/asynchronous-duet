#/bin/bash

set -e

IMAGE_A=${1?Need $1 Duet benchmark docker image A name}
IMAGE_B=${2?Need $2 Duet benchmark docker image B name}

CONTAINER_A=duetA-${IMAGE_A}
CONTAINER_B=duetB-${IMAGE_B}

start_containers () {
    echo "Start containers"
    docker run --name ${CONTAINER_A} -it --rm -d ${IMAGE_A} "/bin/bash"
    docker run --name ${CONTAINER_B} -it --rm -d ${IMAGE_B} "/bin/bash"
}

exec_setup_stage () {
    echo "Execute setup stage"

    for container in ${CONTAINER_A} ${CONTAINER_B} ; do
        docker exec ${container} "./setup.sh"
    done
}

exec_run_stage() {
    echo "Execute run stage"

    for container in ${CONTAINER_A} ${CONTAINER_B} ; do
        docker exec ${container} "./run.sh" &
    done
    wait
}

exec_process_stage () {
    echo "Execute process stage"

    for container in ${CONTAINER_A} ${CONTAINER_B} ; do
        docker exec ${container} "./process.sh"
    done
}

copy_results () {
    echo "Copy results"

    for container in ${CONTAINER_A} ${CONTAINER_B} ; do
        docker cp ${container}:/duet/results.csv ${container}-results.csv
    done
}

stop_containers () {
    echo "Stop containers"

    for container in ${CONTAINER_A} ${CONTAINER_B} ; do
        docker stop ${container}
    done
}

main () {
    echo "DuetBenchmark"
    echo "A image: ${IMAGE_A}"
    echo "B image: ${IMAGE_B}"

    start_containers
    exec_setup_stage
    exec_run_stage
    exec_process_stage
    copy_results
    stop_containers
}

main
