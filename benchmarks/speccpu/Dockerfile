FROM ubuntu

# Build as:
# $ docker build --build-arg SPECCPU_ZIP=./speccpu.zip --build-arg BUILD="$(./build_args_from_duet_conf.py duet.yml)" -t speccpu .

# Path to SPECCPU_ZIP file
ARG SPECCPU_ZIP

# String of benchmarks passed to `runcpu --action build ${BUILD}`
# Ideally base it on your duet.yml file to minimalise the size of the image.
ARG BUILD

WORKDIR /duet
COPY ${SPECCPU_ZIP} duet-gcc-linux-x86.cfg ./

RUN apt-get -y update && apt-get -y upgrade &&\
    apt-get -y install unzip gcc g++ gfortran libomp-dev &&\
    unzip ${SPECCPU_ZIP} && rm ${SPECCPU_ZIP} &&\
    cd speccpu && ./install.sh -f &&\
    mv /duet/duet-gcc-linux-x86.cfg /duet/speccpu/config/ &&\
    bash -c "cd  /duet/speccpu && source shrc && runcpu --config duet-gcc-linux-x86.cfg --action build --tuning peak ${BUILD} 2>&1 | tee /duet/build.log" &&\
    apt-get -y clean

CMD bash
