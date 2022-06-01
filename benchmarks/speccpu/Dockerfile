FROM ubuntu

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install unzip gcc g++ gfortran libomp-dev

WORKDIR /duet
COPY . .

# Unpack speccpu
RUN unzip speccpu.zip
RUN rm speccpu.zip

# Install speccpu and duet config
RUN cd speccpu && ./install.sh -f
RUN mv /duet/duet-gcc-linux-x86.cfg /duet/speccpu/config/

# Build all speccpu bencharks, this might take some time, for that reason consider running it
# via `docker run` -> `docker exec` -> `docker commit`.
RUN bash -c "cd  /duet/speccpu && source shrc && runcpu --config duet-gcc-linux-x86.cfg --action build all 2>&1 | tee /duet/build.log"

CMD bash
