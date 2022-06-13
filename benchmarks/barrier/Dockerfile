FROM openjdk:11.0.8-jdk

WORKDIR /barrier
COPY . .

RUN apt-get update && apt-get upgrade -y &&\
    apt-get install -y ant gcc make &&\
    bash ./install-ubench-agent.sh &&\
    make &&\
    apt-get clean -y

CMD bash
