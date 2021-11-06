FROM alpine:3.15
RUN apk update && apk upgrade
RUN apk --no-cache --update-cache add bash curl openjdk8 gcc gfortran python3 python3-dev\
    build-base wget freetype-dev libpng-dev openblas-dev
RUN python3 -m ensurepip && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install pandas

WORKDIR /duet
COPY . .
RUN ./install.sh
