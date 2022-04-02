FROM ubuntu:19.04

COPY busybox /busybox
COPY rpzsudo /rpzsudo

COPY rpztar /rpztar

COPY data.tgz /reprozip_data.tgz

COPY rpz-files.list /rpz-files.list
RUN \
    chmod +x /busybox /rpzsudo /rpztar && \
    cd / && /rpztar /reprozip_data.tgz /rpz-files.list
