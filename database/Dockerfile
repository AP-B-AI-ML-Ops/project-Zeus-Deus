FROM postgres:latest

RUN echo "initialising database..."

COPY ./init.sql /docker-entrypoint-initdb.d/
