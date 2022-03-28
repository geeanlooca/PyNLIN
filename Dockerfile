# FROM python:3.10.4-alpine3.15
FROM python:3.10.3-bullseye
COPY . .
# RUN apk --no-cache add musl-dev linux-headers g++
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install .[dev]
# RUN python -m pytest
