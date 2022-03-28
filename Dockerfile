FROM python:3.10.3-bullseye
COPY . .
RUN pip install .[dev]
# RUN python -m pytest
