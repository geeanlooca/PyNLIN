FROM python:3.10.4-alpine3.15
COPY . .
RUN pip install .[dev]
# RUN python -m pytest
