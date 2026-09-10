# Analysis and verification image. Not a training image -- the reported runs used
# CUDA 11.3 on a separate GPU machine, and the CPU wheels installed here will not
# reproduce those runs bit for bit.
#
# What this image is good for: `make check`, reading paper/results_frozen/, and
# re-running the table and figure scripts against predictions you mount in.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        make \
        git \
    && tesseract --version \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-dev.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

# Mount the withheld archive here if you have one.
ENV HVF_ROOT=/app \
    HVF_DATA_ROOT=/data

CMD ["make", "check"]
