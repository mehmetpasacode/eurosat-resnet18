FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Use a pip release that normalizes package names in CPU wheel index metadata.
RUN python -m pip install --only-binary=:all: pip==25.3

RUN python -m pip install --only-binary=:all: \
        numpy==2.2.6 scikit-learn==1.7.2 tqdm==4.67.3 \
        pandas==2.3.3 matplotlib==3.10.8 seaborn==0.13.2 Pillow==12.1.1 \
    && python -m pip install --only-binary=:all: \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1+cpu torchvision==0.20.1+cpu \
    && python -m pip check

COPY data.py model.py train.py evaluate.py infer.py ./

CMD ["python", "infer.py", "--help"]
