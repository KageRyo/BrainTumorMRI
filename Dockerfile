FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app
COPY configs ./configs
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1 \
    && pip install --no-cache-dir -e ".[demo]"

EXPOSE 7860

CMD ["python", "app/gradio_app.py", "--checkpoint", "outputs/convnext_tiny_mtl/best.pt", "--device", "cpu", "--host", "0.0.0.0"]
