FROM python:3.11-slim

WORKDIR /app

# writable cache dirs (HF Docker Spaces run as a restricted user)
ENV HF_HOME=/tmp/hf \
    MPLCONFIGDIR=/tmp/mpl \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PYTHONUNBUFFERED=1

# CPU-only torch first (smaller, no CUDA) so the resolver doesn't pull a GPU build
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860
CMD ["python", "app.py"]
