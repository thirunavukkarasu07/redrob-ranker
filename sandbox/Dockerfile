FROM python:3.11-slim

WORKDIR /app

# writable cache dirs (HF Docker Spaces run as a restricted user)
ENV HF_HOME=/tmp/hf \
    MPLCONFIGDIR=/tmp/mpl \
    STREAMLIT_SERVER_HEADLESS=true \
    PYTHONUNBUFFERED=1

# CPU-only torch first (smaller, no CUDA) so the resolver doesn't pull a GPU build
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
