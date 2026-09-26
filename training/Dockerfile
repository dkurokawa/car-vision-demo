FROM nvcr.io/nvidia/pytorch:25.02-py3

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The NGC image already ships a working opencv. Install ultralytics without
# its bundled opencv-python to avoid the cv2.dnn.DictValue ABI clash.
RUN pip install --no-cache-dir --no-deps ultralytics==8.3.40 \
    && pip install --no-cache-dir \
        py-cpuinfo \
        ultralytics-thop \
        seaborn==0.13.2 \
        scikit-learn==1.5.2

RUN mkdir -p /app/datasets /app/models /app/runs /app/scripts

COPY scripts/ /app/scripts/
RUN chmod +x /app/scripts/*.sh || true

CMD ["bash", "/app/scripts/start_training.sh"]
