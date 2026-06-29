FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir nuclio-sdk mmengine opencv-python-headless

RUN pip install --no-cache-dir mmcv \
    -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.4/index.html

RUN pip install --no-cache-dir mmdet mmpose

COPY handler.py               /opt/nuclio/
COPY hrnet-w32_8-kp.py        /opt/nuclio/
COPY RSC_Keypoints.py         /opt/nuclio/
COPY best_coco_AP_epoch_30.pth /opt/nuclio/
