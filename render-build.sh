#!/usr/bin/env bash
set -euxo pipefail

# Install the shared libraries required by OpenCV and MediaPipe on Render.
apt-get update
apt-get install -y --no-install-recommends \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  libgomp1
rm -rf /var/lib/apt/lists/*

python -m pip install --upgrade pip
python -m pip install -r requirements.txt