# Base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update and install GStreamer and plugins
RUN apt-get update && apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-alsa \
    python3 \
    python3-pip \
    python3-opencv \
    && apt-get clean

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY app/ app/
COPY model.pt model.pt

# Set work directory
WORKDIR /app

# Default command to keep the container running (can be overridden)
CMD ["Python", "main.py", "--model", "../model.pt"]
