# Base image
FROM nvcr.io/nvidia/l4t-base:35.4.1

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
    ffmpeg \
    && apt-get clean

# Install additional Python dependencies if needed
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt


COPY app/ app/
COPY yolov8n.trt yolov8n.trt
# Set work directory
WORKDIR /app/

# Default command to keep the container running (can be overridden)
ENTRYPOINT ["python3", "main.py", "--model", "yolov8n.trt"]
