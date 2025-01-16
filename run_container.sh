sudo docker run --gpus all \
    -p 8080:8080 \
    -e source=rtsp://35.184.184.200:8080/melen_2.mp4\
    -e bootstrap_server=localhost:9094 \
    -e topic=my-topic \
    -e ip_rtsp=localhost \
    -e port_rtsp=8080 \
    docker.io/hamzaraouzi/people-counting