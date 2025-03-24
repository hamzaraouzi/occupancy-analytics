sudo docker run --gpus all \
    -p 9094:9094 \
    -e source=melen_2.mp4\
    -e bootstrap_server=localhost:9094 \
    -e topic=my-topic \
    docker.io/hamzaraouzi/people-counting