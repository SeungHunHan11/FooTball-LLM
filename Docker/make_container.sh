docker build -t ftllm .

docker run --gpus all -it -h ftllm \
        -p 1111:1111 \
        --ipc=host \
        --name ftllm \
        ftllm bash

