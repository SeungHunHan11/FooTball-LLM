docker build -t shh_exp .

docker run --gpus all -it -h llm \
        -p 1111:1111 \
        --ipc=host \
        --name llm \
        -v /home/seunghun/바탕화면/deploy:/Project \
        pains bash

