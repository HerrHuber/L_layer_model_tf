L layer model
===============
A simple L Layer Neural Network using TensorFlow.

execute tensorflow docker container
-----------------------------------
CPU:
docker run -it --rm  -u $(id -u):$(id -g) -v /home/user/Projects/ML/L_layer_model_tf:/tmp -w /tmp tensorflow/tensorflow:1.15.4 /bin/bash
GPU:
docker run --gpus all -it --rm  -u $(id -u):$(id -g) -v /home/user/Projects/ML/L_layer_model_tf:/tmp -w /tmp tensorflow/tensorflow:1.15.4-gpu-py3 /bin/bash
