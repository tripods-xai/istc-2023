# sudo docker build . -t idc:latest
# sudo docker run --rm --gpus all --network host -v $(pwd):/code/idc -w /code/idc -u $(id -u):$(id -g) -it idc:latest bash
sudo docker run --rm --gpus all --network host -v $(pwd)/..:/code -w /code -it idc:latest bash
