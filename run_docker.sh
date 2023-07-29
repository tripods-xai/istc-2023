{ # try
    sudo docker run --rm --gpus all --network host -v $(pwd)/..:/code -w /code -it istc_2023:latest bash
} || { # catch
    sudo docker build . -t istc_2023:latest
sudo docker run --rm --gpus all --network host -v $(pwd)/..:/code -w /code -it istc_2023:latest bash
}


