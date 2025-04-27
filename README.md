# istc-2023

Code for _Decomposing the Training of Deep Learned Turbo codes via a Feasible MAP Decoder_

For any questions or inquries, please email me at *abhmul@gmail.com*.

## Instructions for Use

_The below directions have been tested on a computer running Ubuntu 20.04._

### Setting up

Make sure you have the following tools installed first:

1. [Mamba or Micromamba](https://mamba.readthedocs.io/en/latest/installation.html)
2. [Docker](https://www.docker.com/)

#### Setting up a Local Python Environment with CPU Pytorch

This project will use _Python 3.9_. Once you've cloned this repository, in this folder run

```
> mamba env create -f environment.yml
> mamba activate istc_2023
> pip install -r requirements.txt
```

From this environment you will be able to run the code in this repository. You can also run the included test suite. However, this environment will _not_ use a GPU if you have one. I have provided directions below on how to do this with Docker.

#### Setting up a Docker Python Environment with GPU Pytorch

This project will use an image built off of the latest PyTorch GPU Docker image. First start the docker daemon:

```
> sudo dockerd
```

To set up the Docker container run the included `run_docker.sh` script from the root of this repository.

```
> ./run_docker.sh
```

This will build the image if it is not already built and then start a container. You can then run the code in this repository from the container.

## Running the Code

To run the code in this repository from the container, you can use the following command:

```

```
