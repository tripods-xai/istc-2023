# istc-2023
Code for *Decomposing the Training of Deep Learned Turbo codes via a Feasible MAP Decoder*

## TODO
- [x] Transfer code over
- [x] Only keep experiments in experiment_settings.json that are relevant
- [ ] Test for reproducibility
- [ ] Upload checkpoints to UIC google drive and include link
- [ ] Add instructions on how to run


For any questions or inquries, please email me at *abhmul@gmail.com*.

## Instructions for Use

*The below directions have been tested on a computer running Ubuntu 20.04.*

### Setting up

Make sure you have the following tools installed first:
1. [Mamba or Micromamba](https://mamba.readthedocs.io/en/latest/installation.html)
2. [Docker](https://www.docker.com/)

#### Setting up a Local Python Environment with CPU Pytorch

This project will use *Python 3.9*. Once you've cloned this repository, in this folder run
```
> mamba env create -f environment.yml
> mamba activate istc_2023
> pip install -r requirements.txt
```

From this environment you will be able to run the code in this repository. You can also run the included test suite. However, this environment will *not* use a GPU if you have one. I have provided directions below on how to do this with Docker.

#### Setting up a Docker Python Environment with GPU Pytorch

This project will use an image built off of the latest PyTorch GPU Docker image. To set up the Docker container run