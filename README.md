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

This will build the image if it is not already built and then start a container. Once the image is built and the container is running, you will be in a editable version of the repository. Next, you'll need to install the code for the experiments:

```
> root@[your_machine]:/code# ls
Dockerfile  README.md       docker_requirements.txt  example_numpy.py          figs    notebooks   requirements.txt  scripts   src           tests
LICENSE     conventions.md  environment.yml          experiment_settings.json  models  pytest.ini  run_docker.sh     setup.py  src.egg-info
> root@[your_machine]:/code# pip install -e .
Obtaining file:///code
  Preparing metadata (setup.py) ... done
Installing collected packages: src
  DEPRECATION: Legacy editable install of src==0.1.0 from file:///code (setup.py develop) is deprecated. pip 25.3 will enforce this behaviour change. A possible replacement is to add a pyproject.toml or enable --use-pep517, and use setuptools >= 64. If the resulting installation is not behaving as expected, try using --config-settings editable_mode=compat. Please consult the setuptools documentation for more information. Discussion can be found at https://github.com/pypa/pip/issues/11457
  Running setup.py develop for src
Successfully installed src-0.1.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
```

Check the pytorch is using the GPU:

```
> root@[your_machine]:/code# python
>>> import torch
>>> torch.cuda.is_available()
True
```

## Running the Code

The key experiments to run to regenerate the results in the paper are below. The settings of each are specified in `experiment_settings.json`.

1. Train a fresh version of TurboAE with block length 40 (`train_turboae_w9_first_no_front_small_batch_block_len_40_2`)
2. Retrain the decoder of the original TurboAE-cont with block length 40 (`retrain_original_turboae_block_len_40`)
3. Retrain the decoder of the original TurboAE-binary with block length 40 (`retrain_original_turboae_binary_block_len_40`)
4. Compute the BCE decomposition training trajectory of your fresh TurboAE model (`decomposition_trajectory_turboae_40_2`)
5. Compute the BCE training curve of your fresh TurboAE model (`xe_trajectory_turboae_40_2`)
6. Benchmark the BER of the following models:
   - Your fresh TurboAE model (`benchmark_turboae_40_2`)
   - The retrained original TurboAE-cont model (`benchmark_turboae_original_finetune_40`)
   - The retrained original TurboAE-binary model (`benchmark_turboae_original_finetune_40`)
   - The benchmark Turbo-155-7 code with BCJR decoding (`estimate_xe_bcjr_block_len_40`)
7. Benchmark the BER of the same models with a junction tree decoder:
   - Your fresh TurboAE model (`benchmark_turboae_40_2_jtree`)
   - The retrained original TurboAE-cont model (`benchmark_turboae_cont_finetuned_jtree`)
   - The retrained original TurboAE-binary model (`benchmark_turboae_binary_finetuned_jtree`)
8. Get statistics on the maximum cluster size of junction trees for random turbo codes (`cluster_tree_statistics_istc`)

Since some filepaths using outputs of previous experiments are hardcoded, you will need to edit `experiment_settings.json` as you go. I'll detail the steps below:

### Experiment 1: Train a fresh version of TurboAE with block length 40

```
> cd scripts
> python run_experiment.py --experiment_id train_turboae_w9_first_no_front_small_batch_block_len_40_2
```

This will write a JSON file output to `outputs`.
