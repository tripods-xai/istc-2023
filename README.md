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

This will build the image if it is not already built and then start a container. Once the image is built and the container is running, you will be in a editable version of the repository. Next, you'll need to install the code for the experiments (see below). **This step needs to be done every time you start the container:**

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
6. Compute the decomposition of TurboAE-cont and TurboAE-binary at block length 40 (`decomposition_turboae_finetune_40`)
7. Benchmark the BER of the following models:
   - Your fresh TurboAE model (`benchmark_turboae_40_2`)
   - The retrained original TurboAE-cont model (`benchmark_turboae_original_finetune_40`)
   - The retrained original TurboAE-binary model (`benchmark_turboae_original_finetune_40`)
   - The benchmark Turbo-155-7 code with BCJR decoding (`estimate_xe_bcjr_block_len_40`)
8. Benchmark the BER of the same models with a junction tree decoder:
   - Your fresh TurboAE model (`benchmark_turboae_40_2_jtree`)
   - The retrained original TurboAE-cont model (`benchmark_turboae_cont_finetuned_jtree`)
   - The retrained original TurboAE-binary model (`benchmark_turboae_binary_finetuned_jtree`)
9. Get statistics on the maximum cluster size of junction trees for random turbo codes (`cluster_tree_statistics_istc`)

Since some filepaths using outputs of previous experiments are hardcoded, you will need to edit `experiment_settings.json` as you go. I'll detail the steps below. Every experiment will output a JSON file in `data/outputs/` with the same name as the experiment. It will contain model settings, training curve data, and additional statistics.

### Experiment 1: Train a fresh version of TurboAE with block length 40

From the `scripts` directory (use `cd scripts` to get there from the root of the repository), run the following experiment:

```
> python run_experiment.py --experiment_id train_turboae_w9_first_no_front_small_batch_block_len_40_2
```

This will additionally write the following outputs:

- `checkpoints/turboae_trainer_ep[EP]_[YYYY]_[MM]_[DD]_[hh]_[mm]_[ss].pt`: the checkpoint for epoch `EP`, written at time `YYYY-MM-DD hh:mm:ss`. Each epoch will have its own checkpoint file.
- `models/train_turboae_w9_first_no_front_small_batch_block_len_40_2_[HASH].pt`: the final model weights. `HASH` is the argument hash of the experiment settings and is used to differentiate the same experiment name, run with different settings.
- `models/train_turboae_w9_first_no_front_small_batch_block_len_40_2_[HASH].pt.opt`: The last state of the optimizer before training finished. This is useful if you want to resume training from the same state.

**Note:** The above experiment will take a while to run. To avoid this step, you can download my checkpoints from https://drive.google.com/file/d/1Ro-UhlW8WkJae-aEuo4j08wD70B1Eszr/view?usp=sharing.

### Experiment 2 & 3: Retrain the decoder of the original TurboAE-cont with block length 40

From the `scripts` directory, run the following experiments:

```
> python run_experiment.py --experiment_id retrain_original_turboae_block_len_40
> python run_experiment.py --experiment_id retrain_original_turboae_binary_block_len_40
```

This will additionally write the following outputs:

- `models/retrain_original_turboae_block_len_40_[HASH].pt`: The retrained TurboAE-cont encoder-decoder pair.
- `models/retrain_original_turboae_block_len_40_[HASH].pt.opt`: The last state of the optimizer before training finished.
- `models/retrain_original_turboae_binary_block_len_40_[HASH].pt`: The retrained TurboAE-binary encoder-decoder pair.
- `models/retrain_original_turboae_binary_block_len_40_[HASH].pt.opt`: The last state of the optimizer before training finished.

### Experiment 4: Compute the BCE decomposition training trajectory of your fresh TurboAE model

Before running this experiment, we will need to modify its parameters in `experiment_settings.json` to use the checkpoints from Experiment 1. In `experiment_settings.json`, change the `checkpoint_daterange` to include the timestamps of the checkpoints from Experiment 1.

```json
"checkpoint_daterange": [["RANGE_START", "RANGE_END"]],
```

For example, to include checkpoints saved between 2025-04-28 00:00:00 and 2025-04-29 00:00:00, you would change the `checkpoint_daterange` to:

```json
"checkpoint_daterange": [["2025_04_28_00_00_00", "2025_04_29_00_00_00"]],
```

Once this is done, run the experiment from the `scripts` directory:

```
> python run_experiment.py --experiment_id decomposition_trajectory_turboae_40_2
```

### Experiment 5: Compute the BCE training curve of your fresh TurboAE model

Before running this experiment, we again need to modify its parameters in `experiment_settings.json` to use the checkpoints from Experiment 1. In `experiment_settings.json`, change the `checkpoint_daterange` to include the timestamps of the checkpoints from Experiment 1.

```json
"checkpoint_daterange": [["RANGE_START", "RANGE_END"]],
```

Once this is done, run the experiment from the `scripts` directory:

```
> python run_experiment.py --experiment_id xe_trajectory_turboae_40_2
```

### Experiment 6: Compute the decomposition of TurboAE-cont and TurboAE-binary at block length 40

Before running this experiment, we will need to modify its parameters in `experiment_settings.json` to use the finetuned models from Experiment 2 and 3. In `experiment_settings.json`, change the `decoder_path__turboae_type` to include the paths (**relative to the `scripts` directory**) to the finetuned models from Experiment 2 and 3.

```json
"decoder_path__turboae_type": [["PATH_TO_TURBOAE_BINARY_MODEL", "binary"], ["PATH_TO_TURBOAE_CONT_MODEL", "continuous"]],
```

For example, you might have

```json
"decoder_path__turboae_type": [["../models/retrain_original_turboae_binary_block_len_40_e92a5.pt", "binary"], ["../models/retrain_original_turboae_block_len_40_88214.pt", "continuous"]],
```

Once this is done, run the experiment from the `scripts` directory:

```
> python run_experiment.py --experiment_id decomposition_turboae_finetune_40
```

### Experiment 7: Benchmark the BER of the following models:

First we'll need to modify the `experiment_settings.json` file to use our trained model from Experiment 1 and the finetuned models from Experiment 2 and 3.

1. For the experiment `benchmark_turboae_40_2`, change the `encoder_decoder_path` to the path to the model from Experiment 1. For example,

```json
"encoder_decoder_path": ["../models/train_turboae_w9_first_no_front_small_batch_block_len_40_2_e3a1b.pt"],
```

2. For the experiment `benchmark_turboae_original_finetune_40`, change `decoder_path__turboae_type` to include the paths to the finetuned models from Experiment 2 and 3. For example,

```json
"decoder_path__turboae_type": [["../models/retrain_original_turboae_binary_block_len_40_a91f1.pt", "binary"], ["../models/retrain_original_turboae_block_len_40_e45db.pt", "continuous"]],
```

Now, from the `scripts` directory, run the following experiments:

```
> python run_experiment.py --experiment_id benchmark_turboae_40_2
> python run_experiment.py --experiment_id benchmark_turboae_original_finetune_40
> python run_experiment.py --experiment_id estimate_xe_bcjr_block_len_40
```

### Experiment 8: Benchmark the BER of the same models with a junction tree decoder

First we'll need to modify the `experiment_settings.json` file to use our trained model from Experiment 1.

For the experiment `benchmark_turboae_40_2_jtree`, change the `encoder_decoder_path` to the path to the model from Experiment 1. For example,

```json
"encoder_decoder_path": ["../models/train_turboae_w9_first_no_front_small_batch_block_len_40_2_e3a1b.pt"],
```

Now, from the `scripts` directory, run the following experiments:

```
> python run_experiment.py --experiment_id benchmark_turboae_40_2_jtree
> python run_experiment.py --experiment_id benchmark_turboae_cont_finetuned_jtree
> python run_experiment.py --experiment_id benchmark_turboae_binary_finetuned_jtree
```

### Experiment 9: Get statistics on the maximum cluster size of junction trees for random turbo codes

From the `scripts` directory, run the following experiment:

```
> python run_experiment.py --experiment_id cluster_tree_statistics_istc
```

## Generating Figures

All figures are generated using the `notebooks/istc-2023-plots.ipynb` notebook.
