# Project Boilerplate for Data Generation and Deep Learning

This repo contains example usage of some useful libraries for data generation/collection and deep learning in a research setting.
I've been using these tools personally in many applications/projects and have distilled some common use cases here.

## Installation

`pip install -e .`

## Quick Start

### Data Generation

1. Modify `root_dir` in `cfg/generate_data.yaml` to specify where you'd like to keep output files.
2. Run `python scripts/generate_data.py`

The generated data is a simple linear function with Gaussian noise.
We'll train an MLP to fit this data in the next section.

This script uses Hydra to parse the config and generate the folder structure for data saving and logging, and it uses AsyncSavers to save data shards in a separate process.
The `save_every` in the cfg is set to `100`, and `n_samples` is set to `1000` - this means `10` shards will be saved.

The `tag` field gives a non-unique identifier for the data generation run.

3. Run `bash run/generate_all_data.sh` to generate data for 5 random seeds. See the file for how to set the `tag` via commandline arguments.

### Deep Learning

1. Modify `root_dir` in `cfg/train_model.yaml` appropriately.
2. Modify `wandb.logger.{entity, project, group}` with your Weights and Biases account/project details.
3. Run `python scripts/train_model.py` to train the MLP to fit the generated data.

This script uses Hydra to parse configs and generate local logging folder, PyTorch Lightning for training, and Weights and Biases for logging and uploading the trained model.

See `data_learning_boilerplate/data.py` for an example of how to load the shards saved by AsyncSavers.

See `data_learning_boilerplate/model.py` for an implementation of PytorhcLightning module.

## Background

### Data Generation

This is relevant to generating large amounts of data with simulations or some other data collection process.

Common Problems
1. Data size is greater than memory size.
2. Writing to disk is slow.
3. Hard to reproduce/debug

Solutions:
1. Save data in shards.
2. Save data in a parallel process.
3. Save configs/tags

Tools:
* [AsyncSavers](https://github.com/jacky-liang/async_savers) - for 1, 2
* [Hydra Configs](https://hydra.cc/) - for 3

### Deep Learning

Common Problems:
1. A lot of boilerplate code to just get started.
2. Messy logs when runs are across machines/users.
3. Hard to reproduce/debug

Solutions:
1. Use a framework to abstract away non-important logic.
2. Shareable, cloud-based logging.
3. Save configs/tags/models

Tools:
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - for 1, 3
* [Weights and Biases](https://wandb.ai/) - for 2, 3
* [Hydra Configs](https://hydra.cc/) - for 3

### Quick Overview of these libraries:
* [AsyncSavers](https://github.com/jacky-liang/async_savers)
  * Save data in shards
  * Save data in parallel process
* [Hydra Configs](https://hydra.cc/)
  * Composible configs
  * Output directory management
  * Override YAML config via command line
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
  * Abstracts away unimportant training logic
  * Easily scale to multi-gpu, cluster-scale training
* [Weights and Biases](https://wandb.ai/)
  * Much more customizable than TB
  * Offloads logging to the cloud
  * Logs shared by multiple machines / users

### Note

These libraries have many other very useful features that are not explored in this template.
This particular way of combining these libraries is also something that I prefer and has worked for my own research projects.
Modifications are probably necessary depending on your use cases.
My goal is that this can serve as a template/reference for those who are interested in using these tools in research-like projects.