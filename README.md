 # PyNLIN
 A Python package and scripts for the evaluation of nonlinear interference noise in single mode fiber transmissions

# Installation

## End users
Just clone the repository and `pip install` it.
```bash
git clone https://github.com/geeanlooca/PyNLIN.git
cd PyNLIN
pip install .
```


## Development

### Set up the environment

#### Conda
I usually like to install the core numerical packages from conda directly, and let `pip` manage the rest of the dependencies.

```bash
conda create -n <env> python=3.10 --yes
conda activate <env>
conda install numpy scipy matplotlib h5py
```

#### `venv`

```bash
python -m <env>
source <env>/bin/activate # <env>\Scripts\activate.bat under Windows
```

### Install the package
For development purposes, the package should be installed in the editable mode. Changes you make to the package are immediatly reflected on the installed version and consequently on the scripts using the package.

From the root of the repository:
```bash
make install
```
or
```bash
pip install -e .[dev]
```


# Singularity images

Packaging the code in a Singularity image allows us to run code using PyNLIN on the Department's SLURM cluster.

There are two main ways in which you can run build and run a Singularity image:

1. Install Singularity on your local machine, build the image, copy it to the cluster, and submit a job using the image.
2. Build the image using the remote builder and pull the image directly on the cluster to avoid wasting too much time on uploading the image.

> :warning: **The image pulls the latest commit on the `main` branch directly from GitHub. Local edits or commits not pushed to GitHub will not be reflected in the resulting image file

## Local build

Once you have Singularity installed, just run

```bash
sudo singularity build --force singularity.sif singularity.def
```
The resulting `.sif` image file can be used to run python scripts locally using

```bash
singularity exec singularity.sif python <script>.py
```
or uploaded to the cluster.
An example `.slurm` file to run a job on the cluster is provided in the `slurm/` directory of this repository.

## Remote build


## Building 