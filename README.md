 # PyNLIN
 A Python package and scripts for the evaluation of nonlinear interference noise in single mode fiber transmissions 

## Installation

## End users
Just clone the repository and `pip install` it.
```bash
git clone https://github.com/geeanlooca/PyNLIN.git
pip install .
```


## Development

### Set up the environment

Clone the repository
```
https://github.com/geeanlooca/PyNLIN.git
```

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
source <env>/bin/activate
```

For development purposes, the package should be installed in the editable mode. Changes you make to the package are immediatly reflected on the installed version and consequently on the scripts using the package.

From the root of the repository:
```bash
make install
```
or 
```bash
pip install -e .[dev]
```
