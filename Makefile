python_version_full := $(wordlist 2,4,$(subst ., ,$(shell python --version 2>&1)))
python_version_major := $(word 1,${python_version_full})
python_version_minor := $(word 2,${python_version_full})
python_version_patch := $(word 3,${python_version_full})

python_cmd.python.2 := python3
python_cmd.python.3 := python
pip_cmd.python.2 := pip3
pip_cmd.python.3 := pip

PYTHON := ${python_cmd.python.${python_version_major}}
PIP := ${pip_cmd.python.${python_version_major}}

# https://stackoverflow.com/questions/8327144/setting-make-options-in-a-makefile

install:
	$(PIP) install -e .[dev]

version:
	@echo ${python_version_full}
	@echo ${PYTHON}
	@echo ${PIP}


format-all:
	isort pynlin tests scripts
	black -t py36 pynlin tests scripts
	docformatter --in-place --recursive pynlin tests scripts

format:
	@git diff --name-only master... --diff-filter=ACM | grep .py$$ | xargs -r -t isort
	@git diff --name-only master... --diff-filter=ACM | grep .py$$ | xargs -r -t black -t py36
	@git diff --name-only master... --diff-filter=ACM | grep .py$$ | xargs -r -t docformatter --in-place

lint-all:
	flake8 pynlin tests scripts
	pylint pynlin

lint:
	@git diff --name-only master... --diff-filter=ACM | grep .py$$ | xargs -r -t flake8
	@git diff --name-only master... --diff-filter=ACM | grep .py$$ | xargs -r -t pylint

test:
	mkdir -p tests/reports/
	$(PYTHON) -m pytest



.PHONY: clean lint install version
