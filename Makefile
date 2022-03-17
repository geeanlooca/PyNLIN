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
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

version:
	@echo ${python_version_full}
	@echo ${PYTHON}
	@echo ${PIP}

docs:
	$(MAKE) -C docs api
	$(MAKE) -C docs html

pages: docs
	mkdir -p public
	cp -r docs/build/html/* public/

env:
	$(PYTHON) -m venv venv/
	source venv/bin/activate

clean-pages:
	rm -rf public/

clean: clean-docs clean-pages

clean-docs:
	rm -rf docs/source/api
	$(MAKE) -C docs clean

format-all:
	isort pynlin tests 
	black -t py36 pynlin tests
	docformatter --in-place --recursive pynlin tests

format:
	@git diff --name-only master... --diff-filter=ACM | grep .py$$ | xargs -r -t isort
	@git diff --name-only master... --diff-filter=ACM | grep .py$$ | xargs -r -t black -t py36
	@git diff --name-only master... --diff-filter=ACM | grep .py$$ | xargs -r -t docformatter --in-place

lint-all:
	flake8 pynlin examples tests
	pylint pynlin

lint:
	@git diff --name-only master... --diff-filter=ACM | grep .py$$ | xargs -r -t flake8
	@git diff --name-only master... --diff-filter=ACM | grep .py$$ | xargs -r -t pylint

test:
	mkdir -p tests/reports/
	$(PYTHON) -m pytest

doc: clean-docs docs
	xdg-open ./docs/build/html/index.html


.PHONY: clean clean-docs docs lint pages clean-pages doc install version
