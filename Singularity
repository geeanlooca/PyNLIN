Bootstrap: docker
From: python:3.10.3-bullseye

%post
    apt update
    apt install git
    git clone https://github.com/geeanlooca/PyNLIN
    cd PyNLIN
    git checkout dei-not-peg-time
    python -m pip install .

%runscript
    python -c "import pynlin"
