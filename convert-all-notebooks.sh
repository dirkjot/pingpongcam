#/bin/bash

find . -type d -name .ipynb_checkpoints -prune \
       -o \
       -type f -name "*.ipynb" -execdir jupyter nbconvert {} --to python \;

