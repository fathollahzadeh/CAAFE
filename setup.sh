#!/bin/bash

rm -rf venv
python3.9 -m venv venv
source venv/bin/activate
python3.9 -m pip install --upgrade pip
python3.9 -m pip install scipy
python3.9 setup.py install


# sudo apt --fix-broken install
