#!/bin/bash

rm -rf venv
python3.9 -m venv venv
source venv/bin/activate
python3.9 -m pip install --upgrade pip
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
python3.9 -m pip install -r requirements.txt
#pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
#pip install catboost
#pip install xgboost
#pip install hyperopt
#pip install lightgbm
#pip install tiktoken==0.5.2
#pip install groq==0.5.0
#pip install ipython
##pip install tabpfn[full]
#pip install google-generativeai
#pip install scipy
##python3.9 setup.py install


# sudo apt --fix-broken install


