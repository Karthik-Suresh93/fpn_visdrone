#!/bin/bash/
# om sri gurubhyonamaha, harih om. om sri paramathmanenamaha. om gam ganapathayenamaha.om namo narayanaya
#download anaconda (change version if you want)
wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
# run the installer
bash Anaconda3-2018.12-Linux-x86_64.sh
# so changes in your path take place in you current session:
source ~/.bashrc
conda create -n pt0.4.0_py27 python=2
conda activate pt0.4.0_py27
conda install pytorch=0.4.0 -c pytorch
conda install torchvision
pip install scipy pyyaml easydict cython opencv-python matplotlib
