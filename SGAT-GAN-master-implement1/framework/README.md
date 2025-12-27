<h1 align="center">SGAT-GAN</h1>
<div align="center">
  <a href="https://github.com/imperial-qore/PreGAN/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-red.svg" alt="License">
  </a>
   <a>
    <img src="https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg" alt="Python 3.7, 3.8">
  </a>
   <a>
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FPreGAN&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false" alt="Hits">
  </a>
   <a href="https://github.com/imperial-qore/PreGAN/actions">
    <img src="https://github.com/imperial-qore/COSCO/workflows/DeFog-Benchmarks/badge.svg" alt="Actions Status">
  </a>
 <br>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/yolo?label=docker%20pulls%3A%20yolo" alt="Docker pulls yolo">
  </a>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/pocketsphinx?label=docker%20pulls%3A%20pocketsphinx" alt="Docker pulls pocketsphinx">
  </a>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/aeneas?label=docker%20pulls%3A%20aeneas" alt="Docker pulls aeneas">
  </a>
</div>

To address the problem of node overload or even failure caused by limited resources in Industrial Internet of Things (IIoT) edge computing environments, we propose a fault-tolerant method based on edge intelligence called SGAT-GAN.

Install dependencies.
```console
sudo apt -y update
python3 -m pip --upgrade pip
python3 -m pip install matplotlib scikit-learn
python3 -m pip install -r requirements.txt
python3 -m pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
export PATH=$PATH:~/.local/bin
```
Run the code using the following command.
```console
python3 main.py
````

