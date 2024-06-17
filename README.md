# This repo contains the code for paper [BadActs: A Universal Backdoor Defense in the Activation Space](https://arxiv.org/abs/2405.11227), accepted to ACL 2024 Findings.


## Dependencies
```
conda create -n badacts python=3.9.1
conda activate badacts
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

python setup.py install
pip install scipy==1.13.1
pip install --upgrade "aiohttp!=4.0.0a0,!=4.0.0a1"
pip install language_tool_python
pip install matplotlib==3.5.0
```


## Preparation

### Model
download bert-base-uncased in /PTM

### Poisoned Data
already saved in /poison_data


## Attacks
`python Attacks.py`


## Defense

### Detection
`python BadActs_detection.py`

### Purification
`python BadActs_purification.py`

### Detection then Purification
`python BadActs_detection_then_purification.py`