# TAMM: TriAdapter Multi-Modal Learning for 3D Shape Understanding

[\[project\]](https://alanzhangcs.github.io/tamm-page/) [\[paper\]]() 

[***News***] TAMM is accepted by CVPR 2024. See you in Seattle!

Official code of "TAMM: TriAdapter Multi-Modal Learning for 3D Shape Understanding"

## Installation

If you would to run the inference or (and) training locally, you may need to install the dependendices.

1. Create a conda environment and install [pytorch](https://pytorch.org/get-started/previous-versions/), [MinkowskiEngine](https://nvidia.github.io/MinkowskiEngine/quick_start.html), and [DGL](https://www.dgl.ai/pages/start.html) by the following commands or their official guides:
```
conda create -n tamm python=3.9
conda activate tamm
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine
conda install -c dglteam/label/cu113 dgl
```
2. Install the following packages:
```
pip install huggingface_hub tqdm 
```

## Training Data
We adopt the training and inference data from previous work OpenShape. You can find it [here](https://github.com/Colin97/OpenShape_code)

## Training

## TODO List







