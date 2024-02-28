<div align='center'>

<h2><a href="https://arxiv.org/abs/2310.06773">TAMM: TriAdapter Multi-Modal Learning for 3D Shape Understanding</a></h2>

[Zhihao Zhang](https://alanzhangcs.github.io/)<sup>1*</sup>, [Shengcao Cao](https://shengcao.netlify.app/)<sup>
2*</sup>, [Yuxiong Wang](https://yxw.web.illinois.edu/)<sup>2</sup>

<sup>1</sup>[XJTU](https://www.xjtu.edu.cn/), <sup>2</sup>[UIUC](https://illinois.edu/) <br><sup>*</sup> Equal
Contribution

CVPR 2024
</div>

<p align="center">
    <img src="assets/model.png" alt="overview" width="800" />
</p>

We introduce TriAdapter Multi-Modal Learning (TAMM) -- a novel two-stage learning approach based on three synergetic
adapters. First, our CLIP Image Adapter mitigates the domain gap between 3D-rendered images and natural images, by
adapting the visual representations of CLIP for synthetic image-text pairs. Subsequently, our Dual Adapters decouple the
3D shape representation space into two complementary sub-spaces: one focusing on visual attributes and the other for
semantic understanding, which ensure a more comprehensive and effective multi-modal pre-training.

## Installation

Clone this repository and install the required packages:
```
conda create -n tamm python=3.9
conda activate tamm
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine
conda install -c dglteam/label/cu113 dgl
pip install huggingface_hub tqdm
```


## Training Data

We adopt the training and inference data from previous work OpenShape. You can find
it [here](https://github.com/Colin97/OpenShape_code)

## Training

Run the training by the following command:

```

```

## TODO List







