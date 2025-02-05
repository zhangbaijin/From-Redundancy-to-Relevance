# 🔥🔥🔥 [NAACL 2025] From redundancy to relevance: Enhancing explainability in multimodal large language models

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arXiv-2311.17911-B21A1B)](https://arxiv.org/abs/2406.06579)
[![GitHub Stars](https://img.shields.io/github/stars/zhangbaijin/From-Redundancy-to-Relevance?style=social)](zhangbaijin/From-Redundancy-to-Relevance)

![image](https://github.com/zhangbaijin/From-Redundancy-to-Relevance/blob/main/information-flow.png)


## Setup

```
conda env create -f environment.yml
conda activate redundancy
python -m pip install -e transformers-4.29.2
```
#### Our modify in llava.py/llava_arch.py/llava_llama.py
```
retain_grad()
required_grad()=True 
```

## Evaluation

The following evaluation requires for MSCOCO 2014 dataset. Please download [here](https://cocodataset.org/#home) and extract it in your data path.

Besides, it needs you to prepare the following checkpoints of 7B base models:

- Download [LLaVA-1.5 merged 7B model](https://huggingface.co/liuhaotian/llava-v1.5-7b) and specify it at [Line 14](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/eval_configs/llava-1.5_eval.yaml#L14) of `eval_configs/llava-1.5_eval.yaml`.


## Visualization 🔥🔥🔥

```
python demo_smooth_grad_threshold.py
```
![image](https://github.com/zhangbaijin/From-Redundancy-to-Relevance/blob/main/horse.png)


## Citation
```bibtex
@article{zhang2024redundancy,
  title={From Redundancy to Relevance: Enhancing Explainability in Multimodal Large Language Models},
  author={Zhang, Xiaofeng and  Quan, Yihao and Shen, Chen and Yuan, Xiaosong and Yan, Shaotian and Xie, Liang and Wang, Wenxiao and Gu, Chaochen and Tang, Hao and Ye, Jieping},
  journal={Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics},
  year={2025}
}
```
## Acknowledgement

This repo is built on [LLaVA](https://github.com/haotian-liu/LLaVA) (models), [OPERA](https://github.com/shikiw/OPERA) (CHAIR evaluation) and [FastV](https://github.com/pkunlp-icler/FastV) (Image Token Truncation). Many thanks for their efforts. The use of our code should also follow the original licenses.

