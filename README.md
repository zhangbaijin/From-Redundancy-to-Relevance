# 🔥🔥🔥 [NAACL 2025] From redundancy to relevance: Enhancing explainability in multimodal large language models

[Paper](https://arxiv.org/abs/2406.06579)

![image](https://github.com/zhangbaijin/From-Redundancy-to-Relevance/blob/main/horse.png)


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


## 🔥🔥🔥 visualization

```
python demo_grad.py
```

## Citation
``````bibtex
@article{zhang2024redundancy,
  title={From Redundancy to Relevance: Enhancing Explainability in Multimodal Large Language Models},
  author={Zhang, Xiaofeng and  Quan, Yihao and Shen, Chen and Yuan, Xiaosong and Yan, Shaotian and Xie, Liang and Wang, Wenxiao and Gu, Chaochen and Tang, Hao and Ye, Jieping},
  journal={Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics},
  year={2025}
}



