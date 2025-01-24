# [NAACL 2025] From redundancy to relevance: Enhancing explainability in multimodal large language models

[Paper](https://arxiv.org/abs/2406.06579)

![image](https://github.com/zhangbaijin/From-Redundancy-to-Relevance/blob/main/horse.png)


## Setup

The main implementation of Our is in `transformers-4.29.2/src/transformers/generation/utils.py`.

So it is convenient to use Our decoding by just installing our modified `transformers` package.
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


#### OPERA modify
 Note: to implement on other version of transformers, you can follow the steps as the follows:
- Find the file at `transformers-4.29.2/src/transformers/generation/utils.py`.
- Add the arguments in `transformers.generate` function [here](https://github.com/shikiw/OPERA/blob/aa968c7501f4d3d8362f4b3bcab855024f4da5f6/transformers-4.29.2/src/transformers/generation/utils.py#L1156-L1162).
- Add the code in `transformers.generate` function [here](https://github.com/shikiw/OPERA/blob/aa968c7501f4d3d8362f4b3bcab855024f4da5f6/transformers-4.29.2/src/transformers/generation/utils.py#L1619-L1665).
- Copy and paste the `opera_decoding` function [here](https://github.com/shikiw/OPERA/blob/aa968c7501f4d3d8362f4b3bcab855024f4da5f6/transformers-4.29.2/src/transformers/generation/utils.py#L3116-L3674).

## Evaluation

The following evaluation requires for MSCOCO 2014 dataset. Please download [here](https://cocodataset.org/#home) and extract it in your data path.

Besides, it needs you to prepare the following checkpoints of 7B base models:

- Download [LLaVA-1.5 merged 7B model](https://huggingface.co/liuhaotian/llava-v1.5-7b) and specify it at [Line 14](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/eval_configs/llava-1.5_eval.yaml#L14) of `eval_configs/llava-1.5_eval.yaml`.


## Citation
``````bibtex
@article{zhang2024redundancy,
  title={From Redundancy to Relevance: Enhancing Explainability in Multimodal Large Language Models},
  author={Zhang, Xiaofeng and  Quan, Yihao and Shen, Chen and Yuan, Xiaosong and Yan, Shaotian and Xie, Liang and Wang, Wenxiao and Gu, Chaochen and Tang, Hao and Ye, Jieping},
  journal={Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics},
  year={2025}
}



