# OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation (CVPR 2024 Highlight)

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arXiv-2311.17911-B21A1B)](https://arxiv.org/pdf/2311.17911.pdf)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-blue)](https://github.com/huggingface/transformers)
[![GitHub Stars](https://img.shields.io/github/stars/shikiw/OPERA?style=social)](https://github.com/shikiw/OPERA/stargazers)


This repository provides the official PyTorch implementation of the following paper: 
> [**OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation**](https://arxiv.org/pdf/2311.17911.pdf) <br>
> [Qidong Huang](https://shikiw.github.io/)<sup>1,2</sup>, 
> [Xiaoyi Dong](https://scholar.google.com/citations?user=FscToE0AAAAJ&hl=en)<sup>2</sup>, 
> [Pan Zhang](https://panzhang0212.github.io/)<sup>2</sup>,
> [Bin Wang](https://wangbindl.github.io/) <sup>2</sup>,
> [Conghui He](https://conghui.github.io/) <sup>2</sup>, 
> [Jiaqi Wang](https://myownskyw7.github.io/)<sup>2</sup>,
> [Dahua Lin](http://dahua.site/)<sup>2</sup>, 
> [Weiming Zhang](http://staff.ustc.edu.cn/~zhangwm/index.html)<sup>1</sup>, 
> [Nenghai Yu](https://scholar.google.com/citations?user=7620QAMAAAAJ&hl=en)<sup>1</sup> <br>
> <sup>1</sup>University of Science and Technology of China, <sup>2</sup>Shanghai AI Laboratory <br>


## Overview

<p align="center"><img src="./teaser.png" alt="teaser" width="500px" /></p>

Hallucination, posed as a pervasive challenge of multimodal large language models (MLLMs), has significantly impeded their real-world usage that demands precise judgment. Existing methods mitigate this issue with either training with specific designed data or inferencing with external knowledge from other sources, incurring inevitable additional costs. In this paper, we present OPERA, a novel MLLM decoding method grounded in an Over-trust Penalty and a Retrospection-Allocation strategy, serving as a nearly free lunch to alleviate the hallucination issue without additional data, knowledge, or training. Our approach begins with an interesting observation that, most hallucinations are closely tied to the knowledge aggregation patterns manifested in the self-attention matrix, i.e., MLLMs tend to generate new tokens by focusing on a few summary tokens, but not all the previous tokens. Such partial overtrust inclination results in the neglecting of image tokens and describes the image content with hallucination. Based on the observation, OPERA introduces a penalty term on
the model logits during the beam-search decoding to mitigate the over-trust issue, along with a rollback strategy that retrospects the presence of summary tokens in the previously generated tokens, and re-allocate the token selection if necessary. With extensive experiments, OPERA shows significant hallucination-mitigating performance on different MLLMs and metrics, proving its effectiveness and generality.

## Setup

The main implementation of OPERA is in `transformers-4.29.2/src/transformers/generation/utils.py`.

So it is convenient to use OPERA decoding by just installing our modified `transformers` package.
```
conda env create -f environment.yml
conda activate opera
python -m pip install -e transformers-4.29.2
```
#### Note: to implement OPERA on other version of transformers, you can follow the steps as the follows:
- Find the file at `transformers-4.29.2/src/transformers/generation/utils.py`.
- Add the arguments in `transformers.generate` function [here](https://github.com/shikiw/OPERA/blob/aa968c7501f4d3d8362f4b3bcab855024f4da5f6/transformers-4.29.2/src/transformers/generation/utils.py#L1156-L1162).
- Add the code in `transformers.generate` function [here](https://github.com/shikiw/OPERA/blob/aa968c7501f4d3d8362f4b3bcab855024f4da5f6/transformers-4.29.2/src/transformers/generation/utils.py#L1619-L1665).
- Copy and paste the `opera_decoding` function [here](https://github.com/shikiw/OPERA/blob/aa968c7501f4d3d8362f4b3bcab855024f4da5f6/transformers-4.29.2/src/transformers/generation/utils.py#L3116-L3674).

## TL;DR
After setup the environment, you can directly use OPERA on your own MLLM model by:
```
# specify the location indexes of some input tokens
START_INDEX_of_IMAGE_TOKENS = <the location index of the first image token>
END_INDEX_of_IMAGE_TOKENS = <the location index of the last image token>
NUM_of_TOKENS_IN_THE_PROMPT = <the total number of tokens in the user prompt (including image tokens)>

key_position = {
  "image_start": START_INDEX_of_IMAGE_TOKENS, 
  "image_end": END_INDEX_of_IMAGE_TOKENS, 
  "response_start": NUM_of_TOKENS_IN_THE_PROMPT,
}

# add some arguments in the generate function
outputs = MLLM_model.generate(
    input_ids=input_ids,
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    do_sample=False,
    num_beams=5,
    max_new_tokens=512,
    # opera
    opera_decoding=True,
    key_position=key_position,
    scale_factor=50,
    threshold=15,
    num_attn_candidates=5,
    penalty_weights=1,
)
# for a more efficient version, please use the setting below:
outputs = MLLM_model.generate(
    input_ids=input_ids,
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    do_sample=False,
    num_beams=5,
    max_new_tokens=512,
    # opera
    opera_decoding=True,
    key_position=key_position,
    scale_factor=50,
    threshold=25,
    num_attn_candidates=1,
    penalty_weights=1,
)
```

Please refer to `demo.ipynb` [here](https://github.com/shikiw/OPERA/blob/1e74d8b5d082579c81e0e77ef1cf4a44d20ab91e/demo.ipynb) for more details.


## Evaluation

The following evaluation requires for MSCOCO 2014 dataset. Please download [here](https://cocodataset.org/#home) and extract it in your data path.

Besides, it needs you to prepare the following checkpoints of 7B base models:

- Download [LLaVA-1.5 merged 7B model](https://huggingface.co/liuhaotian/llava-v1.5-7b) and specify it at [Line 14](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/eval_configs/llava-1.5_eval.yaml#L14) of `eval_configs/llava-1.5_eval.yaml`.
- Download [Vicuna 7B v1.1 model](https://github.com/lm-sys/FastChat) and specify it at [Line 25](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/minigpt4/configs/models/blip2_instruct_vicuna7b.yaml#L25) of `minigpt4/configs/models/blip2_instruct_vicuna7b.yaml`.
- Download [Vicuna 7B v0 model](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) and specify it at [Line 18](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/minigpt4/configs/models/minigpt4_vicuna0.yaml#L18) of `minigpt4/configs/models/minigpt4_vicuna0.yaml`.
- Download [MiniGPT-4 7B pretrained weights](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) and specify it at [Line 8](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/eval_configs/minigpt4_eval.yaml#L8) of `eval_configs/minigpt4_eval.yaml`.
- Download [Shikra merged 7B model](https://github.com/shikras/shikra#checkpoint) and specify it at [Line 14](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/eval_configs/shikra_eval.yaml#L14) of `eval_configs/shikra_eval.yaml`.

### Arguments

| Argument             | Example             | Description   |
| -------------------- | ------------------- | ------------- |
| `--model`    | `llava-1.5` | Specify the MLLM model, this codebase supports `instructblip`, `minigpt4`, `llava-1.5`, `shikra`. |
| `--data-path`     | `/path/to/dataset` | Path to the dataset file or folder, e.g., `COCO_2014/val2014/`. |
| `--pope-type`     | `random` | Type for POPE evaluation, supports `random`, `popular`, `adversarial`. |
| `--scale_factor`   | `50` | The scale factor to scale up the self-attention weights. Default: 50. |
| `--threshold`      | `15` | The threshold for attending retrospection. Default: 15. |
| `--num_attn_candidates`   | `5` | The number of candidates per beam. Default: 5. |
| `--penalty_weights`| `1` | The weight of penalty term in decoding. Default: 1.  |

### POPE
```bash
python pope_eval.py --model MODEL_NAME --data_path /path/to/COCO --pope-type random --gpu-id GPU_IDs --beam 5 --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1
```
Result on `Random` split:

| Model | Accuracy | Precision | Recall | F1 score| Yes ratio |
| ----- | -------- | --------- | ------ | ------- | --------- |
| InstructBLIP 7B | 90.3 | 93.8 | 87.0 | 90.3 | 47.8 |
| MiniGPT-4 7B | 79.8 | 89.7 | 68.7 | 77.8 | 39.5 |
| LLaVA-1.5 7B | 89.4 | 90.4 | 88.8 | 89.6 | 50.6 |

Result on `Popular` split:

| Model | Accuracy | Precision | Recall | F1 score| Yes ratio |
| ----- | -------- | --------- | ------ | ------- | --------- |
| InstructBLIP 7B | 83.4 | 81.2 | 87.0 | 84.0 | 53.6 |
| MiniGPT-4 7B | 73.6 | 75.9 | 69.0 | 72.3 | 45.4 |
| LLaVA-1.5 7B | 86.0 | 84.1 | 88.8 | 86.4 | 52.8 |

Result on `Adversarial` split:

| Model | Accuracy | Precision | Recall | F1 score| Yes ratio |
| ----- | -------- | --------- | ------ | ------- | --------- |
| InstructBLIP 7B | 80.7 | 77.3 | 87.0 | 81.9 | 56.3 |
| MiniGPT-4 7B | 71.6 | 72.9 | 68.9 | 70.8 | 47.3 |
| LLaVA-1.5 7B | 79.1 | 74.4 | 88.8 | 81.0 | 59.7 |

### CHAIR
- Generate the MLLM's responses and save them in a jsonl file:
```bash
python chair_eval.py --model MODEL_NAME --data_path /path/to/COCO --gpu-id GPU_IDs --beam 5 --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1
```
Note: Please check out our released results in `log/chair_eval_results` for reproduction.

- Calculate CHAIR using the generated jsonl file:
```bash
python chair.py --cap_file /path/to/jsonl --image_id_key image_id --caption_key caption --coco_path /path/to/COCO/annotations_trainval2014/annotations/ --save_path /path/to/save/jsonl
```

### GPT-4V
The GPT-4V evaluation requires you to specify your API key in [Line 88](https://github.com/shikiw/OPERA/blob/559556048224d5c3eae995a21d529156fb150d5f/gpt4v_eval.py#L88) of `gpt4v_eval.py`.
```bash
python gpt4v_eval.py --model MODEL_NAME --data_path /path/to/COCO --gpu-id GPU_IDs --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1
```




## Acknowledgement
This repo is based on the MLLM codebase of [LAVIS](https://github.com/salesforce/LAVIS) and [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) and the CHAIR code of [Maxlinn](https://github.com/Maxlinn/CHAIR-metric-standalone). Thanks for their impressive works!

## Citation
If you find this work useful for your research, please cite [our paper](https://arxiv.org/pdf/2311.17911.pdf):
```
@inproceedings{huang2024opera,
  title={Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation},
  author={Huang, Qidong and Dong, Xiaoyi and Zhang, Pan and Wang, Bin and He, Conghui and Wang, Jiaqi and Lin, Dahua and Zhang, Weiming and Yu, Nenghai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13418--13427},
  year={2024}
}
```


