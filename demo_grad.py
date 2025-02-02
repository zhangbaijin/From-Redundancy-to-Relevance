import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from pope_loader import POPEDataSet
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image
from torchvision.utils import save_image
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

import torch
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from einops import rearrange


class GradCAM():
    def __init__(self, model, target_layer,input_token_len, output_ids):
        self.model = model  # 要进行Grad-CAM处理的模型
        self.target_layer = target_layer  # 要进行特征可视化的目标层
        self.feature_maps = None  # 存储特征图
        self.gradients = None  # 存储梯度

        self.input_token_len = input_token_len
        self.output_ids = output_ids
        
        self.target_ids = self.output_ids[0][self.input_token_len:]
        # 为目标层添加钩子，以保存输出和梯度
        target_layer.register_forward_hook(self.save_feature_maps)
        target_layer.register_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        """保存特征图"""
        # output.requires_grad = True
        self.feature_maps = output
        output.retain_grad()
        # self.feature_maps.requires_grad = True
        num_token = self.feature_maps.shape[1]
        h = int(np.sqrt(num_token))
        # self.feature_maps = self.feature_maps[0,1:,:].reshape((1,h,h,-1))

    def save_gradients(self, module, grad_input, grad_output):
        """保存梯度"""
        self.gradients = grad_output[0].detach()
        print(self.gradients)

    def generate_cam(self, image, qu):
        """生成CAM热力图"""
        # 将模型设置为评估模式
        # self.model.eval()
        # 清空所有梯度
        # self.model.zero_grad()
        # 正向传播
        out = model.generate_output(
            {"image": norm(image), "prompt":qu},  # 输入字典包含更新后的提问
            num_beams=1,  # 使用 beam size 为 1
            max_new_tokens=128,  # 生成的最大新 token 数量
            output_attentions=True,  # 请求返回注意力权重
            # opera_decoding=True,
            # scale_factor=50,
            # threshold=15.0,
            # num_attn_candidates=5,
        )
        # 对目标类进行反向传播

        # target_logits = torch.sum(out.logits[0][self.input_token_len:self.output_ids.shape[1],:][torch.arange(len(self.target_ids)),self.target_ids.int()])
        target_logits = torch.sum(out.logits[0][self.input_token_len:self.output_ids.shape[1],:][torch.arange(len(self.target_ids)),self.target_ids.int()])

        target_logits.retain_grad()
        target_logits.backward(retain_graph=True)

        num_token = self.feature_maps.shape[1]
        h = int(np.sqrt(num_token))
        # self.feature_maps = self.feature_maps[0:,1:,:].detach().reshape((1,h,h,-1))
        self.feature_maps = rearrange(self.feature_maps[0:,34:610,:].detach(),'b (h w) c -> b c h w ',w=24,h=24)
        # 获取平均梯度和特征图
        # self.gradients = self.gradients[0:,1:,:].detach().reshape((1,h,h,-1))
        self.gradients = rearrange(self.gradients[0:,34:610,:].detach(),'b (h w) c -> b h w c',w=24,h=24)
        self.gradients = nn.ReLU()(self.gradients)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 1,2])
        activation = self.feature_maps.squeeze(0)
        for i in range(activation.size(0)):
            activation[i, :, :] *= pooled_gradients[i]

        # 创建热力图
        # activation = activation.permute(0,2,1)
        heatmap = torch.mean(activation, dim=0).squeeze().cpu().numpy().astype(np.float32)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = cv2.resize(heatmap, (image.size(3), image.size(2)))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 将热力图叠加到原始图像上
        original_image = self.unprocess_image(image.squeeze().cpu().numpy())
        superimposed_img = heatmap * 0.4 + original_image
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        return heatmap, superimposed_img

    def unprocess_image(self, image):
        """反预处理图像，将其转回原始图像"""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (((image.transpose(1, 2, 0) * std) + mean) * 255).astype(np.uint8)
        return image


MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

POPE_PATH = {
    "random": "coco_pope/coco_pope_random.json",
    "popular": "coco_pope/coco_pope_popular.json",
    "adversarial": "coco_pope/coco_pope_adversarial.json",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--model", type=str, help="model")
parser.add_argument("--gpu-id", type=int, help="specify the gpu to load the model.")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument("--data_path", type=str, default="/mnt/petrelfs/share_data/wangjiaqi/mllm-data-alg/COCO_2014/ori/val2014/val2014/", help="data path")
parser.add_argument("--batch_size", type=int, help="batch size")
parser.add_argument("--num_workers", type=int, default=2, help="num workers")
args = parser.parse_known_args()[0]


args.model = "llava-1.5"
# args.model = "instructblip"
# args.model = "minigpt4"
# args.model = "shikra"
args.gpu_id = "0"
#args.gpu_id_2 = 1
args.batch_size = 1


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)
setup_seeds(cfg)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


# ========================================
#             Model Initialization
# ========================================
print('Initializing Model')

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()
processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)
print(vis_processors["eval"].transform)
print("Done!")

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)

import re
import seaborn as sns
image_path = "/root/autodl-tmp/SQA/test/23/image.png"
# image_path = "/root/autodl-tmp/OPERA/image/95.jpg"
# image_path = '/root/autodl-tmp/OPERA/shiwulian.jpg'
# image_path = "/root/autodl-tmp/OPERA/sqa_images/test/170/image.png"
raw_image = Image.open(image_path).resize((336,336))
plt.imshow(raw_image)
plt.show()
raw_image = raw_image.convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0)
image = image.to(device)

# question = [{"content":"<image>\nQuestion:Please describe the image."}]
question = [{"content":"<image>\nContext: Below is a food web from a tundra ecosystem in Nunavut, a territory in Northern Canada.\nA food web models how the matter eaten by organisms moves through an ecosystem. The arrows in a food web represent how matter moves between organisms in an ecosystem.\nQuestion: Which of these organisms contains matter that was once part of the lichen?\nOptions: (A) bilberry (B) mushroom\nAnswer with the option's letter from the given choices directly."}]

# question = [{"content":"<image>\nContext: Below is a food web from a tundra ecosystem in Nunavut, a territory in Northern Canada.\nA food web models how the matter eaten by organisms moves through an ecosystem. The arrows in a food web represent how matter moves between organisms in an ecosystem.\nQuestion: where is the container"}]

# question = [{"content":"<image> Context:The image features four different sports-related objects, each with a unique crown-like decoration. The objects include a turtle, a basketball, a crown, and a tennis ball. The turtle is positioned on the left side of the image, while the basketball is located on the right side. The crown is placed in the middle of the image, and the tennis ball is situated at the bottom right corner. The arrangement of these objects creates a visually interesting and diverse scene. Select the best answer. Question: Which property do these three objects have in common? Options: (A) shiny (B) slippery (C) opaque Answer with the option's letter from the given choices directly."}]
# question = [{"content":"<image>\n Question:Is there any person in this image?"}]

prompt_str = '\n'.join([item['content'] for item in question])
template = INSTRUCTION_TEMPLATE[args.model]
# 现在 prompt_str 是一个字符串，可以用它来替换模板中的 <question> 占位符
qu = template.replace("<question>", prompt_str)


# with torch.inference_mode():
# model.eval()
out, input_token_len, output_ids = model.generate(
    {"image": norm(image), "prompt": qu},  # 传入图像和序列化的prompt
    use_nucleus_sampling=False,  # 关闭核采样（top-p采样
    num_beams=1,  # 使用beam size为1，意味着不进行束搜索
    max_new_tokens=128,  # 生成的最大新token数量
    output_attentions=True,  # 请求返回注意力权重
    # opera_decoding=True,
    # scale_factor=50,
    # threshold=15.0
    # num_attn_candidates=5,
)
# breakpoint()
# print(out[0])

for name, param in model.named_parameters():
    if param.grad is None:
        param.requires_grad=True

# 创建GradCAM
# model.llama_model.model.mm_projector
# model.llama_model.model.layers[0].post_attention_layernorm
for i in range (28,32):
    save_path = 'save_cam/23_grad'
    gradcam = GradCAM(model, model.llama_model.model.layers[i].post_attention_layernorm, input_token_len, output_ids)

    # gradcam = GradCAM(model, model.llama_model.model.mm_projector, input_token_len, output_ids)
    heatmap, result = gradcam.generate_cam(image, qu)
    path_cam_img=os.path.join(save_path,f"{image_path.split('/')[-2]}_layer_{i+1}.jpg")
    path_raw_img=os.path.join(save_path,f"{image_path.split('/')[-2]}.jpg")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(path_cam_img,result)

    qu_append = out[0]
    qu = qu + qu_append

# with torch.inference_mode():
#     with torch.no_grad():
#         out = model.generate_output(
#             {"image": norm(image), "prompt":qu},  # 输入字典包含更新后的提问
#             num_beams=1,  # 使用 beam size 为 1
#             max_new_tokens=1024,  # 生成的最大新 token 数量
#             output_attentions=True,  # 请求返回注意力权重
#             # opera_decoding=True,
#             # scale_factor=50,
#             # threshold=15.0,
#             # num_attn_candidates=5,
#         )
# print(out[0])
