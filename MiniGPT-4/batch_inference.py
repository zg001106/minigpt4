#!/usr/bin/env python3
"""
MiniGPT-4 批量图片推理脚本

使用方法:
1. 修改下面的配置参数
2. 确保您的图片文件夹路径正确
3. 运行脚本: python batch_inference.py

输出:
- 每张原图片在上方
- MiniGPT-4的回答文本在下方
- 保存为新的图片文件
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont
import cv2

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# ===== 配置参数 =====
# 请根据您的实际情况修改以下参数

# 输入图片文件夹路径（包含要处理的图片）
INPUT_DIR = "examples"

# 输出结果文件夹路径（处理后的图片将保存在这里）
OUTPUT_DIR = "output_results"

# 提示词（所有图片都会使用这个提示词）
PROMPT = "Please describe this image in detail."

# 配置文件路径（MiniGPT-4的配置文件）
CFG_PATH = "eval_configs/minigpt4_eval.yaml"

# GPU ID（如果有多个GPU，可以选择使用哪个）
GPU_ID = 0

# 字体大小（用于在图片下方显示文本）
FONT_SIZE = 16

# ===== 脚本代码 =====

def setup_seeds(seed=42):
    """设置随机种子，确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def init_model(cfg_path, gpu_id=0):
    """初始化MiniGPT-4模型"""
    # 模拟命令行参数
    class Args:
        def __init__(self):
            self.cfg_path = cfg_path
            self.gpu_id = gpu_id
            self.options = None
    
    args = Args()
    cfg = Config(args)
    
    # 对话模版字典
    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                 'pretrain_llama2': CONV_VISION_LLama2}
    
    print('🚀 正在初始化MiniGPT-4模型...')
    
    # 模型配置
    model_config = cfg.model_cfg
    model_config.device_8bit = gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{gpu_id}')
    
    # 对话模版
    CONV_VISION = conv_dict[model_config.model_type]
    
    # 视觉处理器
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    # 聊天对象
    chat = Chat(model, vis_processor, device=f'cuda:{gpu_id}')
    print('✅ 模型初始化完成!')
    
    return chat, CONV_VISION


def process_single_image(image_path, prompt, chat, conv_vision):
    """处理单张图片并获取回答"""
    try:
        # 创建对话状态
        chat_state = conv_vision.copy()
        img_list = []
        
        # 加载并上传图片
        image = Image.open(image_path).convert('RGB')
        chat.upload_img(image, chat_state, img_list)
        chat.encode_img(img_list)
        
        # 提问
        chat.ask(prompt, chat_state)
        
        # 获取回答
        answer = chat.answer(conv=chat_state,
                            img_list=img_list,
                            num_beams=1,
                            temperature=1.0,
                            max_new_tokens=300,
                            max_length=2000)[0]
        
        return answer, image
        
    except Exception as e:
        print(f"❌ 处理图片 {image_path} 时出错: {str(e)}")
        return None, None


def create_combined_image(original_image, answer_text, font_size=20):
    """将回答文本添加到图片下方"""
    # 转换为PIL图像
    if isinstance(original_image, np.ndarray):
        original_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    
    # 图片尺寸
    img_width, img_height = original_image.size
    
    # 设置字体（尝试使用系统默认字体）
    try:
        # Windows系统字体
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            # 备用字体
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
        except:
            # 默认字体
            font = ImageFont.load_default()
    
    # 计算文本区域高度
    text_lines = []
    max_width = img_width - 20  # 留边距
    words = answer_text.split(' ')
    current_line = ""
    
    # 将文本分行
    for word in words:
        test_line = current_line + word + " "
        try:
            bbox = font.getbbox(test_line)
            text_width = bbox[2] - bbox[0]
        except:
            # 对于旧版本PIL，使用textsize
            text_width = font.getsize(test_line)[0]
        
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                text_lines.append(current_line.strip())
                current_line = word + " "
            else:
                text_lines.append(word)
                current_line = ""
    
    if current_line:
        text_lines.append(current_line.strip())
    
    # 计算文本总高度
    line_height = font_size + 5
    text_height = len(text_lines) * line_height + 20  # 额外边距
    
    # 创建新图片（原图+文本区域）
    new_height = img_height + text_height
    combined_image = Image.new('RGB', (img_width, new_height), 'white')
    
    # 粘贴原图
    combined_image.paste(original_image, (0, 0))
    
    # 绘制文本
    draw = ImageDraw.Draw(combined_image)
    y_offset = img_height + 10
    
    for line in text_lines:
        draw.text((10, y_offset), line, fill='black', font=font)
        y_offset += line_height
    
    return combined_image


def batch_process_images(input_dir, output_dir, prompt, cfg_path, gpu_id=0, font_size=16):
    """批量处理图片文件夹"""
    print("=" * 60)
    print("🎯 MiniGPT-4 批量图片推理")
    print("=" * 60)
    print(f"📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"💬 提示词: {prompt}")
    print(f"⚙️ 配置文件: {cfg_path}")
    print(f"🔧 GPU ID: {gpu_id}")
    print("=" * 60)
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"❌ 错误: 输入目录不存在: {input_dir}")
        return
    
    # 设置随机种子
    setup_seeds()
    
    # 初始化模型
    chat, conv_vision = init_model(cfg_path, gpu_id)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 输出目录已创建: {output_dir}")
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # 获取所有图片文件
    image_files = []
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print(f"❌ 错误: 在 {input_dir} 中没有找到图片文件")
        return
    
    print(f"🖼️ 找到 {len(image_files)} 张图片")
    print("=" * 60)
    
    # 处理每张图片
    success_count = 0
    for i, image_file in enumerate(image_files, 1):
        print(f"🔄 正在处理 ({i}/{len(image_files)}): {image_file}")
        
        image_path = os.path.join(input_dir, image_file)
        
        # 获取模型回答
        answer, original_image = process_single_image(image_path, prompt, chat, conv_vision)
        
        if answer is not None and original_image is not None:
            # 创建组合图片
            combined_image = create_combined_image(original_image, answer, font_size)
            
            # 保存结果
            output_filename = f"result_{os.path.splitext(image_file)[0]}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            combined_image.save(output_path, 'JPEG', quality=95)
            
            print(f"  ✅ 保存到: {output_path}")
            print(f"  💬 回答: {answer[:80]}...")  # 显示前80个字符
            success_count += 1
        else:
            print(f"  ❌ 处理失败")
        
        print()  # 空行分隔
    
    print("=" * 60)
    print(f"🎉 批量处理完成! 成功处理 {success_count}/{len(image_files)} 张图片")
    print(f"📁 结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # 检查配置
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误: 输入目录不存在: {INPUT_DIR}")
        print("请修改脚本顶部的 INPUT_DIR 变量，指向您的图片文件夹")
        exit(1)
    
    if not os.path.exists(CFG_PATH):
        print(f"❌ 错误: 配置文件不存在: {CFG_PATH}")
        print("请确保 MiniGPT-4 的配置文件存在")
        exit(1)
    
    # 运行批量处理
    batch_process_images(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR, 
        prompt=PROMPT,
        cfg_path=CFG_PATH,
        gpu_id=GPU_ID,
        font_size=FONT_SIZE
    )
