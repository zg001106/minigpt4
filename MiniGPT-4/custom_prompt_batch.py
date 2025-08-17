#!/usr/bin/env python3
"""
MiniGPT-4 自定义提示词批量处理脚本

支持功能:
1. 自定义提示词
2. 批量处理多个提示词
3. 为每个提示词创建单独的输出文件夹
4. 交互式提示词选择
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

# ===== 自定义提示词库 =====
PROMPT_CATEGORIES = {
    "基础描述": [
        "Please describe this image in detail.",
        "What do you see in this image?",
        "Describe the main objects and activities in this picture.",
        "Give me a comprehensive description of this image.",
    ],
    
    "中文提示": [
        "请详细描述这张图片。",
        "这张图片中有什么？",
        "描述一下图片中的主要内容。",
        "请分析这张图片的内容和含义。",
    ],
    
    "物体识别": [
        "What objects can you see in this image?",
        "List all the items visible in this picture.",
        "Identify and describe the main objects in this image.",
        "What are the most prominent features in this image?",
    ],
    
    "人物分析": [
        "Describe the people in this image.",
        "What are the people doing in this picture?",
        "How many people are in this image and what are they wearing?",
        "Describe the emotions and expressions of people in this image.",
    ],
    
    "场景分析": [
        "Describe the setting and environment of this image.",
        "What kind of location is this?",
        "What is the weather or lighting condition in this image?",
        "Describe the background and surroundings.",
    ],
    
    "活动分析": [
        "What is the main activity happening in this image?",
        "What story does this image tell?",
        "What events are taking place in this picture?",
        "Describe the action or movement in this image.",
    ],
    
    "情感分析": [
        "What emotions or mood does this image convey?",
        "What feelings does this image evoke?",
        "Describe the atmosphere of this image.",
        "What is the emotional tone of this picture?",
    ],
    
    "创意分析": [
        "What is unusual or interesting about this image?",
        "If you were to give this image a title, what would it be?",
        "What makes this image unique or special?",
        "Tell me something creative about this image.",
    ],
    
    "技术分析": [
        "Describe the composition and visual elements of this image.",
        "What colors dominate this image?",
        "Describe the lighting and shadows in this image.",
        "What is the perspective or viewpoint of this image?",
    ]
}

# 您的自定义提示词 - 在这里添加您想要的问题
MY_CUSTOM_PROMPTS = [
    # 在这里添加您的自定义提示词
    "Your custom question 1 here",
    "Your custom question 2 here",
    # 可以添加更多...
]

def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def init_model(cfg_path, gpu_id=0):
    """初始化MiniGPT-4模型"""
    class Args:
        def __init__(self):
            self.cfg_path = cfg_path
            self.gpu_id = gpu_id
            self.options = None
    
    args = Args()
    cfg = Config(args)
    
    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                 'pretrain_llama2': CONV_VISION_LLama2}
    
    print('🚀 正在初始化MiniGPT-4模型...')
    
    model_config = cfg.model_cfg
    model_config.device_8bit = gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{gpu_id}')
    
    CONV_VISION = conv_dict[model_config.model_type]
    
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    chat = Chat(model, vis_processor, device=f'cuda:{gpu_id}')
    print('✅ 模型初始化完成!')
    
    return chat, CONV_VISION

def process_single_image(image_path, prompt, chat, conv_vision):
    """处理单张图片并获取回答"""
    try:
        chat_state = conv_vision.copy()
        img_list = []
        
        image = Image.open(image_path).convert('RGB')
        chat.upload_img(image, chat_state, img_list)
        chat.encode_img(img_list)
        
        chat.ask(prompt, chat_state)
        
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

def create_combined_image(original_image, answer_text, font_size=16):
    """将回答文本添加到图片下方"""
    if isinstance(original_image, np.ndarray):
        original_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    
    img_width, img_height = original_image.size
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    text_lines = []
    max_width = img_width - 20
    words = answer_text.split(' ')
    current_line = ""
    
    for word in words:
        test_line = current_line + word + " "
        try:
            bbox = font.getbbox(test_line)
            text_width = bbox[2] - bbox[0]
        except:
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
    
    line_height = font_size + 5
    text_height = len(text_lines) * line_height + 20
    
    new_height = img_height + text_height
    combined_image = Image.new('RGB', (img_width, new_height), 'white')
    
    combined_image.paste(original_image, (0, 0))
    
    draw = ImageDraw.Draw(combined_image)
    y_offset = img_height + 10
    
    for line in text_lines:
        draw.text((10, y_offset), line, fill='black', font=font)
        y_offset += line_height
    
    return combined_image

def show_prompt_menu():
    """显示提示词菜单"""
    print("\n" + "=" * 60)
    print("📝 可用的提示词类别")
    print("=" * 60)
    
    all_prompts = []
    category_index = 0
    
    for category, prompts in PROMPT_CATEGORIES.items():
        print(f"\n📂 {category}:")
        for i, prompt in enumerate(prompts):
            print(f"  {len(all_prompts):2d}. {prompt}")
            all_prompts.append(prompt)
    
    if MY_CUSTOM_PROMPTS and MY_CUSTOM_PROMPTS[0] != "Your custom question 1 here":
        print(f"\n📂 我的自定义提示词:")
        for prompt in MY_CUSTOM_PROMPTS:
            print(f"  {len(all_prompts):2d}. {prompt}")
            all_prompts.append(prompt)
    
    return all_prompts

def select_prompts():
    """交互式选择提示词"""
    all_prompts = show_prompt_menu()
    
    print("\n" + "=" * 60)
    print("🎯 选择提示词")
    print("=" * 60)
    print("选择方式:")
    print("1. 输入单个数字 (如: 5)")
    print("2. 输入多个数字，用逗号分隔 (如: 1,3,5)")
    print("3. 输入范围 (如: 1-5)")
    print("4. 输入 'all' 使用所有提示词")
    print("5. 输入 'custom' 直接输入自定义提示词")
    
    user_input = input("\n请输入您的选择: ").strip()
    
    if user_input.lower() == 'all':
        return all_prompts
    elif user_input.lower() == 'custom':
        custom_prompt = input("请输入您的自定义提示词: ").strip()
        return [custom_prompt] if custom_prompt else []
    elif '-' in user_input:
        try:
            start, end = map(int, user_input.split('-'))
            return [all_prompts[i] for i in range(start, min(end+1, len(all_prompts))) if 0 <= i < len(all_prompts)]
        except:
            print("❌ 范围格式错误")
            return []
    elif ',' in user_input:
        try:
            indices = [int(x.strip()) for x in user_input.split(',')]
            return [all_prompts[i] for i in indices if 0 <= i < len(all_prompts)]
        except:
            print("❌ 数字格式错误")
            return []
    else:
        try:
            index = int(user_input)
            if 0 <= index < len(all_prompts):
                return [all_prompts[index]]
            else:
                print("❌ 索引超出范围")
                return []
        except:
            print("❌ 输入格式错误")
            return []

def batch_process_with_prompts(input_dir, output_base_dir, prompts, cfg_path, gpu_id=0):
    """使用多个提示词批量处理"""
    if not prompts:
        print("❌ 没有选择任何提示词")
        return
    
    setup_seeds()
    chat, conv_vision = init_model(cfg_path, gpu_id)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in os.listdir(input_dir) 
                   if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not image_files:
        print(f"❌ 在 {input_dir} 中没有找到图片文件")
        return
    
    print(f"\n🖼️ 找到 {len(image_files)} 张图片")
    print(f"💬 将使用 {len(prompts)} 个提示词处理")
    
    for prompt_idx, prompt in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"🔄 处理提示词 {prompt_idx}/{len(prompts)}")
        print(f"💬 提示词: {prompt}")
        print(f"{'='*60}")
        
        # 为每个提示词创建单独的输出目录
        safe_prompt = "".join(c if c.isalnum() or c in '-_' else '_' for c in prompt[:50])
        output_dir = os.path.join(output_base_dir, f"prompt_{prompt_idx:02d}_{safe_prompt}")
        os.makedirs(output_dir, exist_ok=True)
        
        success_count = 0
        for img_idx, image_file in enumerate(image_files, 1):
            print(f"  📸 ({img_idx}/{len(image_files)}) {image_file}")
            
            image_path = os.path.join(input_dir, image_file)
            answer, original_image = process_single_image(image_path, prompt, chat, conv_vision)
            
            if answer and original_image:
                combined_image = create_combined_image(original_image, answer)
                output_filename = f"result_{os.path.splitext(image_file)[0]}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                combined_image.save(output_path, 'JPEG', quality=95)
                
                print(f"    ✅ 回答: {answer[:60]}...")
                success_count += 1
            else:
                print(f"    ❌ 处理失败")
        
        print(f"\n📊 提示词 {prompt_idx} 完成: {success_count}/{len(image_files)} 张图片成功处理")
        print(f"📁 结果保存在: {output_dir}")

def main():
    """主函数"""
    # 基础配置
    INPUT_DIR = "examples"  # 修改为您的图片文件夹路径
    OUTPUT_BASE_DIR = "custom_prompt_results"  # 输出基础目录
    CFG_PATH = "eval_configs/minigpt4_eval.yaml"
    GPU_ID = 0
    
    print("🎯 MiniGPT-4 自定义提示词批量处理")
    print("=" * 60)
    print(f"📁 输入目录: {INPUT_DIR}")
    print(f"📁 输出基础目录: {OUTPUT_BASE_DIR}")
    
    # 检查输入目录
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误: 输入目录不存在: {INPUT_DIR}")
        print("请修改脚本中的 INPUT_DIR 变量")
        return
    
    # 选择提示词
    selected_prompts = select_prompts()
    
    if not selected_prompts:
        print("❌ 没有选择有效的提示词，程序退出")
        return
    
    print(f"\n✅ 已选择 {len(selected_prompts)} 个提示词:")
    for i, prompt in enumerate(selected_prompts, 1):
        print(f"  {i}. {prompt}")
    
    # 确认处理
    confirm = input(f"\n继续处理 {len(selected_prompts)} 个提示词？(y/N): ").strip().lower()
    if confirm != 'y':
        print("已取消处理")
        return
    
    # 开始批量处理
    batch_process_with_prompts(INPUT_DIR, OUTPUT_BASE_DIR, selected_prompts, CFG_PATH, GPU_ID)
    
    print(f"\n🎉 所有处理完成！")
    print(f"📁 结果保存在: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()
