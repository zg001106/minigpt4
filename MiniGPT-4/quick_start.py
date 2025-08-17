#!/usr/bin/env python3
"""
快速启动脚本 - MiniGPT-4 批量图片问答

只需要修改下面几行配置，然后运行即可！
"""

# ===== 只需要修改这几行配置 =====
INPUT_FOLDER = "examples"  # 您的图片文件夹路径
OUTPUT_FOLDER = "batch_results"  # 输出文件夹
QUESTION = "Please describe this image in detail."  # 想问的问题

# 高级选项（通常不需要修改）
GPU_ID = 0
CONFIG_FILE = "eval_configs/minigpt4_eval.yaml"
# =================================

import os
import sys
from PIL import Image, ImageDraw, ImageFont
import torch
import random
import numpy as np

# 检查输入文件夹是否存在
if not os.path.exists(INPUT_FOLDER):
    print(f"❌ 错误：找不到图片文件夹 '{INPUT_FOLDER}'")
    print("请修改脚本中的 INPUT_FOLDER 变量，指向您的图片文件夹")
    sys.exit(1)

# 导入 MiniGPT-4 模块
try:
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2
    from minigpt4.datasets.builders import *
    from minigpt4.models import *
    from minigpt4.processors import *
    from minigpt4.runners import *
    from minigpt4.tasks import *
except ImportError as e:
    print(f"❌ 错误：无法导入 MiniGPT-4 模块")
    print("请确保您在 MiniGPT-4 项目目录下运行此脚本")
    sys.exit(1)

def quick_setup():
    """快速设置和初始化"""
    print("🚀 正在启动 MiniGPT-4...")
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42) 
    torch.manual_seed(42)
    
    # 配置参数
    class Args:
        def __init__(self):
            self.cfg_path = CONFIG_FILE
            self.gpu_id = GPU_ID
            self.options = None
    
    args = Args()
    cfg = Config(args)
    
    # 初始化模型
    model_config = cfg.model_cfg
    model_config.device_8bit = GPU_ID
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{GPU_ID}')
    
    # 选择对话模板
    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0, 'pretrain_llama2': CONV_VISION_LLama2}
    CONV_VISION = conv_dict[model_config.model_type]
    
    # 视觉处理器
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    # 创建聊天对象
    chat = Chat(model, vis_processor, device=f'cuda:{GPU_ID}')
    
    print("✅ 初始化完成！")
    return chat, CONV_VISION

def ask_image(image_path, question, chat, conv_template):
    """向图片提问并获取回答"""
    try:
        # 创建新的对话
        chat_state = conv_template.copy()
        img_list = []
        
        # 加载图片
        image = Image.open(image_path).convert('RGB')
        
        # 上传图片到对话
        chat.upload_img(image, chat_state, img_list)
        chat.encode_img(img_list)
        
        # 提问
        chat.ask(question, chat_state)
        
        # 获取回答
        answer = chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=1,
            temperature=1.0,
            max_new_tokens=300,
            max_length=2000
        )[0]
        
        return answer, image
        
    except Exception as e:
        print(f"处理 {image_path} 时出错: {e}")
        return None, None

def combine_image_and_text(image, text, font_size=16):
    """将图片和文本组合"""
    img_width, img_height = image.size
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # 文本换行处理
    words = text.split(' ')
    lines = []
    current_line = ""
    max_width = img_width - 20
    
    for word in words:
        test_line = current_line + word + " "
        try:
            text_width = font.getsize(test_line)[0]
        except:
            text_width = len(test_line) * font_size * 0.6  # 估算宽度
        
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line.strip())
                current_line = word + " "
            else:
                lines.append(word)
    
    if current_line:
        lines.append(current_line.strip())
    
    # 计算总高度
    line_height = font_size + 5
    text_height = len(lines) * line_height + 20
    total_height = img_height + text_height
    
    # 创建新图片
    result_image = Image.new('RGB', (img_width, total_height), 'white')
    result_image.paste(image, (0, 0))
    
    # 绘制文本
    draw = ImageDraw.Draw(result_image)
    y = img_height + 10
    
    for line in lines:
        draw.text((10, y), line, fill='black', font=font)
        y += line_height
    
    return result_image

def main():
    """主函数"""
    print("=" * 50)
    print("🎯 MiniGPT-4 批量图片问答")
    print("=" * 50)
    print(f"📁 图片文件夹: {INPUT_FOLDER}")
    print(f"📁 输出文件夹: {OUTPUT_FOLDER}")
    print(f"❓ 问题: {QUESTION}")
    print("=" * 50)
    
    # 初始化
    chat, conv_template = quick_setup()
    
    # 创建输出文件夹
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 获取图片文件
    image_files = []
    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(file)
    
    if not image_files:
        print("❌ 没有找到图片文件！")
        return
    
    print(f"🖼️ 找到 {len(image_files)} 张图片")
    print()
    
    # 处理每张图片
    success = 0
    for i, filename in enumerate(image_files, 1):
        print(f"📸 ({i}/{len(image_files)}) {filename}")
        
        image_path = os.path.join(INPUT_FOLDER, filename)
        answer, image = ask_image(image_path, QUESTION, chat, conv_template)
        
        if answer and image:
            # 组合图片和回答
            result_image = combine_image_and_text(image, answer)
            
            # 保存结果
            output_name = f"result_{filename}"
            output_path = os.path.join(OUTPUT_FOLDER, output_name)
            result_image.save(output_path, 'JPEG', quality=95)
            
            print(f"✅ 回答: {answer[:60]}...")
            print(f"💾 保存: {output_path}")
            success += 1
        else:
            print("❌ 处理失败")
        
        print()
    
    print("=" * 50)
    print(f"🎉 完成！成功处理 {success}/{len(image_files)} 张图片")
    print(f"📁 结果保存在: {OUTPUT_FOLDER}")
    print("=" * 50)

if __name__ == "__main__":
    main()
