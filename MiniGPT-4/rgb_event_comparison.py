#!/usr/bin/env python3
"""
MiniGPT-4 RGB与Event Sensor图像对比分析脚本

功能:
- 对比RGB图像和Event sensor图像在不同提示词下的表现
- 生成3行2列的6宫格对比图片
- 第1行: RGB图片 | Event图片
- 第2行: RGB提示词 | Event提示词(固定)
- 第3行: RGB回答 | Event回答
- 为每个RGB提示词创建单独的输出文件夹用于对比分析

文件命名规则:
- RGB图片: 基础文件名.jpg (例如: interlaken_00_a_left-280.jpg)
- Event图片: 基础文件名+后缀.png (例如: interlaken_00_a_left-280-Accumulate_slow.png)
- 脚本会自动匹配RGB文件名开头的Event图片
- 支持的Event后缀: -Accumulate_slow, -Accumulate_fast, -Events, -Event, _event, _events

匹配示例:
- interlaken_00_a_left-280.jpg → interlaken_00_a_left-280-Accumulate_slow.png
- scene_001.jpg → scene_001-Events.png
- test_image.jpg → test_image_event.png
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
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

def adjust_image_brightness(image, factor):
    """调整图片亮度"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def process_single_image(image_path, prompt, chat, conv_vision, brightness_factor=1.0, max_retries=3):
    """处理单张图片并获取回答，包含重试机制"""
    for attempt in range(max_retries):
        try:
            chat_state = conv_vision.copy()
            img_list = []
            
            image = Image.open(image_path).convert('RGB')
            
            # 调整亮度（如果不是1.0）
            if brightness_factor != 1.0:
                image = adjust_image_brightness(image, brightness_factor)
            
            chat.upload_img(image, chat_state, img_list)
            chat.encode_img(img_list)
            
            chat.ask(prompt, chat_state)
            
            answer = chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=1,
                                temperature=1.0,
                                max_new_tokens=300,
                                max_length=2000)[0]
            
            # 检查回答是否为空或无效
            if answer and answer.strip():
                return answer.strip(), image
            else:
                print(f"    ⚠️ 尝试 {attempt + 1}: 回答为空，重试中...")
                if attempt < max_retries - 1:
                    continue
                else:
                    print(f"    ❌ 多次尝试后仍无法获得有效回答")
                    return "模型未能生成有效回答，可能是图片质量或提示词问题。", image
                    
        except Exception as e:
            print(f"    ⚠️ 尝试 {attempt + 1} 时出错: {str(e)}")
            if attempt < max_retries - 1:
                print(f"    🔄 正在重试...")
                continue
            else:
                print(f"    ❌ 多次尝试后仍然失败")
                return f"处理失败: {str(e)}", None
    
    return "处理失败: 未知错误", None

def wrap_text(text, max_width, font):
    """文本换行处理 - 改进版本，更好地处理长词和标点符号"""
    if not text or not text.strip():
        return [""]
    
    # 先按换行符分割
    paragraphs = text.split('\n')
    all_lines = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            all_lines.append("")
            continue
            
        # 按空格分词，但保留标点符号
        words = paragraph.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            # 测试添加当前词后的行宽
            test_line = current_line + word + " " if current_line else word + " "
            
            try:
                # 尝试使用新的getbbox方法
                bbox = font.getbbox(test_line.strip())
                text_width = bbox[2] - bbox[0]
            except AttributeError:
                # 兼容旧版本PIL
                try:
                    text_width = font.getsize(test_line.strip())[0]
                except:
                    # 最后的备选方案：估算宽度
                    text_width = len(test_line.strip()) * 8
            
            # 如果宽度超出限制
            if text_width > max_width:
                if current_line:
                    # 保存当前行
                    lines.append(current_line.strip())
                    current_line = word + " "
                else:
                    # 单词本身太长，强制换行
                    if len(word) > 50:  # 防止过长的单词
                        # 截断长词
                        lines.append(word[:47] + "...")
                        current_line = ""
                    else:
                        current_line = word + " "
            else:
                current_line = test_line
        
        # 添加最后一行
        if current_line.strip():
            lines.append(current_line.strip())
        
        all_lines.extend(lines)
    
    # 确保至少返回一个空行
    return all_lines if all_lines else [""]

def draw_text_in_box(draw, text_lines, x, y, width, height, font, bg_color):
    """在指定区域绘制带背景的文本，彻底解决重叠问题"""
    # 绘制背景
    draw.rectangle([x, y, x + width, y + height], fill=bg_color, outline='gray')
    
    # 如果没有文本，直接返回
    if not text_lines:
        return
    
    # 计算更准确的字体指标
    try:
        # 使用多个测试字符串获取最大高度
        test_strings = ["Ag", "英文", "Test", "gjpqy", "ABCDEFG"]
        max_font_height = 0
        
        for test_str in test_strings:
            try:
                bbox = font.getbbox(test_str)
                font_height = bbox[3] - bbox[1]
                max_font_height = max(max_font_height, font_height)
            except:
                try:
                    font_height = font.getsize(test_str)[1]
                    max_font_height = max(max_font_height, font_height)
                except:
                    max_font_height = max(max_font_height, 16)
        
        # 使用更大的行间距，确保不重叠
        line_height = max(18, int(max_font_height * 1.5))  # 至少18像素，或字体高度的1.5倍
        
    except Exception as e:
        # 安全的默认值
        line_height = 20
    
    # 设置更大的边距
    top_margin = 12
    bottom_margin = 12
    left_margin = 10
    available_height = height - top_margin - bottom_margin
    
    # 确保至少有空间显示一行
    if available_height < line_height:
        line_height = max(12, available_height - 4)  # 留出最小边距
    
    # 计算实际可显示的行数
    max_lines = max(1, available_height // line_height)
    
    # 严格限制显示行数，确保不超出边界
    actual_lines_to_draw = min(len(text_lines), max_lines)
    
    # 如果空间太小，至少显示第一行的一部分
    if actual_lines_to_draw == 0 and len(text_lines) > 0:
        actual_lines_to_draw = 1
    
    # 计算起始Y位置
    text_start_y = y + top_margin
    
    # 绘制文本行，严格控制边界
    lines_drawn = 0
    for i in range(actual_lines_to_draw):
        if i >= len(text_lines):
            break
            
        line = text_lines[i]
        current_y = text_start_y + i * line_height
        
        # 严格检查是否会超出底部边界
        if current_y + line_height > y + height - bottom_margin:
            break
        
        # 如果是最后一行且还有更多文本，添加省略号
        if i == actual_lines_to_draw - 1 and len(text_lines) > actual_lines_to_draw:
            # 为省略号预留空间
            if len(line) > 3:
                line = line[:-3] + "..."
            else:
                line = line + "..."
        
        # 绘制当前行
        draw.text((x + left_margin, current_y), line, fill='black', font=font)
        lines_drawn += 1
    
    # 调试信息（可选，用于排查问题）
    # print(f"Box: {width}x{height}, Lines: {len(text_lines)}, Drew: {lines_drawn}, LineHeight: {line_height}")

def create_six_panel_comparison_image(rgb_image, event_image, rgb_prompt, event_prompt, rgb_answer, event_answer, font_size=11):
    """创建3行2列的6宫格对比图片，彻底解决文本重叠问题"""
    
    # 设置字体 - 使用更小的字体确保有足够空间
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        title_font = ImageFont.truetype("arial.ttf", font_size + 2)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
            title_font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size + 2)
        except:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
    
    # 统一图片尺寸（取两张图片的最大尺寸）
    rgb_width, rgb_height = rgb_image.size
    event_width, event_height = event_image.size
    max_width = max(rgb_width, event_width)
    max_height = max(rgb_height, event_height)
    
    # 创建统一尺寸的图片
    rgb_resized = Image.new('RGB', (max_width, max_height), 'white')
    event_resized = Image.new('RGB', (max_width, max_height), 'white')
    
    # 居中粘贴原图
    rgb_x = (max_width - rgb_width) // 2
    rgb_y = (max_height - rgb_height) // 2
    rgb_resized.paste(rgb_image, (rgb_x, rgb_y))
    
    event_x = (max_width - event_width) // 2
    event_y = (max_height - event_height) // 2
    event_resized.paste(event_image, (event_x, event_y))
    
    # 设置面板尺寸 - 更保守的文本区域设置
    panel_width = max_width
    panel_height = max_height
    prompt_height = 100   # 提示词区域高度
    answer_height = 250   # 回答区域高度 - 进一步增加
    margin = 15           # 增加边距
    title_height = 30     # 标题区域高度
    
    # 创建最终图片
    total_width = panel_width * 2 + margin * 3
    total_height = title_height + panel_height + prompt_height + answer_height + margin * 4
    
    final_image = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(final_image)
    
    # 绘制标题 - 调整位置
    title_y = margin // 2
    draw.text((margin, title_y), "RGB Image", fill='black', font=title_font)
    draw.text((panel_width + margin*2, title_y), "Event Sensor Image", fill='black', font=title_font)
    
    # 第一行：粘贴图片
    y_offset = title_height + margin
    final_image.paste(rgb_resized, (margin, y_offset))
    final_image.paste(event_resized, (panel_width + margin*2, y_offset))
    
    # 第二行：绘制提示词
    y_offset += panel_height + margin
    
    # RGB提示词区域
    rgb_prompt_lines = wrap_text(rgb_prompt, panel_width - margin*2, font)
    draw_text_in_box(draw, rgb_prompt_lines, margin, y_offset, panel_width, prompt_height, font, 'lightblue')
    
    # Event提示词区域
    event_prompt_lines = wrap_text(event_prompt, panel_width - margin*2, font)
    draw_text_in_box(draw, event_prompt_lines, panel_width + margin*2, y_offset, panel_width, prompt_height, font, 'lightgreen')
    
    # 第三行：绘制回答
    y_offset += prompt_height + margin
    
    # RGB回答区域
    rgb_answer_lines = wrap_text(rgb_answer, panel_width - margin*2, font)
    draw_text_in_box(draw, rgb_answer_lines, margin, y_offset, panel_width, answer_height, font, 'lightyellow')
    
    # Event回答区域
    event_answer_lines = wrap_text(event_answer, panel_width - margin*2, font)
    draw_text_in_box(draw, event_answer_lines, panel_width + margin*2, y_offset, panel_width, answer_height, font, 'lightpink')
    
    return final_image

def find_event_image_path(event_dir, rgb_image_name):
    """找到对应的Event sensor图片 - 处理Event图片的特殊命名格式"""
    # 移除RGB图片的扩展名，获取基础文件名
    rgb_base_name = os.path.splitext(rgb_image_name)[0]
    
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # Event图片常见后缀模式（可以根据实际情况扩展）
    event_suffixes = [
        '-Accumulate_slow',
        '-Accumulate_fast', 
        '-Events',
        '-Event',
        '_event',
        '_events'
    ]
    
    # 第一步：尝试匹配带有常见Event后缀的文件名
    for suffix in event_suffixes:
        for ext in image_extensions:
            event_filename = rgb_base_name + suffix + ext
            event_path = os.path.join(event_dir, event_filename)
            if os.path.exists(event_path):
                return event_path
    
    # 第二步：扫描Event目录，寻找以RGB文件名开头的文件
    try:
        event_files = os.listdir(event_dir)
        for event_file in event_files:
            # 检查Event文件是否以RGB基础文件名开头
            event_base_name = os.path.splitext(event_file)[0]
            if event_base_name.lower().startswith(rgb_base_name.lower()):
                # 确认是图片文件
                event_ext = os.path.splitext(event_file)[1].lower()
                if event_ext in image_extensions:
                    return os.path.join(event_dir, event_file)
    except Exception as e:
        print(f"    ⚠️ 扫描Event目录时出错: {str(e)}")
    
    return None

def validate_file_matching(rgb_dir, event_dir):
    """验证RGB和Event图片的文件名匹配情况"""
    print("🔍 验证RGB和Event图片文件名匹配情况...")
    print("📝 匹配规则: RGB基础文件名 → Event文件名(包含RGB文件名+后缀)")
    print("-" * 70)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # 获取RGB图片文件
    rgb_files = [f for f in os.listdir(rgb_dir) 
                 if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    # 获取Event图片文件
    event_files = [f for f in os.listdir(event_dir) 
                   if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    print(f"📁 RGB目录: {len(rgb_files)} 个图片文件")
    print(f"📁 Event目录: {len(event_files)} 个图片文件")
    print()
    
    matched_pairs = []
    unmatched_rgb = []
    unmatched_event = set(event_files)  # 使用set来追踪未匹配的event文件
    
    # 检查每个RGB文件是否有对应的Event文件
    for rgb_file in rgb_files:
        event_path = find_event_image_path(event_dir, rgb_file)
        if event_path:
            event_file = os.path.basename(event_path)
            matched_pairs.append((rgb_file, event_file))
            unmatched_event.discard(event_file)  # 从未匹配列表中移除
            
            # 提取RGB基础名和Event后缀信息
            rgb_base = os.path.splitext(rgb_file)[0]
            event_base = os.path.splitext(event_file)[0]
            suffix = event_base[len(rgb_base):] if len(event_base) > len(rgb_base) else ""
            
            print(f"✅ {rgb_file} ↔ {event_file}")
            if suffix:
                print(f"   └─ 检测到Event后缀: '{suffix}'")
        else:
            unmatched_rgb.append(rgb_file)
            print(f"❌ {rgb_file} (未找到匹配的Event图片)")
    
    print("\n" + "=" * 70)
    print("📊 匹配结果统计:")
    print(f"✅ 成功匹配: {len(matched_pairs)} 对")
    print(f"❌ 未匹配的RGB图片: {len(unmatched_rgb)} 个")
    print(f"⚠️ 未匹配的Event图片: {len(unmatched_event)} 个")
    
    if unmatched_rgb:
        print(f"\n❌ 未匹配的RGB图片:")
        for rgb_file in unmatched_rgb:
            print(f"   - {rgb_file}")
        print(f"   💡 提示: 确保每个RGB图片都有对应的Event图片(含后缀)")
    
    if unmatched_event:
        print(f"\n⚠️ 未匹配的Event图片:")
        for event_file in unmatched_event:
            print(f"   - {event_file}")
        print(f"   💡 提示: 这些Event图片没有对应的RGB图片")
    
    # 显示匹配模式示例
    if matched_pairs:
        print(f"\n📋 匹配模式示例:")
        sample_rgb, sample_event = matched_pairs[0]
        rgb_base = os.path.splitext(sample_rgb)[0]
        event_base = os.path.splitext(sample_event)[0]
        suffix = event_base[len(rgb_base):] if len(event_base) > len(rgb_base) else ""
        print(f"   RGB:   {sample_rgb}")
        print(f"   Event: {sample_event}")
        if suffix:
            print(f"   后缀:  '{suffix}'")
    
    print("=" * 70)
    
    if len(matched_pairs) == 0:
        print("❌ 错误: 没有找到任何匹配的图片对！")
        print("请检查文件命名是否符合规则:")
        print("  - RGB: 基础文件名.jpg")
        print("  - Event: 基础文件名+后缀.png")
        return False, []
    
    if len(unmatched_rgb) > 0:
        print(f"⚠️ 警告: 有 {len(unmatched_rgb)} 个RGB图片没有对应的Event图片")
        user_choice = input("是否继续处理已匹配的图片对？(y/n): ").strip().lower()
        if user_choice != 'y':
            return False, []
    
    return True, matched_pairs

def process_rgb_event_comparison(rgb_path, event_path, rgb_prompt, event_prompt, chat, conv_vision, brightness_factor=1.0):
    """处理RGB和Event图片的对比，增加错误处理"""
    try:
        # 处理RGB图片（带亮度调整）
        print(f"    🔄 处理RGB图片...")
        rgb_answer, rgb_image = process_single_image(rgb_path, rgb_prompt, chat, conv_vision, brightness_factor)
        
        # 处理Event图片（不调整亮度）
        print(f"    🔄 处理Event图片...")
        event_answer, event_image = process_single_image(event_path, event_prompt, chat, conv_vision, 1.0)
        
        # 检查结果是否有效
        if rgb_answer and event_answer and rgb_image and event_image:
            # 确保答案不为空
            if rgb_answer.strip() and event_answer.strip():
                return rgb_image, event_image, rgb_answer, event_answer
            else:
                print(f"    ⚠️ 警告: 获得了空的回答")
                print(f"    📝 RGB回答长度: {len(rgb_answer.strip()) if rgb_answer else 0}")
                print(f"    📝 Event回答长度: {len(event_answer.strip()) if event_answer else 0}")
                return rgb_image, event_image, rgb_answer or "无有效回答", event_answer or "无有效回答"
        else:
            print(f"    ❌ 某些处理结果为空")
            return None, None, None, None
            
    except Exception as e:
        print(f"    ❌ 处理对比图片时出错: {str(e)}")
        return None, None, None, None

def batch_process_rgb_event_comparison(rgb_dir, event_dir, output_base_dir, rgb_prompts, event_prompt, cfg_path, gpu_id=0):
    """批量处理RGB和Event图片对比，包含多种亮度调整"""
    # 设置随机种子
    setup_seeds()
    
    # 首先验证文件匹配情况
    print("=" * 80)
    validation_success, matched_pairs = validate_file_matching(rgb_dir, event_dir)
    if not validation_success:
        print("❌ 文件验证失败，终止处理。")
        return
    
    print(f"\n✅ 文件验证通过，将处理 {len(matched_pairs)} 对匹配的图片")
    print("=" * 80)
    
    # 初始化模型
    chat, conv_vision = init_model(cfg_path, gpu_id)
    
    # 从匹配的文件对中提取RGB文件列表
    rgb_files = [pair[0] for pair in matched_pairs]
    
    # 定义亮度因子
    brightness_factors = [
        (0.1, "factor0.1"),
        (0.5, "factor0.5"),
        (1.0, "normal"),
        (7.0, "factor7.0"),
        (15.0, "factor15.0")
    ]
    
    print(f"🖼️ 将处理 {len(rgb_files)} 张匹配的RGB图片")
    print(f"💬 将使用 {len(rgb_prompts)} 个RGB提示词进行对比")
    print(f"� 将使用 {len(brightness_factors)} 种亮度设置: {[name for _, name in brightness_factors]}")
    print(f"�🎯 Event固定提示词: {event_prompt}")
    print("=" * 80)
    
    # 遍历每个RGB提示词
    for prompt_idx, rgb_prompt in enumerate(rgb_prompts, 1):
        print(f"\n🔄 正在处理RGB提示词 {prompt_idx}/{len(rgb_prompts)}")
        print(f"💬 RGB提示词: {rgb_prompt}")
        print("-" * 80)
        
        # 为每种亮度因子处理
        for brightness_factor, factor_name in brightness_factors:
            print(f"\n🌟 处理亮度设置: {factor_name} (factor={brightness_factor})")
            
            # 为每个RGB提示词和亮度因子创建输出目录
            output_dir = os.path.join(output_base_dir, f"rgb_prompt_{prompt_idx:02d}_{factor_name}")
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 输出目录: {output_dir}")
            
            success_count = 0
            
            # 处理每张RGB图片
            for img_idx, rgb_file in enumerate(rgb_files, 1):
                print(f"  📸 ({img_idx}/{len(rgb_files)}) 处理图片: {rgb_file}")
                
                rgb_path = os.path.join(rgb_dir, rgb_file)
                event_path = find_event_image_path(event_dir, rgb_file)
                
                if event_path is None:
                    print(f"    ❌ 错误: 未找到匹配的Event图片")
                    continue
                
                event_file = os.path.basename(event_path)
                print(f"    🔗 匹配Event图片: {event_file}")
                print(f"    🌟 亮度因子: {brightness_factor}")
                
                # 处理RGB和Event图片对比
                rgb_image, event_image, rgb_answer, event_answer = process_rgb_event_comparison(
                    rgb_path, event_path, rgb_prompt, event_prompt, chat, conv_vision, brightness_factor
                )
                
                if all([rgb_image, event_image, rgb_answer, event_answer]):
                    # 创建6宫格对比图片
                    comparison_image = create_six_panel_comparison_image(
                        rgb_image, event_image, rgb_prompt, event_prompt, rgb_answer, event_answer
                    )
                    
                    # 保存结果
                    output_filename = f"comparison_{os.path.splitext(rgb_file)[0]}_{factor_name}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    comparison_image.save(output_path, 'JPEG', quality=95)
                    
                    print(f"    ✅ 保存对比图: {output_filename}")
                    print(f"    💭 RGB回答: {rgb_answer[:60]}...")
                    print(f"    💭 Event回答: {event_answer[:60]}...")
                    success_count += 1
                else:
                    print(f"    ❌ 处理失败")
            
            # 当前亮度设置处理完成的统计
            print(f"\n📊 RGB提示词 {prompt_idx} - {factor_name} 处理完成:")
            print(f"  ✅ 成功: {success_count}/{len(rgb_files)} 张图片")
            print(f"  📁 结果保存在: {output_dir}")
        
        if prompt_idx < len(rgb_prompts):
            print(f"\n⏳ 准备处理下一个RGB提示词...")
    
    print("\n" + "=" * 80)
    print("🎉 所有RGB提示词和亮度对比处理完成!")
    print(f"📁 所有结果保存在: {output_base_dir}")
    print(f"📊 总共生成了 {len(rgb_prompts)} × {len(brightness_factors)} = {len(rgb_prompts) * len(brightness_factors)} 个对比文件夹")
    print("=" * 80)

if __name__ == "__main__":
    # ===== 配置参数 - 请根据您的实际情况修改 =====
    
    # 设置工作目录为MiniGPT-4目录
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"🔧 设置工作目录为: {script_dir}")
    
    # 输入路径配置
    RGB_DIR = r"C:\hku\program\det_DSEC\test1\condition_images"          # RGB图片文件夹路径
    EVENT_DIR = r"C:\hku\program\det_DSEC\test1\images"      # Event sensor图片文件夹路径
    OUTPUT_BASE_DIR = r"C:\hku\program\det_DSEC\rgb_event_comparison_results"  # 输出基础目录
    
    # Event sensor固定提示词（根据您的需求修改）
    EVENT_PROMPT = "Describe the events occurring in this event sensor image."

    # RGB图片的自定义提示词 - 这些会变化用于对比
    RGB_CUSTOM_PROMPTS = [
        # 基础描述类
        "Please describe this image in detail.",
        "What do you see in this image?",
        "Describe the main objects and activities in this picture.",
        
        # # 中文提示词
        # "请详细描述这张图片。",
        # "这张图片中有什么？",
        # "描述一下图片中的主要内容。",
        
        # 特定任务类
        "What objects can you see in this image?",
        "Describe the people in this image.",
        # "What is the main activity happening in this image?",
        # "What emotions or mood does this image convey?",
        # "Describe the setting and environment of this image.",
        
        # # 分析类
        # "What is unusual or interesting about this image?",
        # "What story does this image tell?",
        # "If you were to give this image a title, what would it be?",
        
        # 您可以在这里添加更多RGB提示词
        # "Your custom RGB prompt here",
    ]
    
    # RGB提示词范围配置
    RGB_PROMPT_RANGE = None  # 设置为 (start, end) 处理部分提示词，如 (0, 3) 或 None 处理所有
    
    # 技术配置
    CFG_PATH = "eval_configs/minigpt4_eval.yaml"  # 配置文件路径
    GPU_ID = 0  # GPU ID
    
    # ===== 处理模式判断 =====
    print("🎯 MiniGPT-4 RGB-Event对比分析")
    print("=" * 60)
    print(f"📁 RGB目录: {RGB_DIR}")
    print(f"📁 Event目录: {EVENT_DIR}")
    print(f"📁 输出目录: {OUTPUT_BASE_DIR}")
    print(f"⚙️ 配置文件: {CFG_PATH}")
    print(f"🔧 GPU ID: {GPU_ID}")
    print()
    print("📋 文件命名规则:")
    print("  RGB:   基础文件名.jpg (例如: interlaken_00_a_left-280.jpg)")
    print("  Event: 基础文件名+后缀.png (例如: interlaken_00_a_left-280-Accumulate_slow.png)")
    print("=" * 60)
    
    # 检查输入目录
    if not os.path.exists(RGB_DIR):
        print(f"❌ 错误: RGB图片目录不存在: {RGB_DIR}")
        print("请修改脚本中的 RGB_DIR 变量")
        exit(1)
    
    if not os.path.exists(EVENT_DIR):
        print(f"❌ 错误: Event图片目录不存在: {EVENT_DIR}")
        print("请修改脚本中的 EVENT_DIR 变量")
        exit(1)
    
    # RGB-Event对比模式
    if RGB_PROMPT_RANGE is not None:
        start_idx, end_idx = RGB_PROMPT_RANGE
        selected_rgb_prompts = RGB_CUSTOM_PROMPTS[start_idx:end_idx]
        print(f"🔹 模式: RGB-Event对比分析 (RGB提示词范围: {start_idx}-{end_idx-1})")
    else:
        selected_rgb_prompts = RGB_CUSTOM_PROMPTS
        print(f"🔹 模式: RGB-Event对比分析 (全部RGB提示词)")
    
    print(f"💬 Event固定提示词: {EVENT_PROMPT}")
    print(f"💬 将使用 {len(selected_rgb_prompts)} 个RGB提示词进行对比:")
    for i, prompt in enumerate(selected_rgb_prompts):
        print(f"  {i+1:2d}. {prompt}")
    
    print(f"\n🌟 RGB亮度设置:")
    print(f"  • factor=0.1  (很暗)")
    print(f"  • factor=0.5  (较暗)")
    print(f"  • factor=1.0  (原图)")
    print(f"  • factor=7.0  (较亮)")
    print(f"  • factor=15.0 (很亮)")
    
    print(f"\n📁 输出基础目录: {OUTPUT_BASE_DIR}")
    print("   (每个RGB提示词×亮度组合会创建单独的对比文件夹)")
    print("\n📋 输出格式: 3行2列的6宫格对比图")
    print("   第1行: RGB图片(调整亮度) | Event图片")
    print("   第2行: RGB提示词 | Event提示词")
    print("   第3行: RGB回答 | Event回答")
    print("\n📂 文件夹命名示例:")
    print("   • rgb_prompt_01_factor0.1/")
    print("   • rgb_prompt_01_normal/")
    print("   • rgb_prompt_01_factor15.0/")
    
    # 计算总处理量
    total_combinations = len(selected_rgb_prompts) * 5  # 5种亮度设置
    print(f"\n📊 总处理量: {len(selected_rgb_prompts)} 个提示词 × 5 种亮度 = {total_combinations} 个文件夹")
    
    # 确认继续
    user_input = input(f"\n继续处理 {total_combinations} 种组合的对比分析？按 Enter 继续，或输入 'q' 退出: ").strip().lower()
    if user_input == 'q':
        print("已取消处理。")
        exit(0)
    
    # 运行RGB-Event对比处理
    batch_process_rgb_event_comparison(RGB_DIR, EVENT_DIR, OUTPUT_BASE_DIR, selected_rgb_prompts, EVENT_PROMPT, CFG_PATH, GPU_ID)
