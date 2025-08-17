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


def setup_seeds(seed=42):
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
    
    print('正在初始化MiniGPT-4模型...')
    
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
    print('模型初始化完成!')
    
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
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        return None, None


def create_combined_image(original_image, answer_text, font_size=20):
    """将回答文本添加到图片下方（保留原有功能用于兼容性）"""
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


def create_six_panel_comparison_image(rgb_image, event_image, rgb_prompt, event_prompt, rgb_answer, event_answer, font_size=16):
    """创建3行2列的6宫格对比图片"""
    
    # 设置字体
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
    
    # 设置面板尺寸
    panel_width = max_width
    panel_height = max_height
    prompt_height = 80  # 提示词区域高度
    answer_height = 120  # 回答区域高度
    margin = 10
    
    # 创建最终图片
    total_width = panel_width * 2 + margin * 3
    total_height = panel_height + prompt_height + answer_height + margin * 4
    
    final_image = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(final_image)
    
    # 绘制标题
    draw.text((margin, margin//2), "RGB Image", fill='black', font=title_font)
    draw.text((panel_width + margin*2, margin//2), "Event Sensor Image", fill='black', font=title_font)
    
    # 第一行：粘贴图片
    y_offset = margin + 20
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


def wrap_text(text, max_width, font):
    """文本换行处理"""
    words = text.split(' ')
    lines = []
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
                lines.append(current_line.strip())
                current_line = word + " "
            else:
                lines.append(word)
                current_line = ""
    
    if current_line:
        lines.append(current_line.strip())
    
    return lines


def draw_text_in_box(draw, text_lines, x, y, width, height, font, bg_color):
    """在指定区域绘制带背景的文本"""
    # 绘制背景
    draw.rectangle([x, y, x + width, y + height], fill=bg_color, outline='gray')
    
    # 绘制文本
    line_height = 20
    text_y = y + 5
    
    for line in text_lines:
        if text_y + line_height < y + height:  # 确保不超出区域
            draw.text((x + 5, text_y), line, fill='black', font=font)
            text_y += line_height
        else:
            break


def find_event_image_path(event_dir, rgb_image_name):
    """找到对应的Event sensor图片"""
    # 移除RGB图片的扩展名
    base_name = os.path.splitext(rgb_image_name)[0]
    
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # 尝试找到对应的event图片
    for ext in image_extensions:
        event_path = os.path.join(event_dir, base_name + ext)
        if os.path.exists(event_path):
            return event_path
    
    # 如果找不到完全匹配的，尝试寻找包含base_name的文件
    for file in os.listdir(event_dir):
        if base_name.lower() in file.lower():
            return os.path.join(event_dir, file)
    
    return None


def process_rgb_event_comparison(rgb_path, event_path, rgb_prompt, event_prompt, chat, conv_vision):
    """处理RGB和Event图片的对比"""
    try:
        # 处理RGB图片
        rgb_answer, rgb_image = process_single_image(rgb_path, rgb_prompt, chat, conv_vision)
        
        # 处理Event图片
        event_answer, event_image = process_single_image(event_path, event_prompt, chat, conv_vision)
        
        if rgb_answer and event_answer and rgb_image and event_image:
            return rgb_image, event_image, rgb_answer, event_answer
        else:
            return None, None, None, None
            
    except Exception as e:
        print(f"处理对比图片时出错: {str(e)}")
        return None, None, None, None


def batch_process_rgb_event_comparison(rgb_dir, event_dir, output_base_dir, rgb_prompts, event_prompt, cfg_path, gpu_id=0):
    """批量处理RGB和Event图片对比"""
    # 设置随机种子
    setup_seeds()
    
    # 初始化模型
    chat, conv_vision = init_model(cfg_path, gpu_id)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # 获取所有RGB图片文件
    rgb_files = [f for f in os.listdir(rgb_dir) 
                 if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not rgb_files:
        print(f"❌ 在 {rgb_dir} 中没有找到RGB图片文件")
        return
    
    print(f"🖼️ 找到 {len(rgb_files)} 张RGB图片")
    print(f"💬 将使用 {len(rgb_prompts)} 个RGB提示词进行对比")
    print(f"🎯 Event固定提示词: {event_prompt}")
    print("=" * 80)
    
    # 遍历每个RGB提示词
    for prompt_idx, rgb_prompt in enumerate(rgb_prompts, 1):
        print(f"\n🔄 正在处理RGB提示词 {prompt_idx}/{len(rgb_prompts)}")
        print(f"💬 RGB提示词: {rgb_prompt}")
        print("-" * 80)
        
        # 为每个RGB提示词创建输出目录
        safe_prompt = "".join(c if c.isalnum() or c in '-_' else '_' for c in rgb_prompt[:50])
        output_dir = os.path.join(output_base_dir, f"rgb_prompt_{prompt_idx:02d}_{safe_prompt}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 输出目录: {output_dir}")
        
        success_count = 0
        
        # 处理每张RGB图片
        for img_idx, rgb_file in enumerate(rgb_files, 1):
            print(f"  📸 ({img_idx}/{len(rgb_files)}) 处理图片: {rgb_file}")
            
            rgb_path = os.path.join(rgb_dir, rgb_file)
            event_path = find_event_image_path(event_dir, rgb_file)
            
            if event_path is None:
                print(f"    ❌ 未找到对应的Event图片")
                continue
            
            # 处理RGB和Event图片对比
            rgb_image, event_image, rgb_answer, event_answer = process_rgb_event_comparison(
                rgb_path, event_path, rgb_prompt, event_prompt, chat, conv_vision
            )
            
            if all([rgb_image, event_image, rgb_answer, event_answer]):
                # 创建6宫格对比图片
                comparison_image = create_six_panel_comparison_image(
                    rgb_image, event_image, rgb_prompt, event_prompt, rgb_answer, event_answer
                )
                
                # 保存结果
                output_filename = f"comparison_{os.path.splitext(rgb_file)[0]}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                comparison_image.save(output_path, 'JPEG', quality=95)
                
                print(f"    ✅ 保存对比图: {output_filename}")
                print(f"    💭 RGB回答: {rgb_answer[:60]}...")
                print(f"    💭 Event回答: {event_answer[:60]}...")
                success_count += 1
            else:
                print(f"    ❌ 处理失败")
        
        # 当前提示词处理完成的统计
        print(f"\n📊 RGB提示词 {prompt_idx} 处理完成:")
        print(f"  ✅ 成功: {success_count}/{len(rgb_files)} 张图片")
        print(f"  📁 结果保存在: {output_dir}")
        
        if prompt_idx < len(rgb_prompts):
            print(f"\n⏳ 准备处理下一个RGB提示词...")
    
    print("\n" + "=" * 80)
    print("🎉 所有RGB提示词对比处理完成!")
    print(f"📁 所有结果保存在: {output_base_dir}")
    print("=" * 80)
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


def batch_process_images_with_multiple_prompts(input_dir, output_base_dir, prompts, cfg_path, gpu_id=0):
    """使用多个提示词批量处理图片文件夹"""
    # 设置随机种子
    setup_seeds()
    
    # 初始化模型（只初始化一次，提高效率）
    chat, conv_vision = init_model(cfg_path, gpu_id)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # 获取所有图片文件
    image_files = []
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print(f"❌ 在 {input_dir} 中没有找到图片文件")
        return
    
    print(f"🖼️ 找到 {len(image_files)} 张图片")
    print(f"💬 将使用 {len(prompts)} 个提示词处理")
    print("=" * 80)
    
    # 遍历每个提示词
    for prompt_idx, prompt in enumerate(prompts, 1):
        print(f"\n🔄 正在处理提示词 {prompt_idx}/{len(prompts)}")
        print(f"💬 当前提示词: {prompt}")
        print("-" * 80)
        
        # 为每个提示词创建单独的输出目录
        safe_prompt = "".join(c if c.isalnum() or c in '-_' else '_' for c in prompt[:50])
        output_dir = os.path.join(output_base_dir, f"prompt_{prompt_idx:02d}_{safe_prompt}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 输出目录: {output_dir}")
        
        success_count = 0
        
        # 遍历当前提示词下的所有图片
        for img_idx, image_file in enumerate(image_files, 1):
            print(f"  📸 ({img_idx}/{len(image_files)}) 处理图片: {image_file}")
            
            image_path = os.path.join(input_dir, image_file)
            
            # 获取模型回答
            answer, original_image = process_single_image(image_path, prompt, chat, conv_vision)
            
            if answer is not None and original_image is not None:
                # 创建组合图片
                combined_image = create_combined_image(original_image, answer)
                
                # 保存结果
                output_filename = f"result_{os.path.splitext(image_file)[0]}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                combined_image.save(output_path, 'JPEG', quality=95)
                
                print(f"    ✅ 保存到: {output_filename}")
                print(f"    💭 回答: {answer[:80]}...")  # 显示前80个字符
                success_count += 1
            else:
                print(f"    ❌ 处理失败")
        
        # 当前提示词处理完成的统计
        print(f"\n📊 提示词 {prompt_idx} 处理完成:")
        print(f"  ✅ 成功: {success_count}/{len(image_files)} 张图片")
        print(f"  📁 结果保存在: {output_dir}")
        
        if prompt_idx < len(prompts):
            print(f"\n⏳ 准备处理下一个提示词...")
    
    print("\n" + "=" * 80)
    print("🎉 所有提示词处理完成!")
    print(f"📁 所有结果保存在: {output_base_dir}")
    print("=" * 80)


def batch_process_images(input_dir, output_dir, prompt, cfg_path, gpu_id=0):
    """批量处理图片文件夹"""
    # 设置随机种子
    setup_seeds()
    
    # 初始化模型
    chat, conv_vision = init_model(cfg_path, gpu_id)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # 获取所有图片文件
    image_files = []
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 处理每张图片
    for i, image_file in enumerate(image_files, 1):
        print(f"正在处理 ({i}/{len(image_files)}): {image_file}")
        
        image_path = os.path.join(input_dir, image_file)
        
        # 获取模型回答
        answer, original_image = process_single_image(image_path, prompt, chat, conv_vision)
        
        if answer is not None and original_image is not None:
            # 创建组合图片
            combined_image = create_combined_image(original_image, answer)
            
            # 保存结果
            output_filename = f"result_{os.path.splitext(image_file)[0]}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            combined_image.save(output_path, 'JPEG', quality=95)
            
            print(f"  ✓ 保存到: {output_path}")
            print(f"  ✓ 回答: {answer[:100]}...")  # 显示前100个字符
        else:
            print(f"  ✗ 处理失败")
        
        print()  # 空行分隔
    
    print("批量处理完成!")


if __name__ == "__main__":
    # ===== 配置参数 - 请根据您的实际情况修改 =====
    
    # 输入和输出路径
    INPUT_DIR = "examples"  # 输入图片文件夹路径
    OUTPUT_BASE_DIR = "multi_prompt_results"  # 输出基础目录（每个提示词会有单独的子文件夹）
    
    # 自定义提示词 - 在这里设置您想问的问题
    CUSTOM_PROMPTS = [
        # 基础描述类
        "Please describe this image in detail.",
        "What do you see in this image?",
        "Describe the main objects and activities in this picture.",
        
        # 中文提示词
        "请详细描述这张图片。",
        "这张图片中有什么？",
        "描述一下图片中的主要内容。",
        
        # 特定任务类
        "What objects can you see in this image?",
        "Describe the people in this image.",
        "What is the main activity happening in this image?",
        "What emotions or mood does this image convey?",
        "Describe the setting and environment of this image.",
        
        # 分析类
        "What is unusual or interesting about this image?",
        "What story does this image tell?",
        "If you were to give this image a title, what would it be?",
        
        # 您可以在这里添加更多自定义提示词
        # "Your custom prompt here",
    ]
    
    # 处理模式选择
    PROCESS_MODE = "multiple"  # "single" = 只处理一个提示词, "multiple" = 处理所有提示词
    
    # 单个提示词模式的配置（当 PROCESS_MODE = "single" 时使用）
    SELECTED_PROMPT_INDEX = 0  # 选择要使用的提示词索引
    CUSTOM_PROMPT = None  # 或者直接设置自定义提示词（会覆盖上面的选择）
    
    # 多提示词模式的配置（当 PROCESS_MODE = "multiple" 时使用）
    PROMPT_RANGE = None  # 设置为 (start, end) 处理部分提示词，如 (0, 5) 或 None 处理所有
    
    # 技术配置
    CFG_PATH = "eval_configs/minigpt4_eval.yaml"  # 配置文件路径
    GPU_ID = 0  # GPU ID
    
    # ===== 处理模式判断 =====
    print("🎯 MiniGPT-4 批量处理配置")
    print("=" * 60)
    print(f"📁 输入目录: {INPUT_DIR}")
    print(f"⚙️ 配置文件: {CFG_PATH}")
    print(f"🔧 GPU ID: {GPU_ID}")
    print("=" * 60)
    
    if PROCESS_MODE == "single":
        # 单个提示词模式
        if CUSTOM_PROMPT is not None:
            final_prompt = CUSTOM_PROMPT
            print(f"🔹 模式: 单个自定义提示词")
            print(f"💬 提示词: {final_prompt}")
        else:
            if 0 <= SELECTED_PROMPT_INDEX < len(CUSTOM_PROMPTS):
                final_prompt = CUSTOM_PROMPTS[SELECTED_PROMPT_INDEX]
                print(f"🔹 模式: 单个选定提示词")
                print(f"💬 提示词 #{SELECTED_PROMPT_INDEX}: {final_prompt}")
            else:
                print(f"❌ 错误：提示词索引 {SELECTED_PROMPT_INDEX} 超出范围 (0-{len(CUSTOM_PROMPTS)-1})")
                print("可用的提示词列表：")
                for i, prompt in enumerate(CUSTOM_PROMPTS):
                    print(f"  {i}: {prompt}")
                exit(1)
        
        print(f"📁 输出目录: {OUTPUT_BASE_DIR}")
        
        # 确认继续
        user_input = input("\n按 Enter 继续，或输入 'q' 退出: ").strip().lower()
        if user_input == 'q':
            print("已取消处理。")
            exit(0)
        
        # 运行单个提示词处理
        batch_process_images(INPUT_DIR, OUTPUT_BASE_DIR, final_prompt, CFG_PATH, GPU_ID)
        
    elif PROCESS_MODE == "multiple":
        # 多个提示词模式
        if PROMPT_RANGE is not None:
            start_idx, end_idx = PROMPT_RANGE
            selected_prompts = CUSTOM_PROMPTS[start_idx:end_idx]
            print(f"� 模式: 多提示词处理 (范围: {start_idx}-{end_idx-1})")
        else:
            selected_prompts = CUSTOM_PROMPTS
            print(f"🔹 模式: 多提示词处理 (全部)")
        
        print(f"💬 将处理 {len(selected_prompts)} 个提示词:")
        for i, prompt in enumerate(selected_prompts):
            print(f"  {i+1:2d}. {prompt}")
        
        print(f"� 输出基础目录: {OUTPUT_BASE_DIR}")
        print("   (每个提示词会创建单独的子文件夹)")
        
        # 确认继续
        user_input = input(f"\n继续处理 {len(selected_prompts)} 个提示词？按 Enter 继续，或输入 'q' 退出: ").strip().lower()
        if user_input == 'q':
            print("已取消处理。")
            exit(0)
        
        # 运行多提示词处理
        batch_process_images_with_multiple_prompts(INPUT_DIR, OUTPUT_BASE_DIR, selected_prompts, CFG_PATH, GPU_ID)
        
    else:
        print(f"❌ 错误：未知的处理模式 '{PROCESS_MODE}'")
        print("请设置 PROCESS_MODE 为 'single' 或 'multiple'")
        exit(1)




