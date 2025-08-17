# MiniGPT-4 批量图片推理使用指南

## 功能说明

这个脚本可以批量处理文件夹中的所有图片，使用 MiniGPT-4 模型进行推理，并将回答文本添加到图片下方，生成新的组合图片。

## 效果展示

```
原始图片
+------------------+
|                  |
|   您的图片内容    |
|                  |
+------------------+
MiniGPT-4的回答文本会
显示在这里，支持自动
换行处理...
```

## 使用步骤

### 1. 准备工作

确保您已经：
- 安装并配置好 MiniGPT-4 环境
- 下载了预训练权重
- GPU 可用且 CUDA 环境正常

### 2. 修改配置

编辑 `batch_inference.py` 文件顶部的配置参数：

```python
# 输入图片文件夹路径
INPUT_DIR = "your/image/folder"

# 输出结果文件夹路径  
OUTPUT_DIR = "output_results"

# 提示词（所有图片都会使用这个提示词）
PROMPT = "Please describe this image in detail."

# 配置文件路径
CFG_PATH = "eval_configs/minigpt4_eval.yaml"

# GPU ID
GPU_ID = 0

# 字体大小
FONT_SIZE = 16
```

### 3. 运行脚本

在 MiniGPT-4 项目根目录下运行：

```bash
python batch_inference.py
```

## 配置说明

### INPUT_DIR（输入目录）
- 包含要处理的图片的文件夹路径
- 支持的图片格式：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
- 示例：`"examples"` 或 `"C:/Users/YourName/Pictures"`

### OUTPUT_DIR（输出目录）
- 处理后的图片保存位置
- 如果目录不存在，脚本会自动创建
- 示例：`"output_results"` 或 `"C:/Output"`

### PROMPT（提示词）
根据您的需求设置不同的提示词：

```python
# 通用描述
PROMPT = "Please describe this image in detail."

# 中文提示
PROMPT = "请详细描述这张图片。"

# 特定任务
PROMPT = "What objects can you see in this image?"
PROMPT = "Describe the people in this image."
PROMPT = "What is the main activity in this image?"
```

### CFG_PATH（配置文件）
选择合适的配置文件：
- `"eval_configs/minigpt4_eval.yaml"` - 标准评估配置
- `"eval_configs/minigpt4_llama2_eval.yaml"` - LLaMA2 版本

### GPU_ID
- 如果只有一个 GPU，设置为 `0`
- 如果有多个 GPU，可以选择 `0`, `1`, `2` 等

## 输出文件命名

处理后的文件会以 `result_` 前缀命名：
- 原文件：`photo1.jpg`
- 输出文件：`result_photo1.jpg`

## 错误处理

### 常见问题

1. **"输入目录不存在"**
   - 检查 `INPUT_DIR` 路径是否正确
   - 使用绝对路径：`"C:/Users/YourName/Pictures"`

2. **"配置文件不存在"**
   - 检查 `CFG_PATH` 是否正确
   - 确保您在 MiniGPT-4 项目根目录下运行脚本

3. **"CUDA out of memory"**
   - 减少 `max_new_tokens` 参数
   - 使用更小的 GPU ID
   - 一次处理更少的图片

4. **"没有找到图片文件"**
   - 确保输入目录包含支持的图片格式
   - 检查文件扩展名是否正确

## 自定义选项

### 修改输出格式

如果您想修改文本在图片中的显示方式，可以编辑 `create_combined_image` 函数：

```python
def create_combined_image(original_image, answer_text, font_size=20):
    # 修改字体颜色
    draw.text((10, y_offset), line, fill='blue', font=font)
    
    # 修改背景颜色
    combined_image = Image.new('RGB', (img_width, new_height), 'lightgray')
    
    # 修改边距
    max_width = img_width - 40  # 更大的边距
```

### 批量处理不同提示词

如果您想用不同的提示词处理同一批图片：

```python
prompts = [
    "Please describe this image in detail.",
    "What objects can you see in this image?",
    "Describe the emotions shown in this image."
]

for i, prompt in enumerate(prompts):
    output_dir = f"output_results_prompt_{i+1}"
    batch_process_images(INPUT_DIR, output_dir, prompt, CFG_PATH, GPU_ID)
```

## 性能优化

- 使用 SSD 存储提高 I/O 性能
- 确保有足够的 GPU 内存
- 关闭其他占用 GPU 的程序
- 使用较新的 PyTorch 版本

## 注意事项

1. 确保有足够的磁盘空间存储输出图片
2. 处理大量图片时建议分批进行
3. 第一次运行时模型加载需要一些时间
4. 输出图片的质量设置为 95%，可以根据需要调整
