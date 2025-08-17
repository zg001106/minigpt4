import os
from PIL import Image, ImageEnhance

def dim_image(input_path, output_path, factor=0.5):
    """
    调暗图片亮度并保存到指定路径。

    参数:
    - input_path: 输入图片路径
    - output_path: 输出图片路径（包含文件夹）
    - factor: 亮度因子（0.0 完全黑，1.0 原始亮度）
    """
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 打开图片
    image = Image.open(input_path)

    # 创建亮度增强器
    enhancer = ImageEnhance.Brightness(image)

    # 调整亮度
    dimmed_image = enhancer.enhance(factor)

    # 保存结果
    dimmed_image.save(output_path)
    print(f"✅ 图片已保存到 {output_path}")

if __name__ == "__main__":
    # 示例：输入和输出路径
    input_image = "C:\hku\program\det_DSEC/test\conditioning_images\interlaken_00_a_left-500.jpg"  # 替换为你的原始图片路径
    output_image = "C:\hku\program\det_DSEC\darkerimg\interlaken_00_a_left-500-brighter15.0.jpg"  # 输出路径（含文件夹）
    brightness_factor = 15.0  # 越小越暗（0.0 到 1.0）

    dim_image(input_image, output_image, brightness_factor)
