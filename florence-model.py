import os
import sys
import torch
import subprocess
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


def check_and_install_dependencies():
    """检查并安装必要的依赖库"""
    required_packages = [
        "torch", 
        "Pillow", 
        "transformers"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"检测到缺失依赖 {package}，正在安装...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT
                )
                print(f"{package} 安装成功")
            except subprocess.CalledProcessError:
                print(f"错误：{package} 安装失败，请手动运行 'pip install {package}' 后重试")
                sys.exit(1)


def get_valid_image_directory():
    """获取有效的图片文件夹路径"""
    while True:
        image_dir = input("请输入图片文件夹路径：").strip()
        if os.path.isdir(image_dir):
            return image_dir
        print(f"错误：路径 '{image_dir}' 不是有效的文件夹，请重新输入")


def get_concept_sentence():
    """获取非空的触发词"""
    while True:
        concept = input("请输入触发词：").strip()
        if concept:
            return concept
        print("错误：触发词不能为空，请重新输入")


def get_max_labels():
    """获取有效的最大标签数量（默认10）"""
    while True:
        input_str = input("请输入每个图片最多提取的标签数量（默认10）：").strip()
        if not input_str:
            return 10
        # 修复语法错误：去掉多余的input()，修正括号匹配
        if input_str.isdigit() and int(input_str) > 0:
            return int(input_str)
        print("错误：请输入一个正整数")


# 设备配置（优先使用GPU）
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def run_captioning(image_path, concept_sentence=None, model=None, processor=None):
    """使用Florence-2模型处理单张图片"""
    if model is None or processor is None:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "./Florence-2-large-no-flash-attn",
                torch_dtype=torch_dtype,
                trust_remote_code=True
            ).to(device)
            processor = AutoProcessor.from_pretrained(
                "./Florence-2-large-no-flash-attn",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"模型加载失败：{str(e)}")
            print("请确保模型文件夹 'Florence-2-large-no-flash-attn' 位于脚本同一目录下")
            sys.exit(1)
    
    try:
        image = Image.open(image_path).convert("RGB")
        prompt = "<DETAILED_CAPTION>"
        
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(device, torch_dtype)
        
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height)
        )
        
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        if concept_sentence:
            caption_text = f"{concept_sentence} {caption_text}"
            
        labels = parsed_answer.get("<OD>", {}).get("labels", [])
        
        return {
            "caption": caption_text,
            "labels": labels,
            "image_size": (image.width, image.height)
        }
        
    except Exception as e:
        print(f"处理图片 {os.path.basename(image_path)} 时出错: {str(e)}")
        return None


def batch_captioning(image_paths, concept_sentence=None, max_labels=10):
    """批量处理图片并保存结果"""
    model = AutoModelForCausalLM.from_pretrained(
        "./Florence-2-large-no-flash-attn",
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "./Florence-2-large-no-flash-attn",
        trust_remote_code=True
    )
    
    processed_count = 0
    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        
        print(f"正在处理图片 {i+1}/{len(image_paths)}: {image_name}")
        
        result = run_captioning(image_path, concept_sentence, model, processor)
        if result:
            caption_text = result["caption"]
            if concept_sentence and caption_text.startswith(concept_sentence):
                caption_text = caption_text[len(concept_sentence):].strip()
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"{concept_sentence},{caption_text}")
            
            processed_count += 1
            print(f"✓ 已生成标注文件: {os.path.basename(txt_path)}")
    
    model.to("cpu")
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return processed_count


def main():
    check_and_install_dependencies()
    
    print("=" * 50)
    print("图片批量打标工具 - 使用Florence-2模型生成描述和标签")
    print("=" * 50)
    
    image_dir = get_valid_image_directory()
    concept_sentence = get_concept_sentence()
    max_labels = get_max_labels()
    
    supported_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    image_paths = [
        os.path.join(image_dir, filename)
        for filename in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, filename)) and
        os.path.splitext(filename)[1].lower() in supported_extensions
    ]
    
    if not image_paths:
        print(f"错误：在 '{image_dir}' 中未找到任何支持的图片文件")
        print(f"支持的格式：{', '.join(supported_extensions)}")
        return
    
    print(f"\n开始处理 {len(image_paths)} 张图片...")
    print(f"图片文件夹: {image_dir}")
    print(f"触发词: {concept_sentence}")
    print(f"最大标签数量: {max_labels}")
    
    processed_count = batch_captioning(
        image_paths,
        concept_sentence=concept_sentence,
        max_labels=max_labels
    )
    
    print(f"\n{'=' * 50}")
    print(f"处理完成！共成功处理 {processed_count}/{len(image_paths)} 张图片")
    print(f"标注文件已保存至: {image_dir}")


if __name__ == "__main__":
    main()
    