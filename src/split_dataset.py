import os
import shutil
import random
from pathlib import Path
import re

# ── 路径配置（第 1 批：把原来写死的绝对路径抽到这里）──────────────────────────
# 数据集与标签**不入库**（体积 4+ GB，且原数据集许可不允许再分发）。
# 使用前按 README 把数据放到这些位置，或用环境变量覆盖：
#   PLANT_DATA_DIR   原始数据目录（默认 ./data）
#   PLANT_OUTPUT_DIR 划分后数据集目录（默认 ./datasets）
DATA_DIR = Path(os.environ.get("PLANT_DATA_DIR", "data"))
IMAGE_DIR = DATA_DIR / "image"
LABEL_DIR = DATA_DIR / "labels"
OUTPUT_DIR = Path(os.environ.get("PLANT_OUTPUT_DIR", "datasets"))
CLASSES_FILE = Path(__file__).resolve().parent.parent / "configs" / "classes.json"



def create_class_mapping():
    """创建增强型分类映射表（支持多语言字符处理）"""
    mapping = {
        # 苹果类病害
        'Apple___Apple_scab（苹果-苹果疮痂病）': {
            'target_class': 'Apple_Apple_scab',
            'label_dir': Path('高频病害(4种)') / '苹果疮痂病'
        },
        'Apple___Black_rot（苹果-黑腐病）': {
            'target_class': 'Apple_Black_rot',
            'label_dir': Path('罕见病害(5种)') / '苹果黑腐病'
        },
        'Apple___Cedar_apple_rust（苹果-雪松苹果锈病）': {
            'target_class': 'Apple_Cedar_apple_rust',
            'label_dir': Path('罕见病害(5种)') / '苹果雪松苹果锈病'
        },
        'Apple___healthy（健康苹果）': {
            'target_class': 'Apple_healthy',
            'label_dir': Path('健康类(12种)') / '健康苹果'
        },
        # ...（其他类别保持相同映射结构）
    }

    # 添加字符标准化处理
    for k, v in mapping.items():
        mapping[k]['clean_name'] = (
            re.sub(r'[（）()\[\]\{\}\<\>]', '', k)
            .replace('___', '_')
            .replace('．', '.')
            .replace('－', '-')
        )

    return mapping


def repair_filename(filename):
    """文件名修复函数（处理常见编码问题）"""
    return (
        str(filename)
        .encode('utf-8')
        .decode('latin-1')
        .encode('latin-1')
        .decode('utf-8')
    )


def reorganize_dataset(val_ratio=0.2):
    # 配置路径（自动路径规范化）
    image_source_base = (IMAGE_DIR / '数据集1' / '植物叶片病害原始数据集').resolve()
    label_source_base = LABEL_DIR.resolve()
    target_base = OUTPUT_DIR.resolve()

    # 确保目标基础目录存在
    target_base.mkdir(parents=True, exist_ok=True)

    # 创建目标目录结构
    for split in ['train', 'val']:
        (target_base / split).mkdir(exist_ok=True)

    # 获取分类映射
    class_mapping = create_class_mapping()

    # 统计信息
    total_images = 0
    labels_found = 0
    labels_missing = 0
    missing_labels = []

    # 处理图片和标注
    print("📂 正在智能重组数据集...")
    for src_dir_name in os.listdir(image_source_base):
        src_dir = image_source_base / src_dir_name

        if src_dir.is_dir() and src_dir_name in class_mapping:
            mapping_info = class_mapping[src_dir_name]
            class_name = mapping_info['target_class']
            label_dir = mapping_info['label_dir']

            # 获取图片文件（支持多格式）
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPEG']
            all_files = [f for f in src_dir.iterdir() if f.suffix.lower() in image_extensions]
            random.shuffle(all_files)
            total_images += len(all_files)

            print(f"处理类别: {src_dir_name} -> {class_name} ({len(all_files)} 张图片)")

            # 计算分割点
            split_idx = int(len(all_files) * (1 - val_ratio))

            # 创建分类目录
            for split in ['train', 'val']:
                (target_base / split / class_name).mkdir(exist_ok=True, parents=True)

            # 复制图片文件
            for i, img_path in enumerate(all_files):
                split = 'train' if i < split_idx else 'val'

                # 文件名修复
                fixed_name = repair_filename(img_path.name)
                target_img_path = target_base / split / class_name / fixed_name

                # 复制并保留元数据
                shutil.copy2(str(img_path), str(target_img_path))

                # 智能标注匹配
                label_found = False
                stem = Path(fixed_name).stem

                # 尝试多种匹配模式
                for pattern in [
                    stem + '.txt',
                    stem + '_0.txt',
                    stem.split('（')[0] + '.txt',
                    stem.split('(')[0] + '.txt',
                    re.sub(r'\d+$', '', stem) + '.txt',  # 去除末尾数字
                    mapping_info['clean_name'] + '.txt'
                ]:
                    for label_ext in ['.txt', '.TXT']:
                        label_path = label_source_base / label_dir / (pattern + label_ext)
                        if label_path.exists():
                            # 修复标签文件名并复制
                            fixed_label = repair_filename(label_path.name)
                            shutil.copy2(
                                str(label_path),
                                str(target_img_path.with_name(fixed_label))
                            )
                            labels_found += 1
                            label_found = True
                            break
                    if label_found:
                        break

                if not label_found:
                    missing_labels.append({
                        'class': class_name,
                        'image': fixed_name,
                        'expected_label': str(label_source_base / label_dir / (stem + '.txt'))
                    })
                    labels_missing += 1

    # 生成增强型data.yaml
    (target_base / 'data.yaml').write_text(
        f"train: {str(target_base / 'train')}\n"
        f"val: {str(target_base / 'val')}\n"
        f"nc: {len(class_mapping)}\n"
        f"names: {[v['target_class'] for v in class_mapping.values()]}\n"
        f"path: {str(target_base)}\n"
    )

    # 生成诊断报告
    report = f"""✅ 数据集重组完成！
总处理图片数: {total_images}
找到的标签文件: {labels_found}
缺失的标签文件: {labels_missing}

数据集路径: {str(target_base)}
配置文件: {str(target_base / 'data.yaml')}
"""
    if missing_labels:
        report += "\n⚠️ 前10个缺失标签示例："
        for i, m in enumerate(missing_labels[:10]):
            report += f"\n{i + 1}. 类别: {m['class']}, 图片: {m['image']}, 预期标签路径: {m['expected_label']}"

    print(report)
    return target_base


if __name__ == "__main__":
    reorganize_dataset(val_ratio=0.2)
    print("\n💡 推荐操作：")
    print("1. 检查生成的data.yaml文件")
    print("2. 补充缺失的标注文件")
    print("3. 运行验证命令：yolo task=detect mode=validate model=yolov8n.pt data=data.yaml")