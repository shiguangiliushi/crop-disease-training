# -*- coding: utf-8 -*-
import os
import shutil
import random
import re
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm

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
    """创建完整的分类映射表"""
    return {
        # 苹果类病害
        'Apple___Apple_scab（苹果-苹果疮痂病）': {
            'target_class': 'Apple_Apple_scab',
            'label_dir': Path('高频病害(4种)') / '苹果疮痂病',
            'name_pattern': lambda name: re.sub(r'Apple___', '', name)
        },
        'Apple___Black_rot（苹果-黑腐病）': {
            'target_class': 'Apple_Black_rot',
            'label_dir': Path('罕见病害(5种)') / '苹果黑腐病',
            'name_pattern': lambda name: re.sub(r'Apple___', '', name)
        },
        'Apple___Cedar_apple_rust（苹果-雪松苹果锈病）': {
            'target_class': 'Apple_Cedar_apple_rust',
            'label_dir': Path('罕见病害(5种)') / '苹果雪松苹果锈病',
            'name_pattern': lambda name: re.sub(r'Apple___', '', name)
        },
        'Apple___healthy（健康苹果）': {
            'target_class': 'Apple_healthy',
            'label_dir': Path('健康类(12种)') / '健康苹果',
            'name_pattern': lambda name: re.sub(r'Apple___', '', name)
        },

        # 蓝莓类
        'Blueberry___healthy（健康蓝莓）': {
            'target_class': 'Blueberry_healthy',
            'label_dir': Path('健康类(12种)') / '健康蓝莓',
            'name_pattern': lambda name: name.replace('___', '_')
        },

        # 樱桃类
        'Cherry___healthy（健康樱桃）': {
            'target_class': 'Cherry_healthy',
            'label_dir': Path('健康类(12种)') / '健康樱桃',
            'name_pattern': lambda name: name.replace('___', '_')
        },

        # 玉米类病害
        'Corn___Cercospora_leaf_spot Gray_leaf_spot（玉米尾孢菌叶斑病灰斑病）': {
            'target_class': 'Corn_Cercospora_leaf_spot',
            'label_dir': Path('罕见病害(5种)') / '玉米尾孢菌叶斑病灰斑病',
            'name_pattern': lambda name: name.replace('___', '_').replace(' Gray_leaf_spot', '')
        },
        'Corn___healthy（健康玉米）': {
            'target_class': 'Corn_healthy',
            'label_dir': Path('健康类(12种)') / '健康玉米',
            'name_pattern': lambda name: name.replace('___', '_')
        },
        'Corn___Northern_Leaf_Blight（玉米大斑病）': {
            'target_class': 'Corn_Northern_Leaf_Blight',
            'label_dir': Path('普通病害(3种)') / '玉米大斑病',
            'name_pattern': lambda name: name.replace('___', '_')
        },

        # 葡萄类病害
        'Grape___Black_rot（葡萄黑腐病）': {
            'target_class': 'Grape_Black_rot',
            'label_dir': Path('普通病害(3种)') / '葡萄黑腐病',
            'name_pattern': lambda name: name.replace('___', '_')
        },
        'Grape___healthy(健康葡萄)': {
            'target_class': 'Grape_healthy',
            'label_dir': Path('健康类(12种)') / '健康葡萄',
            'name_pattern': lambda name: name.replace('___', '_')
        },

        # 桃类
        'Peach___healthy（健康桃子）': {
            'target_class': 'Peach_healthy',
            'label_dir': Path('健康类(12种)') / '健康桃子',
            'name_pattern': lambda name: name.replace('___', '_')
        },

        # 辣椒类
        'Pepper,_bell___healthy（健康甜椒）': {
            'target_class': 'Pepper_bell_healthy',
            'label_dir': Path('健康类(12种)') / '健康甜椒',
            'name_pattern': lambda name: name.replace('Pepper,_bell___', 'healthy_sweet_pepper')
        },

        # 马铃薯类病害
        'Potato___Early_blight（马铃薯早疫病）': {
            'target_class': 'Potato_Early_blight',
            'label_dir': Path('普通病害(3种)') / '马铃薯早疫病',
            'name_pattern': lambda name: name.replace('___', '_')
        },
        'Potato___healthy（健康马铃薯）': {
            'target_class': 'Potato_healthy',
            'label_dir': Path('健康类(12种)') / '健康马铃薯',
            'name_pattern': lambda name: name.replace('___', '_')
        },
        'Potato___Late_blight（马铃薯晚疫病）': {
            'target_class': 'Potato_Late_blight',
            'label_dir': Path('高频病害(4种)') / '马铃薯晚疫病',
            'name_pattern': lambda name: name.replace('___', '_')
        },

        # 树莓类
        'Raspberry___healthy（健康树莓）': {
            'target_class': 'Raspberry_healthy',
            'label_dir': Path('健康类(12种)') / '健康树莓',
            'name_pattern': lambda name: name.replace('___', '_')
        },

        # 大豆类
        'Soybean___healthy（健康大豆）': {
            'target_class': 'Soybean_healthy',
            'label_dir': Path('健康类(12种)') / '健康大豆',
            'name_pattern': lambda name: name.replace('___', '_')
        },

        # 南瓜类病害
        'Squash___Powdery_mildew（南瓜白粉病）': {
            'target_class': 'Squash_Powdery_mildew',
            'label_dir': Path('罕见病害(5种)') / '南瓜白粉病',
            'name_pattern': lambda name: name.replace('___', '_')
        },

        # 草莓类病害
        'Strawberry___healthy（健康草莓）': {
            'target_class': 'Strawberry_healthy',
            'label_dir': Path('健康类(12种)') / '健康草莓',
            'name_pattern': lambda name: name.replace('___', '_')
        },
        'Strawberry___Leaf_scorch（草莓叶焦病）': {
            'target_class': 'Strawberry_Leaf_scorch',
            'label_dir': Path('罕见病害(5种)') / '草莓叶焦病',
            'name_pattern': lambda name: name.replace('___', '_')
        },

        # 番茄类病害
        'Tomato___Early_blight（番茄早疫病）': {
            'target_class': 'Tomato_Early_blight',
            'label_dir': Path('高频病害(4种)') / '番茄早疫病',
            'name_pattern': lambda name: name.replace('___', '_')
        },
        'Tomato___healthy（健康番茄）': {
            'target_class': 'Tomato_healthy',
            'label_dir': Path('健康类(12种)') / '健康番茄',
            'name_pattern': lambda name: name.replace('___', '_')
        },
        'Tomato___Late_blight（番茄晚疫病）': {
            'target_class': 'Tomato_Late_blight',
            'label_dir': Path('高频病害(4种)') / '番茄晚疫病',
            'name_pattern': lambda name: name.replace('___', '_')
        }
    }


def normalize_filename(name):
    """规范化文件名：提取数字标识"""
    # 提取所有数字序列
    numbers = re.findall(r'\d+', name)
    # 返回最后一个数字序列（通常是最有意义的）
    return numbers[-1] if numbers else name


def build_label_index(label_dir):
    """构建标签文件索引（数字->文件路径），处理备份文件"""
    index = {}

    # 先处理标准.txt文件
    for label_file in label_dir.glob('*.txt'):
        # 跳过特殊文件
        if label_file.name == 'classes.txt' or label_file.name.endswith('_original'):
            continue

        # 提取文件中的数字标识
        numbers = re.findall(r'\d+', label_file.stem)
        if numbers:
            num_key = numbers[-1]  # 使用最后一个数字作为键
            # 处理重复数字（保留第一个找到的）
            if num_key not in index:
                index[num_key] = label_file

    # 处理备份文件（.txt_original）
    for backup_file in label_dir.glob('*.txt_original'):
        # 提取文件中的数字标识
        numbers = re.findall(r'\d+', backup_file.stem)
        if numbers:
            num_key = numbers[-1]
            # 只有当标准文件不存在时，才使用备份文件
            if num_key not in index:
                # 创建对应的标准文件名
                standard_file = backup_file.with_suffix('.txt').with_stem(backup_file.stem)
                index[num_key] = standard_file

    return index


def convert_backup_files(label_source_base, class_mapping):
    """转换备份标签文件为标准格式"""
    print("\n🔄 转换备份标签文件...")
    converted_count = 0

    for class_info in class_mapping.values():
        label_dir = label_source_base / class_info['label_dir']
        if not label_dir.exists():
            continue

        for backup_file in label_dir.glob('*.txt_original'):
            # 创建对应的标准文件名
            standard_name = backup_file.stem + '.txt'
            standard_file = backup_file.with_name(standard_name)

            # 如果标准文件不存在，则重命名备份文件
            if not standard_file.exists():
                backup_file.rename(standard_file)
                converted_count += 1
                print(f"  已转换: {backup_file.name} -> {standard_file.name}")

    print(f"✅ 已转换 {converted_count} 个备份标签文件为标准格式")
    return converted_count


def reorganize_dataset(val_ratio=0.2):
    # 配置路径
    image_source_base = IMAGE_DIR / '数据集1' / '植物叶片病害原始数据集'
    label_source_base = LABEL_DIR
    target_base = OUTPUT_DIR

    # 确保目标基础目录存在
    target_base.mkdir(parents=True, exist_ok=True)

    # 创建YOLO标准目录结构
    for split in ['train', 'val']:
        (target_base / 'images' / split).mkdir(parents=True, exist_ok=True)
        (target_base / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 获取分类映射
    class_mapping = create_class_mapping()

    # 转换备份文件
    convert_backup_files(label_source_base, class_mapping)

    # 统计信息
    total_images = 0
    labels_found = 0
    labels_missing = 0
    missing_labels = []
    match_report = []

    # 处理图片和标注
    print("\n📂 正在处理数据集...")
    for src_dir_name in os.listdir(image_source_base):
        src_dir = image_source_base / src_dir_name

        if src_dir.is_dir() and src_dir_name in class_mapping:
            mapping_info = class_mapping[src_dir_name]
            class_name = mapping_info['target_class']
            label_dir_path = label_source_base / mapping_info['label_dir']

            # 获取图片文件（仅限图片扩展名）
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            all_files = [f for f in src_dir.iterdir() if f.suffix.lower() in image_extensions]
            random.shuffle(all_files)  # 随机打乱顺序
            total_images += len(all_files)

            # 构建标签索引（处理备份文件）
            label_index = {}
            if label_dir_path.exists():
                label_index = build_label_index(label_dir_path)

            print(f"  处理: {src_dir_name} -> {class_name} (图片: {len(all_files)}, 标签: {len(label_index)})")

            # 计算分割点
            split_idx = int(len(all_files) * (1 - val_ratio))

            # 复制图片文件
            for i, img_path in enumerate(tqdm(all_files, desc=f"处理 {class_name}")):
                if i < split_idx:
                    split = 'train'
                else:
                    split = 'val'

                # 复制图片
                target_img_path = target_base / 'images' / split / img_path.name
                shutil.copy2(str(img_path), str(target_img_path))

                # 处理对应标注文件
                # 从图片文件名提取数字标识
                img_num = normalize_filename(img_path.stem)

                # 查找匹配的标签文件
                label_file = label_index.get(img_num)

                if label_file:
                    # 确保标签文件路径正确
                    if not label_file.exists():
                        # 检查是否存在.txt_original版本
                        backup_file = label_file.with_suffix('.txt_original')
                        if backup_file.exists():
                            label_file = backup_file

                    if label_file.exists():
                        target_label_path = target_base / 'labels' / split / f"{img_path.stem}.txt"
                        shutil.copy2(str(label_file), str(target_label_path))
                        labels_found += 1
                        match_report.append({
                            'class': class_name,
                            'image': img_path.name,
                            'label': label_file.name,
                            'status': 'MATCHED',
                            'image_num': img_num
                        })
                    else:
                        labels_missing += 1
                        match_report.append({
                            'class': class_name,
                            'image': img_path.name,
                            'label': 'NOT FOUND',
                            'status': 'MISSING',
                            'image_num': img_num
                        })
                else:
                    # 记录未匹配的图片
                    labels_missing += 1
                    missing_labels.append({
                        'class': class_name,
                        'image': str(img_path),
                        'image_num': img_num,
                        'available_labels': list(label_index.keys())[:5]  # 显示前5个可用标签
                    })
                    match_report.append({
                        'class': class_name,
                        'image': img_path.name,
                        'label': 'NOT FOUND',
                        'status': 'MISSING',
                        'image_num': img_num
                    })

    print("\n✅ 数据集重组完成！")
    print(f"总处理图片数: {total_images}")
    print(f"找到的标签文件: {labels_found}")
    print(f"缺失的标签文件: {labels_missing}")

    # 创建YOLO格式的数据配置文件
    dataset_config = {
        'path': str(target_base.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 24,
        'names': [
            "Apple_Apple_scab",
            "Apple_Black_rot",
            "Apple_Cedar_apple_rust",
            "Apple_healthy",
            "Blueberry_healthy",
            "Cherry_healthy",
            "Corn_Cercospora_leaf_spot",
            "Corn_healthy",
            "Corn_Northern_Leaf_Blight",
            "Grape_Black_rot",
            "Grape_healthy",
            "Peach_healthy",
            "Pepper_bell_healthy",
            "Potato_Early_blight",
            "Potato_healthy",
            "Potato_Late_blight",
            "Raspberry_healthy",
            "Soybean_healthy",
            "Squash_Powdery_mildew",
            "Strawberry_healthy",
            "Strawberry_Leaf_scorch",
            "Tomato_Early_blight",
            "Tomato_healthy",
            "Tomato_Late_blight"
        ]
    }

    # 保存配置文件
    config_path = target_base / "generate_data.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_config, f, allow_unicode=True)

    print(f"✅ 数据集配置文件已保存至: {config_path}")

    # 保存匹配报告
    report_path = target_base / "match_report.csv"
    pd.DataFrame(match_report).to_csv(report_path, index=False)
    print(f"📝 匹配报告已保存至: {report_path}")

    # 打印缺失标签的详细信息
    if missing_labels:
        print("\n⚠️ 缺失标签文件示例（最多显示10个）:")
        for i, missing in enumerate(missing_labels[:10]):
            print(f"{i + 1}. 类别: {missing['class']}")
            print(f"   图片: {Path(missing['image']).name}")
            print(f"   提取的数字: {missing['image_num']}")
            print(f"   可用标签数字: {missing['available_labels']}")

    print("\n请检查以下目录：")
    print(f"- 训练集图像: {target_base / 'images/train'}")
    print(f"- 训练集标签: {target_base / 'labels/train'}")
    print(f"- 验证集图像: {target_base / 'images/val'}")
    print(f"- 验证集标签: {target_base / 'labels/val'}")

    # 生成标签文件统计报告
    print("\n📊 标签文件统计报告:")
    stats = []
    for class_name, class_info in class_mapping.items():
        label_dir = label_source_base / class_info['label_dir']
        img_dir = image_source_base / class_name

        img_count = len(list(img_dir.glob('*.*'))) if img_dir.exists() else 0
        label_count = 0

        if label_dir.exists():
            # 计算标准标签文件
            txt_files = list(label_dir.glob('*.txt'))
            # 计算备份标签文件
            backup_files = list(label_dir.glob('*.txt_original'))
            # 排除特殊文件
            label_count = len([f for f in txt_files + backup_files
                               if f.name != 'classes.txt' and not f.name.endswith('_original')])

        stats.append({
            '类别': class_name,
            '图片数量': img_count,
            '标签数量': label_count,
            '匹配率': f"{label_count / img_count * 100:.1f}%" if img_count > 0 and label_count > 0 else "N/A"
        })

    stats_path = target_base / "label_stats.csv"
    pd.DataFrame(stats).to_csv(stats_path, index=False)
    print(f"📊 统计报告已保存至: {stats_path}")

    # 清理临时文件
    print("\n🧹 清理临时文件...")
    for root, dirs, files in os.walk(target_base):
        for file in files:
            if file.endswith('.txt_original'):
                os.remove(os.path.join(root, file))
    print("✅ 临时文件清理完成")


if __name__ == "__main__":
    reorganize_dataset(val_ratio=0.2)