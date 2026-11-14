import json
import os
from pathlib import Path
import shutil

import chardet

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



def clean_yolo_labels(label_dir, mapping_file, backup_suffix="_original"):
    """
    清洗YOLO标签文件（增强编码兼容性版）
    """
    # 加载类别映射
    with open(mapping_file, 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)

    # 遍历处理每个标签文件
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith('.txt'):
                label_path = os.path.join(root, file)
                backup_path = f"{label_path}{backup_suffix}"

                # 创建备份目录
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)

                # 自动检测文件编码
                try:
                    with open(label_path, 'rb') as f:
                        rawdata = f.read(10000)
                    encoding = chardet.detect(rawdata)['encoding'] or 'utf-8'
                except:
                    encoding = 'utf-8'

                # 创建备份
                if os.path.exists(label_path):
                    shutil.copy2(label_path, backup_path)

                # 处理标签文件
                try:
                    with open(backup_path, 'r', encoding=encoding, errors='ignore') as f_in:
                        lines = f_in.readlines()
                except Exception as e:
                    print(f"⚠️ 无法读取文件: {backup_path} - {str(e)}")
                    continue

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue  # 跳过无效行

                    # 尝试匹配中英文类别名
                    class_name = ' '.join(parts[:-4])
                    if class_name in class_mapping:
                        class_id = class_mapping[class_name]
                        coords = parts[-4:]
                        new_lines.append(f"{class_id} {' '.join(coords)}\n")
                    elif parts[0].isdigit():
                        # 保留原有数字索引格式
                        new_lines.append(line)

                # 写入清洗后的文件
                if new_lines:
                    with open(label_path, 'w', encoding='utf-8') as f_out:
                        f_out.writelines(new_lines)

# 使用示例
clean_yolo_labels(
    label_dir=str(LABEL_DIR),
    mapping_file=str(CLASSES_FILE)
)