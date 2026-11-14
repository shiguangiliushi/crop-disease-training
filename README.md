# 植物叶片病害检测 —— 训练代码（YOLOv11 / Ultralytics）

> 这是**训练侧**代码，与线上系统 [shiguangliushi/crop-disease-system](https://github.com/shiguangliushi/crop-disease-system)（Django + YOLO 推理）分离：
> 那边只负责推理（Django + YOLO 权重），这边只负责"数据 → 模型"。
> 数据集与权重都不入库（体积 + 许可原因），按本文件说明自行准备。
>
> **线上模型实测**（39 类，val 4528 张，CPU）：Precision **87.8%** / Recall **89.6%** / mAP50 **91.5%**。
> 详见 [docs/评估方法.md](docs/评估方法.md) —— 那里也写清了这份数据集的检测其实是分类这个关键前提。

## 一、这套代码做了什么

```
原始数据（按病害分文件夹的图片 + YOLO 标签）
   │
   ├─ src/clean_labels.py        清洗标签（含 GBK/UTF-8 混杂编码修复）
   ├─ src/split_dataset.py       按比例切分 train/val，生成 datasets/
   ├─ src/build_data_config.py   根据切分结果生成 __data.yaml
   └─ src/train.py               YOLOv11 训练（CPU 优化参数）
          │
          └─ runs/detect/.../weights/best.pt  → 交给线上系统的 ai_service
```

## 二、38/39 类与 24 类：这里有两次实验，别搞混

| 实验 | 类别数 | 配置 | 权重 | 状态 |
|---|---|---|---|---|
| **A（线上采用）** | **39** | `configs/plant_disease_39.yaml` | `best.pt` 约 6 MB | 后端 `ai_service.class_names` 就是这 39 条 |
| B（早期对照） | 24 | `configs/data.generated.yaml` | 1.39 MB | 只跑过一次对照，未上线 |

`configs/plant_disease_39.yaml` 来自数据集发布包，**与线上模型一致**；`data.generated.yaml` 是训练脚本自动生成的 24 类配置。
默认用 39 类那个。

## 三、快速开始

### 1）准备环境

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt                   # ultralytics==8.3.143
```

### 2）准备数据

数据**不入库**。目录结构需为（Ultralytics 检测任务标准布局）：

```
data/
├── image/数据集1/植物叶片病害原始数据集/<病害名>/*.jpg   # 原始图片（按病害分文件夹）
└── labels/*.txt                                          # 对应的 YOLO 标签
datasets/                                                 # 切分后自动生成
├── images/train/*.jpg   images/val/*.jpg
└── labels/train/*.txt   labels/val/*.txt
```

路径可用环境变量覆盖（见 `src/*.py` 顶部 `DATA_DIR` 区）：

```bash
set PLANT_DATA_DIR=D:\my-data          # 原始数据根目录
set PLANT_OUTPUT_DIR=D:\my-datasets    # 切分输出目录
```

### 3）切分 → 生成配置 → 训练

```bash
python src/clean_labels.py       # 可选：先清洗标签
python src/split_dataset.py      # 生成 datasets/{train,val}
python src/build_data_config.py  # 生成 __data.yaml
python src/train.py              # 训练（默认读 configs/plant_disease_39.yaml）
```

> ⚠️ **`path:` 一定要填绝对路径**。Ultralytics 会把相对的 `path` 按它自己的
> `datasets_dir` 设置解析（不是按 yaml 所在目录），相对路径是这里最常见的坑。

### 4）在 Colab 上跑（零本地环境）

`docs/Google_Colab_训练指南.md` 是当时在 Colab 上跑通的步骤记录，
适合没有 GPU 的机器先跑通一遍。

## 四、产物

```
runs/detect/train/weights/best.pt     # 最优权重
runs/detect/train/results.png         # 训练曲线
runs/detect/train/confusion_matrix.png
```

权重不进 git（`.gitignore` 已排除 `*.pt`）。要发布就放到 GitHub Release 附件或网盘，README 里给链接。

## 五、数据来源与许可

- 数据集：Kaggle *Plant Leaves Diseases Dataset with Augmentation*（PlantVillage 派生）；
- **不要再分发数据本身**（许可不明），只写"从哪下载、如何摆放"；
- 本项目只保留**类别映射**（`configs/classes.json`、`configs/class_mapping.txt`）与代码。

## 六、实测环境

| 项 | 值 |
|---|---|
| Ultralytics | 8.3.143 |
| 训练设备 | Windows + CPU（`imgsz=128, batch=4, epochs=20`，为省内存调低） |
| 数据规模 | 原始 8.7 万文件；切分后 train 8.2 万 / val 3.9 万（含增强） |
| 线上权重 | 约 6 MB（`best.pt`，SHA256 `F86C031B…`） |
| 线上指标 | P 0.8777 / R 0.8959 / mAP50 0.9149（val 全量 4528 张，CPU） |

## 七、目录结构

```
crop-disease-training/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── plant_disease_39.yaml     # 39 类（与线上一致，默认）
│   ├── data.generated.yaml       # 24 类（早期对照）
│   ├── yolov11.yaml              # 模型结构
│   ├── minimal_yolov8.yaml       # 最小化结构（内存不足时的退路）
│   ├── classes.json              # 中英别名映射（120 条）
│   └── class_mapping.txt
├── docs/
│   ├── 评估方法.md               # ⭐ 真实指标 + 复现命令 + 这些数字该怎么看
│   ├── pr_curve.png
│   └── Google_Colab_训练指南.md
└── src/
    ├── clean_labels.py
    ├── split_dataset.py
    ├── build_data_config.py
    └── train.py
```
