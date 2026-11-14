# -*- coding: utf-8 -*-
import os
import sys
import torch
import traceback
import yaml
import locale
import shutil
import re
import time
import logging
import cv2
from pathlib import Path
from ultralytics import YOLO
import psutil  # 用于监控系统资源

# 配置日志级别
logging.getLogger("ultralytics").setLevel(logging.WARNING)


def setup_environment():
    """环境配置与验证 - Windows CPU优化"""
    # 设置本地化环境
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        locale.setlocale(locale.LC_ALL, 'C')

    os.environ["PYTHONUTF8"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())  # 使用所有CPU核心
    os.environ["KMP_BLOCKTIME"] = "0"  # 减少线程切换延迟

    # Windows CPU优化设置
    os.environ["DNNL_ENABLE_PRIMITIVE_CACHE"] = "1"
    os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX2"  # 改为兼容Windows的AVX2设置

    # 设备配置
    print("\n⚠️ 警告: 将使用多核CPU模式")
    device = torch.device('cpu')
    # 设置CPU线程数
    torch.set_num_threads(os.cpu_count())
    os.environ["CU极速赛车开奖网_VISIBLE_DEVICES"] = "-1"

    # Windows CPU优化设置
    torch.backends.quantized.engine = 'fbgemm'  # Windows兼容的量化引擎
    if hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True

    # 打印硬件信息
    print(f"\n💻 硬件信息:")
    print(f"  处理器: 12th Gen Intel(R) Core(TM) i5-1240P")
    print(f"  核心数: {os.cpu_count()} (物理+逻辑)")
    print(f"  当前设备: {device}")

    return device


def validate_dataset(data_config):
    """数据集完整性验证（优化版）"""
    print("\n🔍 验证数据集完整性...")

    # 获取基础路径
    base_path = Path(data_config['path'])

    # 创建标签备份目录
    backup_dir = base_path / "labels_backup"
    backup_dir.mkdir(exist_ok=True)

    # 确保labels目录存在后再备份
    labels_dir = base_path / "labels"
    if labels_dir.exists():
        shutil.copytree(labels_dir, backup_dir / "labels", dirs_exist_ok=True)
        print(f"✅ 标签文件已备份至: {backup_dir}")
    else:
        print(f"⚠️ 警告: 标签目录不存在: {labels_dir}")

    # 验证路径存在
    train_path = base_path / data_config['train']
    val_path = base_path / data_config['val']

    if not base_path.exists():
        raise FileNotFoundError(f"数据集根路径不存在: {base_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"训练集路径不存在: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"验证集路径不存在: {val_path}")

    # 验证样本数量
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    train_images = list(train_path.glob('*.*'))
    val_images = list(val_path.glob('*.*'))

    # 过滤非图像文件
    train_images = [f for f in train_images if f.suffix.lower() in image_extensions]
    val_images = [f for f in val_images if f.suffix.lower() in image_extensions]

    if not train_images:
        raise FileNotFoundError(f"训练集中未找到图片文件: {train_path}")
    if not val_images:
        raise FileNotFoundError(f"验证集中未找到图片文件: {val_path}")

    print(f"  训练集图片: {len(train_images)} 张")
    print(f"  验证集图片: {len(val_images)} 张")

    # 验证类别数量
    nc = data_config['nc']
    names = data_config['names']

    if nc != len(names):
        raise ValueError(f"类别数量不匹配: 配置({nc}) vs 名称列表({len(names)})")
    else:
        print(f"✅ 类别数量一致: {nc} 类")

    # 增强的标注文件验证
    print("\n🔍 检查标注文件完整性...")
    missing_labels = []

    # 准备标签目录路径 - 确保目录存在
    train_label_dir = base_path / 'labels' / 'train'
    val_label_dir = base_path / 'labels' / 'val'

    # 创建标签目录如果不存在 - 修复了这里的拼写错误
    train_label_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)

    # 检查训练集标注
    for img_file in train_images:
        # 使用更安全的文件名处理
        normalized_stem = re.sub(r'[^\w\.-]', '', img_file.stem)
        label_file = train_label_dir / f"{normalized_stem}.txt"

        if not label_file.exists():
            missing_labels.append(img_file)
            if len(missing_labels) < 5:  # 只显示前5个缺失文件
                print(f"  ⚠️ 训练集标注文件缺失: {img_file.name} (预期: {label_file.name})")

    # 检查验证集标注 - 同上处理
    for img_file in val_images:
        normalized_stem = re.sub(r'[^\w\.-]', '', img_file.stem)
        label_file = val_label_dir / f"{normalized_stem}.txt"

        if not label_file.exists():
            missing_labels.append(img_file)
            if len(missing_labels) < 5:
                print(f"  ⚠️ 验证集标注文件缺失: {img_file.name} (预期: {label_file.name})")

    if missing_labels:
        print(f"\n❌ 共发现 {len(missing_labels)} 个图片缺失标注文件")

        # 创建空标签作为快速解决方案
        print("💡 为缺失文件创建空标签...")
        created_count = 0

        for img_file in missing_labels:
            if img_file.parent.name == 'train':
                label_dir = train_label_dir
            else:
                label_dir = val_label_dir

            normalized_stem = re.sub(r'[^\w\.-]', '', img_file.stem)
            label_file = label_dir / f"{normalized_stem}.txt"

            # 创建空标签文件
            if not label_file.exists():
                with open(label_file, 'w') as f:
                    f.write("")
                created_count += 1

                if created_count < 10:  # 只显示前10个创建
                    print(f"  已创建空标签: {label_file.name}")

        print(f"✅ 已为 {created_count} 个缺失文件创建空标签")

        # 生成报告
        repair_report_path = base_path / "missing_labels_report.csv"
        with open(repair_report_path, "w", encoding="utf-8") as f:
            f.write("图片文件,标签文件,状态\n")
            for img_file in missing_labels:
                f.write(f"{img_file.name},{normalized_stem}.txt,创建空标签\n")

        print(f"📝 缺失文件报告已保存至: {repair_report_path}")
        print("⚠️ 注意: 空标签表示图片中没有检测目标")
        print("      如果图片中应有目标，请手动添加标注")

        # 返回而不是重新验证
        print("✅ 继续训练流程...")
        return

    print("✅ 所有图片都有对应的标注文件")
    print("✅ 数据集验证通过")

    # 数据预加载（可选）
    print("\n🔍 预加载数据集样本...")
    for img_file in train_images[:50]:
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"⚠️ 无法读取图片: {img_file}")
            del img
        except Exception as e:
            print(f"⚠️ 预加载出错: {img_file} - {str(e)}")
    print("✅ 数据集预加载完成")


def load_model(model_cfg="configs/yolov11.yaml"):
    """加载自定义YOLOv11模型（优化版）"""
    print(f"\n🔍 加载模型: {model_cfg}")

    # 检查模型配置文件是否存在
    if not Path(model_cfg).exists():
        raise FileNotFoundError(f"模型配置文件不存在: {model_cfg}")

    # 尝试读取配置文件内容（用于错误诊断）
    try:
        with open(model_cfg, 'r', encoding='utf-8') as f:
            cfg_content = f.read()
    except Exception as e:
        print(f"⚠️ 无法读取配置文件: {str(e)}")
        cfg_content = ""

    try:
        # 初始化模型（禁用详细日志）
        model = YOLO(model_cfg, verbose=False)
        print(f"✅ 模型加载成功: {model_cfg}")

        # 添加尺寸验证
        try:
            print("🔍 验证模型尺寸兼容性...")
            # 创建虚拟输入（使用更小的尺寸）
            dummy_input = torch.randn(1, 3, 128, 128)  # 使用128x128分辨率
            # 测试前向传播
            with torch.no_grad():
                outputs = model(dummy_input)
            print("✅ 模型尺寸兼容性验证通过")
        except Exception as e:
            print(f"❌ 模型尺寸兼容性验证失败: {str(e)}")
            print("💡 建议: 检查模型配置中的通道数和层索引")
            raise

        return model
    except Exception as e:
        print(f"\n❌ 模型加载失败: {str(e)}")
        print("💡 建议操作:")
        print("  1. 检查模型配置文件中的通道数设置")
        print("  2. 确保所有Concat层引用的索引正确")
        print("  3. 验证特征图尺寸匹配")

        # 提供详细的调试信息
        print("\n🔧 调试信息:")
        print(f"  当前配置文件: {model_cfg}")
        print(f"  配置内容:\n{cfg_content}")

        # 尝试创建最小可用配置
        minimal_cfg = Path("configs/minimal_yolov8.yaml")
        with open(minimal_cfg, 'w', encoding='utf-8') as f:
            f.write("""# 最小化YOLOv8配置
nc: 24
depth_multiple: 0.25
width_multiple: 0.20

backbone:
  - [-1, 1, Conv, [32, 3, 2]]
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3, [256]]
  - [-1, 1, SPPF, [256, 5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3, [128]]

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3, [64]]

  - [-1, 1, Conv, [64, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3, [128]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 5], 1, Concat, [1]]
  - [-1, 2, C3, [256]]

  - [[7, 10, 13], 1, Detect, [nc]]
""")
        print(f"  已创建最小化配置文件: {minimal_cfg}")
        print("  请尝试使用此配置: model = YOLO('configs/minimal_yolov8.yaml')")
        raise


def print_training_progress(epoch, total_epochs, metrics):
    """打印训练进度信息（精简版）"""
    print(f"\nEpoch {epoch + 1}/{total_epochs} 进度:")
    print(f"  定位损失 (box_loss): {metrics.get('train/box_loss', 0):.3f}")
    print(f"  分类损失 (cls_loss): {metrics.get('train/cls_loss', 0):.3f}")

    if 'metrics/mAP50' in metrics:
        print(f"  验证精度 (mAP50): {metrics['metrics/mAP50']:.4f}")
    if 'metrics/mAP50-95' in metrics:
        print(f"  验证精度 (mAP50-95): {metrics['metrics/mAP50-95']:.4f}")


def print_system_stats():
    """打印系统资源使用情况"""
    mem = psutil.virtual_memory()
    print(f"内存使用: {mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB ({mem.percent}%)")
    cpu_percent = psutil.cpu_percent()
    print(f"CPU使用率: {cpu_percent}%")


def main():
    try:
        # 环境配置
        print("=" * 60)
        print("🚀 植物叶片病害检测模型训练 (YOLOv11 Windows CPU优化版)")
        print("=" * 60)
        device = setup_environment()

        # 加载数据集配置
        data_yaml_path = os.environ.get("PLANT_DATA_YAML", "configs/plant_disease_39.yaml")
        print(f"\n📂 加载数据集配置: {data_yaml_path}")

        # 检查配置文件是否存在
        if not Path(data_yaml_path).exists():
            raise FileNotFoundError(f"数据集配置文件不存在: {data_yaml_path}")

        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)

        # 检查数据集路径是否存在
        data_path = Path(data_config['path'])
        if not data_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {data_path}")

        print(f"✅ 数据集路径: {data_path.resolve()}")

        # 验证数据集
        validate_dataset(data_config)

        # 初始化模型
        model_cfg = "configs/yolov11.yaml"
        model = load_model(model_cfg)

        # CPU优化训练参数配置 - 针对内存和速度优化
        train_params = {
            'data': str(data_yaml_path),
            'epochs': 20,
            'imgsz': 128,  # 降低分辨率以节省内存
            'device': 'cpu',
            'batch': 4,    # 减少批次大小以节省内存
            'workers': 0,   # Windows上设置为0
            'patience': 15,
            'project': "runs/train",
            'name': "plant_disease_yolov11_win_cpu_optimized",
            'exist_ok': True,
            'verbose': False,
            'save': True,
            'val': False,  # 训练期间禁用验证以加速
            'plots': False,
            'show': False,
            'save_period': 15,
            'optimizer': "SGD",
            'amp': False,
            'cos_lr': True,
            'seed': 42,
            'deterministic': True,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 2,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'cache': False,  # 禁用缓存
            'single_cls': False,
            'rect': True,
        }

        # 显示训练参数
        print("\n⚙️ CPU优化训练参数配置 (针对内存优化):")
        print("  关键优化点:")
        print("  - 降低图像分辨率(imgsz=128)减少内存占用")
        print("  - 减少批次大小(batch=4)缓解内存压力")
        print("  - 数据加载工作进程(workers=0)避免Windows多进程问题")
        print("  - 禁用磁盘缓存(cache=False)避免磁盘I/O")
        print("  - 训练期间禁用验证(val=False)加速训练过程")

        # 打印所有参数
        for key, value in train_params.items():
            print(f"  {key}: {value}")

        # 启动训练
        print("\n🚀 开始训练YOLOv11模型...")
        print(f"日志将保存至: runs\\train\\{train_params['name']}")
        print("训练中可按 Ctrl+C 安全停止")
        print("注意: 由于数据集较大，训练可能需要较长时间，请耐心等待...")

        # 训练开始时间
        start_time = time.time()

        try:
            # 启动训练
            results = model.train(**train_params)
        except KeyboardInterrupt:
            print("\n🛑 训练被用户中断")
            # 尝试保存最后状态
            last_pt = Path(f"runs/train/{train_params['name']}/weights/last.pt")
            if last_pt.exists():
                final_model = Path("plant_disease_yolov11_interrupted.pt")
                shutil.copy(last_pt, final_model)
                print(f"📦 已保存中断时的模型: {final_model}")
            return 1

        # 训练结束时间
        total_time = time.time() - start_time
        print(f"\n⏱️ 总训练时间: {total_time / 3600:.2f} 小时")
        print(f"⏱️ 平均每epoch时间: {total_time / train_params['epochs']:.2f} 秒")

        # 保存最终模型
        best_model = Path(f"runs/train/{train_params['name']}/weights/best.pt")
        if best_model.exists():
            final_model = Path("plant_disease_yolov11_best.pt")
            shutil.copy(best_model, final_model)
            print(f"\n🎉 训练完成! 最佳模型已保存至: {final_model}")

        # 导出ONNX模型
        try:
            model.export(format="onnx")
            print("✅ 模型已导出为ONNX格式")
        except Exception as e:
            print(f"⚠️ ONNX导出失败: {str(e)}")

        # 训练完成后进行验证 - 优化版
        print("\n🔍 开始最终模型验证...")
        try:
            # 优化验证参数
            val_params = {
                'data': str(data_yaml_path),
                'batch': 8,  # 使用较小的批次
                'imgsz': 128,
                'workers': 0,   # Windows上设置为0
                'device': 'cpu',
                'plots': False,
                'save_json': False,
                'save_hybrid': False,
                'conf': 0.25,
                'iou': 0.45,
                'half': False,
                'rect': True
            }

            val_results = model.val(**val_params)

            # 打印结果
            if 'metrics/mAP50-95(B)' in val_results.results_dict:
                print(f"验证结果 mAP50-95: {val_results.results_dict['metrics/mAP50-95(B)']:.4f}")
            else:
                print("⚠️ 未找到mAP50-95指标")

            if 'metrics/mAP50(B)' in val_results.results_dict:
                print(f"验证结果 mAP50: {val_results.results_dict['metrics/mAP50(B)']:.4f}")
            else:
                print("⚠️ 未找到mAP50指标")

        except Exception as e:
            print(f"⚠️ 验证过程中发生错误: {str(e)}")
            print("💡 建议: 尝试减小批次大小或禁用某些验证选项")

        return 0

    except Exception as e:
        print(f"\n❌ 错误发生: {str(e)}")
        print("\n🔍 错误追踪:")
        traceback.print_exc()

        # 错误诊断
        print("\n🛠️ 诊断建议:")
        if "标注文件缺失" in str(e):
            print("  1. 确保所有图片都有对应的标注文件")
            print("  2. 检查文件名中的括号格式（中文括号 vs 英文括号）")
            print("  3. 运行数据准备脚本重新生成数据集")
            print("  4. 检查标注文件是否在正确的类别目录下")

        elif "CUDA" in str(e):
            print("  1. 检查CUDA和cuDNN安装")
            print("  2. 验证PyTorch与CUDA版本兼容性")
            print("  3. 尝试设置device='cpu'")

        elif "generate_data.yaml" in str(e):
            print("  1. 检查generate_data.yaml文件格式")
            print("  2. 验证路径是否正确")
            print("  3. 确保数据集目录包含图片和标注")

        elif "out of memory" in str(e):
            print("  1. 减少batch大小")
            print("  2. 降低图像分辨率(imgsz)")
            print("  3. 使用更小的模型架构")

        elif "No labels found" in str(e):
            print("  1. 验证标注文件是否存在")
            print("  2. 检查标注文件格式是否符合YOLO标准")
            print("  3. 确保标注文件与图片同名且在同一目录")

        elif "nc" in str(e) or "number of classes" in str(e):
            print("  1. 检查generate_yolov11.yaml中的nc值是否与generate_data.yaml一致")
            print("  2. 验证generate_data.yaml中的names列表数量与nc匹配")
            print(
                f"  3. 当前配置: generate_yolov11.yaml nc=24, generate_data.yaml nc={data_config['nc'] if 'data_config' in locals() else '未知'}")

        else:
            print("  1. 检查数据集路径是否包含特殊字符")
            print("  2. 尝试使用绝对路径")
            print("  3. 降低训练参数复杂度")

        return 1


if __name__ == "__main__":
    # 启动训练并获取退出代码
    exit_code = main()

    # 显示完成消息
    print("\n" + "=" * 60)
    print("🏁 训练脚本执行完成")
    print("=" * 60)

    # 根据退出代码暂停
    if exit_code != 0:
        print("⚠️ 训练过程中遇到错误，请查看上方日志")
    else:
        print("✅ 训练成功完成!")

    # 在Windows上暂停以便查看结果
    if sys.platform == "win32":
        input("\n按Enter键退出...")

    sys.exit(exit_code)