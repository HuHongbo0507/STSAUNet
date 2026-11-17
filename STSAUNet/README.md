# STSAUNet

这是面向红外小目标分割的 STSAUNet 独立源码目录。以推荐的七帧配置为例，模型输入为 `[B,7,1,H,W]`，将帧和通道合并为 `[B,7,H,W]`，依次展开更新背景与稀疏目标。随后恢复时间维，以 `grid_sample` 实现可变形时序对齐，输出第 4 帧的分割 logits `[B,1,H,W]`。预备数据格式采用 256×256 灰度帧。

## 安装

使用 Python 3.10 或更高版本。进入本目录，在独立的 Python 环境中执行：

```sh
python -m pip install -r requirements.txt
```

若需要 GPU，请先通过 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)为当前平台选装 CUDA 版本。命令支持 `--device cpu`、`--device cuda` 或自动选择的 `--device auto`。OpenCV 可选，用于加速连通域计算；未安装时使用内置回退实现。

## 准备外部数据

数据集下载：[STSAU 数据集（百度网盘）](https://pan.baidu.com/s/5F_ItCOxW5A6PCq6Zx8QqhQ)。

`--data-root` 若直接指向含 `manifest.jsonl` 的数据根目录，`--dataset STSAU_wide9000` 只是逻辑名称，无需重命名磁盘上的数据目录。若传入的是父目录，`--dataset` 需要与实际子目录名一致。推荐使用以下预备数据格式：

```text
<dataset root>/
  manifest.jsonl
  splits/
    train.jsonl
    val.jsonl
    test.jsonl
  sources/
    <source_dataset>/<source_sequence>/<clip_id>/
      images/00001.png ... 00007.png
      masks/00001.png ... 00007.png
```

每个 JSONL 行对应一个片段，字段如下。列表中的路径相对数据根目录；每个划分中的记录须与总清单中对应的记录完全一致。

| 字段 | 含义 |
| --- | --- |
| `source_dataset`、`source_sequence`、`clip_id` | 来源、序列和片段 ID，与目录名一致 |
| `parent_sequence` | 母序列 ID；同一来源的同一母序列只能属于一个划分 |
| `split` | `train`、`val` 或 `test` |
| `source_frame_ids`、`center_frame` | 7 个递增且连续的原始帧号；`center_frame` 固定填 4（从 1 开始计） |
| `images`、`masks` | 按时间排序的 7 个相对 PNG 路径，每帧为 256×256；标签为 0 或 255 |
| `annotation_type` | 例如精确像素标注 `pixel_mask`、填充框弱标注 `bbox_weak` 或重渲染标注 `resized_synthetic_pixel_mask`；评价按该字段分别汇总 |
| `generated` | 布尔值；`real_synthetic` 按该值分为两组。未标记为生成的公开来源不等于原生实拍 |

预备数据只按明确的划分索引读取；每个片段取第 4 帧 mask 监督，整段连续帧作为输入。若使用 WideIRSTD 生成片段，放在训练划分。不使用清单的 `STSAU_dataset` 布局使用 `STSAU/` 存放图像、`STSAUmasks/` 存放掩码，可由外部数据目录中的 `stsau_dataset_pairs.csv` 提供配对，列为 `image_sequence,mask_sequence,frame_offset`；本包不包含实际配对表。

```sh
python verify_data.py --data-root /path/to/dataset --dataset STSAU_wide9000
```

## 训练与推理

将示例命令中的 `/path/to/dataset` 替换为你的数据目录（路径有空格时加引号）。50 轮示例配置位于 [configs/wide9000_50ep.args](configs/wide9000_50ep.args)，一行写一个参数名或值。它使用七帧、256×256、batch 2、Adam（学习率 1e-4）、seed 1，以及均衡有放回采样与困难背景 BCE + soft IoU。均衡采样每轮抽取训练集大小相同的次数，并不保证每个独立片段都出现。

```sh
python train.py "@configs/wide9000_50ep.args" --data-root /path/to/dataset --output-dir outputs/train --device auto
python infer.py --checkpoint outputs/train/best.pt --data-root /path/to/dataset --dataset STSAU_wide9000 --output-dir outputs/predictions --device auto
```

训练从头初始化，按验证 IoU 保存 `best.pt`，每轮更新 `last.pt`。也可将 `--checkpoint` 换成用户自有的 STSAUNet 检查点（含 `model_config`、`model_state`）。只加载可信来源的 PyTorch 检查点。推理保存便于查看的灰度概率图和二值 mask。

## 评估和复杂度

```sh
python evaluate.py --checkpoint outputs/train/best.pt --data-root /path/to/dataset --dataset STSAU_wide9000 --split test --output-json outputs/test_scores.json --device auto
python -m stsaunet.benchmark.evaluate --checkpoint outputs/train/best.pt --data-root /path/to/dataset --dataset STSAU_wide9000 --split test --output-dir outputs/test_curves --device auto
python -m stsaunet.benchmark.roc --inputs outputs/test_curves --output outputs/test_roc --low-fpr
python -m stsaunet.benchmark.pr --inputs outputs/test_curves --output outputs/test_pr
python -m stsaunet.benchmark.profile_model --checkpoint outputs/train/best.pt --shape 1 7 1 256 256 --device auto --output outputs/complexity.json
```

基础评估给出固定阈值 0.5 下的 IoU、Precision、Recall、F1、PD 和 FA。PD 按 8 连通目标的质心在 3 像素内做一对一匹配，FA 是误报背景像素数除以真值背景像素总数。曲线入口使用预备数据中心帧 logits 的原始 sigmoid 浮点概率，按全像素、来源和标注类型分别输出指标。ROC 以像素 FPR 为横轴、TPR 为纵轴；低 FPR 指标取在给定误报预算下的最高观测 TPR，不插值。AP 和梯形 PR-AUC 分开报告；`--mode exact` 使用原始分数，数据量大时可选 `--mode histogram` 近似。绘图入口读取已导出的曲线；叠加曲线时要求各组数据使用同一批有序样本。若需要选择阈值，应先在验证集确定，再固定到测试集。

复杂度统计包含一次完整片段推理和辅助重建头；卷积按 1 MAC = 2 FLOPs，采样及部分逐元素操作使用估算规则。输出 JSON 区分卷积计数、估算总量与未计数算子。

可以不使用数据或权重做 CPU 前向检查：

```sh
python -c "import torch; from stsaunet.model import STSAUNet; m=STSAUNet(window_size=7).eval(); print(torch.no_grad()(m)(torch.rand(1,7,1,16,16))['logits'].shape)"
```
