# SNN-OCR Minimal (Standard Library Only)

**标准库 Only · MIT License · Python 3.10+**

纯视觉 SNN OCR 原型：不依赖 numpy / Pillow / torch / opencv 等第三方包，仅凭 Python 标准库跑通 “数据合成 → 脉冲编码 → LIF Backbone → OTC → 序列头 → CTC → 评测” 的教学级闭环。课程式训练模仿儿童识字：S1 数字 → S2 大/小写字母 → S3 单词 → S4 简单句。

## 目录
1. [环境与声明](#0-环境与声明)  
2. [项目目标](#1-项目目标)  
3. [架构概览](#2-架构概览)  
4. [数据合成](#3-数据合成无第三方)  
5. [脉冲编码](#4-脉冲编码静态图--时域)  
6. [SNN 基元与训练](#5-snn-基元与训练)  
7. [训练日程](#6-训练日程curriculum)  
8. [目录结构](#7-目录结构生成物约定)  
9. [CLI 使用方法](#8-使用方法cli)  
10. [指标与能耗](#9-指标与能耗估算)  
11. [限制与扩展](#10-已知限制与扩展)  
12. [许可证](#11-许可证)

---

## 0. 环境与声明
- **Python 3.10+**；仅使用 `typing / math / random / itertools / dataclasses / argparse / json / time / os / pathlib / sys` 等标准库。
- **禁止** 引入 numpy、Pillow、torch、opencv 及其他第三方依赖。
- **格式**：自实现 PGM/PPM（Netpbm）写入，可被常见查看器打开。
- **字体**：内置 5×7 / 7×9 位图字体（0–9、A–Z、a–z、标点），无需额外授权。
- **性能**：全部 Python 循环实现，适合小批量实验、教材演示或算法验证。

## 1. 项目目标
- 纯视觉输入完成 OCR，不引入语言模型、词典或外部语料。
- 使用 LIF 神经元 + 替代梯度 / TET，低时步实现时域编码与特征抽取。
- 通过 OTC（Optical Token Compressor）沿高度压缩至 H′≈1，保留列序列。
- 依靠 Python 版 CTC（前向-后向 + 贪心/Beam 解码）完成无对齐转写。
- 课程式训练（S1→S4）确保模型逐步习得数字、字母、单词与句子。

## 2. 架构概览
```
PGM/PPM Image
└─ Spike Encoders (TTFS / Poisson，ON/OFF 双通道)
   └─ Early Visual Blocks（位图卷积替代 Gabor/DoG）
      └─ SNN Backbone（多层 LIF 1D/2D 卷积 + 残差）
         └─ OTC（高度压缩，控制 token 列数 W′）
            └─ Sequence Head（1D DWConv ×2 + 轻量注意力）
               └─ CTC Logits（含 `<blank>`）
                  └─ 解码（贪心 or 小束宽 Beam）
```

仅标准库下的折衷：
- 卷积/注意力均用 Python 循环模拟（小通道、小核），强调可读性。
- 重点放在跑通训练与推理闭环：S1/S2 使用分类损失，S3/S4 进入 CTC。

## 3. 数据合成（无第三方）

### 3.1 位图字体渲染
- `bitfont.py` 提供 5×7 / 7×9 字形，覆盖 0–9、A–Z、a–z、标点及空格。
- 文本拼接支持：
  - 整数倍缩放（最近邻）、水平错切（伪斜体）、3×3 膨胀变粗。
  - 抖动 / 噪声 / 对比度扰动、±1 像素扫视、程序化/纯色背景。
- 输出格式：
  ```text
  P5\n{W} {H}\n255\n<二进制灰度...>
  ```
  或
  ```text
  P6\n{W} {H}\n255\n<RGB 字节...>
  ```

### 3.2 课程式样本
| 阶段 | 文本类型 | 长度 & 特征 |
| --- | --- | --- |
| S1 | 单个数字 | 0–9，统一尺寸，轻噪声 |
| S2 | 单个字母 | A–Z / a–z，扰动更强 |
| S3 | 单词 | 长度 3–12，仅 `<space>` 分隔 |
| S4 | 句子 | 15–60 字符，含 `<space>` + 标点 (.,!? 及换行) |

所有样本由 `synth.py` 生成至 `data/` 目录，并写入 `labels.jsonl`。

## 4. 脉冲编码（静态图 → 时域）
- **ON/OFF 双流**：正负亮度变化分别编码；对静态图注入微扰获得事件。
- **TTFS**：像素亮度越高越早发放，S1/S2 默认 `T=4`。
- **Poisson 率编码**：S3/S4 使用 `T=6~8`，亮度转发放率，`random.random()` 抽样。
- **多 τ**：层内采用不同衰减系数 `alpha=exp(-Δt/τ)`，平衡快慢特征。

## 5. SNN 基元与训练

### 5.1 LIF（软复位）
前向更新：
```text
v[t] = alpha * v[t-1] + i[t] - r[t-1] * v_th
r[t] = H(v[t] - v_th)
```
使用分段线性替代梯度近似 `H'(x)`，阈值附近保留梯度，远离阈值置零。

### 5.2 TET（Temporal Efficient Training）
跨时步加权（均匀 / 尾部偏置）抑制首时步主导；配合平滑项提升泛化。

### 5.3 OTC（Optical Token Compressor）
- 仅压缩高度维，使 H′≈1，保持宽度顺序。
- 依据列方差/熵估算信息量，低信息列在窗口内合并，生成更短序列。

### 5.4 CTC（纯 Python）
- 在对数域完成前向-后向，计算 `-log P(target | inputs)`。
- 推断时提供贪心与小束宽 Beam（默认 5–10），无语言模型依赖。

## 6. 训练日程（Curriculum）

| 阶段 | 任务 | 编码/时步 | 序列头 | 损失 | 备注 |
| --- | --- | --- | --- | --- | --- |
| S1 | 数字分类 | TTFS · T=4 | GAP + 线性 | CE + TET | 打稳底座 |
| S2 | 字母分类 | TTFS · T=4 | 同上 | CE + TET | 混入 5–10% S1 重放 |
| S3 | 单词 CTC | Poisson · T=6~8 | 1D DWConv×2 | CTC + TET | 冻前层，训序列头 |
| S4 | 句子 CTC | Poisson · T=6~8 | 同上 | CTC + TET + 平滑 | 含 `<space>`、标点、换行 |

- 优化器：纯 Python SGD / Adam，支持余弦退火。
- 小批量（1–8）+ Grad Clip；每阶段定期评估 CER/WER。

## 7. 目录结构（生成物约定）
```
snn_ocr_minimal/
├── README.md
├── LICENSE
├── snn_ocr/
│   ├── bitfont.py      # 位图字体与绘制
│   ├── pgm.py          # PGM/PPM IO
│   ├── render.py       # 文本→灰度图，多行/扰动
│   ├── synth.py        # 数据合成 + JSONL 标签
│   ├── spikes.py       # TTFS / Poisson 编码
│   ├── lif.py          # LIF 模块、替代梯度、TET
│   ├── otc.py          # Optical Token Compressor
│   ├── seq.py          # 序列头 (DWConv + 注意力)
│   ├── ctc.py          # CTC Loss、贪心/Beam
│   ├── train.py        # 课程化训练脚本
│   ├── eval.py         # 指标、能耗、可视化
│   ├── cli.py          # 单入口 CLI (synth/train/eval/preview)
│   └── utils.py        # 计时、日志、纯标准库工具
├── data/
│   ├── s1_digits/
│   ├── s2_letters/
│   ├── s3_words/
│   └── s4_sentences/
├── runs/
│   ├── s1/ckpt.json
│   ├── s2/ckpt.json
│   └── ...
└── tests/
    ├── test_bitfont.py
    ├── test_pgm.py
    ├── test_lif.py
    ├── test_seq.py
    └── test_ctc.py
```

## 8. 使用方法（CLI）
所有命令都走 `python -m snn_ocr.cli ...`，无需任何第三方依赖。

### 数据生成
```bash
# S1
python -m snn_ocr.cli synth --stage S1 --n 2000 --out data/s1_digits --width 28 --height 28 --seed 7

# S2
python -m snn_ocr.cli synth --stage S2 --n 5000 --out data/s2_letters

# S3 / S4
python -m snn_ocr.cli synth --stage S3 --n 8000 --out data/s3_words
python -m snn_ocr.cli synth --stage S4 --n 12000 --out data/s4_sentences
```

### 训练
```bash
# S1 分类
python -m snn_ocr.cli train --stage S1 --data data/s1_digits --epochs 5 --lr 0.003 --batch 8

# S2 分类 + 重放
python -m snn_ocr.cli train --stage S2 --data data/s2_letters --replay data/s1_digits --epochs 8

# S3 / S4 CTC
python -m snn_ocr.cli train --stage S3 --data data/s3_words --epochs 10 --beam 5
python -m snn_ocr.cli train --stage S4 --data data/s4_sentences --epochs 12 --beam 8
```

### 评估与可视化
```bash
python -m snn_ocr.cli eval --stage S4 --data data/s4_sentences --ckpt runs/s4/ckpt.json
python -m snn_ocr.cli preview --text "HELLO\\nWORLD!" --out /tmp/hello.pgm --ascii
```

## 9. 指标与能耗（估算）
- S1/S2：Top-1 准确率。
- S3/S4：CER / WER。
- 能耗代理：记录总脉冲数、平均每像素/列脉冲、有效时步占比。
- OTC 统计：输出 W → W′ 压缩比与对应 CER 变化。

## 10. 已知限制与扩展
- 纯 Python 版本运行较慢，适合教学与概念验证；如需性能可在不改接口的前提下另建分支接入 numpy/numba/torch。
- 位图字体较简单，复杂连笔/花体需自行扩展 `bitfont.py`。
- 训练默认在小数据集上演示；大规模实验需自行调参并谨慎评估。

## 11. 许可证
- 代码与 5×7 / 7×9 位图字体均遵循 **MIT License**。
- 合成数据仅供研究、测试与教学使用。
