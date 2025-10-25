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
12. [训练配方参考](#11-训练配方参考)  
13. [常见数值问题 FAQ](#12-常见数值问题-faq)  
14. [标准库 Only 性能建议](#13-标准库-only-性能建议)  
15. [验证流程（从快到全）](#14-验证流程从快到全)  
16. [许可证](#15-许可证)

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
python -m snn_ocr.cli train --stage S1 --out runs/s1_baseline --epochs 5 --lr 0.003 --batch 8

# S2 分类 + 重放（输入 `--replay 0.1` 表示 10% 上一阶段样本）
python -m snn_ocr.cli train --stage S2 --out runs/s2_baseline --epochs 8 --batch 8 --replay 0.1

# S3 / S4 CTC
python -m snn_ocr.cli train --stage S3 --out runs/s3_baseline --epochs 10 --batch 6 --lr 0.002 --clip 0.8
python -m snn_ocr.cli train --stage S4 --out runs/s4_baseline --epochs 12 --batch 4 --lr 0.0018 --clip 0.7

# 推荐：S3 单词 CTC 训练（目标 CER≤0.18）
python -m snn_ocr.cli train   --stage S3   --epochs 12   --batch 6   --lr 0.002   --clip 0.7   --out runs/s3_recommended
```

### 评估与可视化
```bash
python -m snn_ocr.cli eval --stage S4 --data data/s4_sentences --ckpt runs/s4/ckpt.json
python -m snn_ocr.cli preview --text "HELLO\\nWORLD!" --out /tmp/hello.pgm --ascii


# 推荐：配套 S3 评估（输出 CER/WER + runs/vis 报告）
python -m snn_ocr.cli eval \
  --stage S3 \
  --data data/s3_words \
  --ckpt runs/s3/ckpt.json \
  --examples 6 \
  --vis runs/vis
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

## 11. 训练配方参考

以下配方与 CLI 约定保持一致（所有命令以 `python -m snn_ocr.cli ...` 开头）。

> `train` 子命令基于课程采样器，不需要 `--data`/`--beam`/`--T` 等数据路径参数；`synth` 生成的数据可用于评估、可视化或回归基准。

### 11.1 Smoke 快速验证（10–20 分钟）
目标：S1 Top‑1 ≥ 90%，S2 Top‑1 ≥ 85%，S3 CER ≤ 0.40，S4 CER ≤ 0.55。

```bash
# S1：生成 + 速训 + 验证
python -m snn_ocr.cli synth --stage S1 --n 1200 --out data/s1_smoke --width 28 --height 28 --seed 7
python -m snn_ocr.cli train --stage S1 --out runs/s1_smoke --epochs 3 --batch 8 --lr 0.003 \
  --warmup-steps 200 --cosine --clip 1.0
python -m snn_ocr.cli eval  --stage S1 --data data/s1_smoke --ckpt runs/s1_smoke/ckpt.json

# S2：加入 10% 重放
python -m snn_ocr.cli synth --stage S2 --n 2400 --out data/s2_smoke
python -m snn_ocr.cli train --stage S2 --out runs/s2_smoke --epochs 4 --batch 8 --replay 0.1 \
  --lr 0.003 --warmup-steps 200 --cosine --clip 1.0

# S3：CTC 小束宽
python -m snn_ocr.cli synth --stage S3 --n 3000 --out data/s3_smoke
python -m snn_ocr.cli train --stage S3 --out runs/s3_smoke --epochs 5 --batch 4 --lr 0.002 --clip 0.8
python -m snn_ocr.cli eval  --stage S3 --data data/s3_smoke --ckpt runs/s3_smoke/ckpt.json --examples 5

# S4：CTC 长序列
python -m snn_ocr.cli synth --stage S4 --n 4000 --out data/s4_smoke
python -m snn_ocr.cli train --stage S4 --out runs/s4_smoke --epochs 6 --batch 4 --lr 0.0018 --clip 0.7
python -m snn_ocr.cli eval  --stage S4 --data data/s4_smoke --ckpt runs/s4_smoke/ckpt.json --examples 6
```

若未达标，优先检查：`--clip` 与 `--warmup-steps` 是否生效、`eval` 输出的 `blank_ratio` 是否落在 0.4–0.8 区间。

### 11.2 Standard（默认推荐链路）
目标：S1 Top‑1 ≥ 96%，S2 Top‑1 ≥ 93%，S3 CER ≤ 0.25，S4 CER ≤ 0.35，同时保持 OTC `W→W' ≥ 3×`。

```bash
# S1：8 个 epoch，余弦退火
python -m snn_ocr.cli synth --stage S1 --n 6000 --out data/s1_digits --width 28 --height 28 --seed 7
python -m snn_ocr.cli train --stage S1 --out runs/s1 --epochs 8 --batch 8 --lr 0.003 \
  --warmup-steps 500 --cosine --clip 1.0

# S2：15% 重放 + 蒸馏
python -m snn_ocr.cli synth --stage S2 --n 12000 --out data/s2_letters
python -m snn_ocr.cli train --stage S2 --out runs/s2 --epochs 10 --batch 8 --lr 0.003 \
  --replay 0.15 --distill --distill-lambda 0.3 --distill-temp 2.0 \
  --warmup-steps 500 --cosine --clip 1.0

# S3：更长序列、定期 DEV
python -m snn_ocr.cli synth --stage S3 --n 20000 --out data/s3_words
python -m snn_ocr.cli train --stage S3 --out runs/s3 --epochs 12 --batch 6 --lr 0.002 \
  --clip 0.7 --dev-steps 10 --dev-every 2 --save-every 500
python -m snn_ocr.cli eval  --stage S3 --data data/s3_words --ckpt runs/s3/ckpt.json --examples 8

# S4：蒸馏 + 断点监控
python -m snn_ocr.cli synth --stage S4 --n 30000 --out data/s4_sentences
python -m snn_ocr.cli train --stage S4 --out runs/s4 --epochs 16 --batch 4 --lr 0.0015 \
  --clip 0.6 --cosine --distill --teacher runs/s3/ckpt.json --save-every 400
python -m snn_ocr.cli eval  --stage S4 --data data/s4_sentences --ckpt runs/s4/ckpt.json --examples 8
```

若 CER 上不去，可：调低 `--lr` 0.0003、增大 `--warmup-steps`、在 `spikes.encode_poisson` 中将 `rate_scale` 调至 0.9。

### 11.3 Repro（多 Seed 聚合）
目标：多次训练后的 CER/WER 方差 ≤ 0.01，并产出 `repro.json` 记录超参、commit 与环境。

```bash
for SEED in 3 7 13; do
  python -m snn_ocr.cli train --stage S4 --out runs/s4_seed${SEED} --epochs 10 --batch 4 \
    --lr 0.003 --warmup-steps 500 --cosine --clip 1.0 --seed $SEED
  python -m snn_ocr.cli repro --stage S4 --seed $SEED --ckpt runs/s4_seed${SEED}/ckpt.json \
    --command "python -m snn_ocr.cli train --stage S4 --seed $SEED ..." \
    --out runs/repro/s4_seed${SEED}.json
done
# 训练结束后，可写一个三十行脚本统计 CER/WER 的均值、方差并对 error bucket 做复盘。
```

建议在 `runs/repro/` 下保留 `vis_dir` / `report.json` 路径，方便日后回溯 Top‑K 错误样例。

## 12. 常见数值问题 FAQ

| 症状 | 可能原因 | 快速排查 | 解决方案 |
| --- | --- | --- | --- |
| CTC 解码全 `<blank>`（空白塌缩） | OTT 压得过窄 / Beam 太小 / Blank logits 偏高 | `python -m snn_ocr.cli eval --stage S4 ...` 查看 `blank_ratio` 是否 >0.7 | 1) 调高 OTC `max_merge`; 2) 训练期拉大 `--beam`; 3) 在 `train.py` 的 logits 上加 `-0.1` 的 blank 偏置。 |
| 发放过多（脉冲爆，能耗飙升） | `rate_scale` 偏大或输入对比度过高 | 查看 `compute_fire_rate`（训练日志）是否 >0.35 | 1) 在 `spikes.encode_poisson` 调低 `rate_scale`; 2) 合成数据降低 `--contrast`; 3) OTC 前加入列归一。 |
| 发放过少（序列稀疏，梯度停滞） | TTFS 阈值过高 / 输入过暗 | `python -m snn_ocr.cli profile --stage S3 ...` 检查 `spike_total` | 1) 增大合成噪声、加随机亮度；2) 将 LIF `v_th` 从 1.0 调至 0.85；3) 使用 Poisson 编码替换 TTFS。 |
| 梯度爆炸 | 学习率过大 / clip 失效 | 日志出现 `nan loss`、`grad_inf` | 1) 减小 `--lr` 和 `--clip`; 2) 检查是否对空 batch 回传；3) 在 `train.py` 中启用 `clip_mode=value`。 |
| 梯度消失 | TET 权重全部落在前时步 / 输入序列过长 | Loss 始终 >1 且无下降 | 1) 切换 `tet_mode` 为 `tail`; 2) 缩短 OTC 输出宽度；3) 添加知识蒸馏（`--distill`）。 |

更多诊断：
1. `python -m snn_ocr.cli doctor` → 自检 Import 与 smoke 测试。
2. `python -m snn_ocr.cli profile --stage S4 --sample 3` → 单样本推理耗时、blank 时间线。
3. `python -m snn_ocr.cli repro --stage S3 --ckpt runs/s3/ckpt.json --out runs/repro/s3.json` → 固化调参状态。

## 13. 标准库 Only 性能建议

1. **预分配列表**：在卷积、OTC 等核心循环中使用 `[0.0 for _ in range(n)]` 一次性创建，再用索引覆写，避免频繁 `append`。
2. **memoryview/bytearray/array**：PGM/PPM IO、spike tensor 和中间特征使用 `memoryview(bytearray(...))` 或 `array('h')` 原位操作，减少复制与 GC 压力。
3. **行缓冲 + 滑窗**：在 `lif._linear_conv2d` / `conv1d` 中缓存上一行/列的 padded 结果，同时预计算滑窗索引，避免重复切片和 `range` 重新构造。
4. **减少对象创建**：训练循环里复用 `dict` / `list` 模板 (`template = [0.0]*C`)；跨步的 `grad`、`log_probs` 可以从 `model.grad_template()` 拷贝而不是新建。
5. **批量随机 & 统一种子**：所有模块经由 `random.Random(seed)` 创建，并提前生成列表再循环消费；`synth/train/eval` 共用同一个 seed 以便 repro。
6. **稳定的 log_sum_exp**：提取公共实现、统一 `epsilon` 与裁剪，避免在循环内创建临时列表或重复计算 `math.exp`。
7. **OTC 信息量**：方差/熵都要做极小值钳制；合并时保持列序不变；DP 方案建议使用前缀和 + 一维 `best[k]`，复杂度 O(W·K)。
8. **日志与聚合**：训练日志写入 JSONL（每行一个 dict），CLI 输出一次只打印一行；配合一个 30 行的脚本即可聚合 loss/CER。

## 14. 验证流程（从快到全）

### 14.1 自检（必须先过）
- 命令：`python scripts/self_check.py` 或 `python -m snn_ocr.cli doctor`。
- 通过标准：终端打印 `非标准库依赖 0 条`；关键 API 均为 `OK`；`missing.json` 被生成/更新且仅记录缺失项。

### 14.2 Smoke 验证（功能完整性）
- 数据：使用 [11.1 Smoke](#111-smoke-快速验证10–20-分钟) 命令生成。
- 通过标准：各阶段命令可独立跑通，S3/S4 的 `blank_ratio` 介于 0.4–0.8，能耗代理（脉冲总数 / 像素 / duty）随 `T` 增大而单调上升。

### 14.3 Standard 验证（数值与稳定性）
- 数据：使用 [11.2 Standard](#112-standard默认推荐链路) 命令生成。
- 通过标准：指标达到目标值；打开/关闭 OTC 时 W→W′ 曲线合理且 CER 无显著回退；断点恢复（`--resume`）后首个 epoch 的 loss 差 < 1e‑6。

### 14.4 Repro（多 seed 聚合）
- 数据：使用 [11.3 Repro](#113-repro多-seed-聚合) 工程流程。
- 通过标准：3 个 seed 的 CER/WER 方差 ≤ 0.01；`runs/repro/*.json` 记录 seed、命令行、git commit 与系统信息；`runs/vis/` 的 Top‑K error 可回溯到图片/路径。

## 15. 许可证

MIT License，详见仓库根目录的 `LICENSE` 文件。
- 代码与 5×7 / 7×9 位图字体均遵循 **MIT License**。
- 合成数据仅供研究、测试与教学使用。
