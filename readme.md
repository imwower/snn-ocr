SNN‑OCR‑Minimal (Standard Library Only)

一个纯视觉脉冲神经网络（SNN）微型 OCR 原型：不依赖第三方包（如 numpy、Pillow、PyTorch 等），仅使用 Python 标准库 完成数据合成→脉冲编码→SNN 前端→自适应视觉 Token 压缩（OTC）→序列头→CTC 转写→评测的闭环。
训练方式模仿儿童识字：数字 → 大/小写字母 → 简单词 → 简单句。

0. 环境与声明

Python：3.10+（仅用标准库：typing、math、random、itertools、dataclasses、argparse、json、time、os、pathlib、sys 等）。

不使用：numpy / Pillow / torch / opencv / 任何第三方。

图像格式：自实现 PGM/PPM 写入（Netpbm，最简单的便携位图格式），可被多数图像查看器打开。

字体：内置一套 5×7/7×9 位图字体（ASCII 子集：0–9、A–Z、a–z、标点），版权自持。

性能说明：全部以 Python 循环实现，能跑通但速度有限，适合小规模训练/验证与教学演示。

1. 项目目标

只靠视觉信号完成 OCR，不接语言模型/词典/外部语料。

用 SNN（LIF 神经元、替代梯度/TET）实现低时步的时域编码与特征抽取。

用 OTC（Optical Token Compressor） 沿高度压缩到 H′≈1，保留宽度序列以适应行文读取。

用 CTC 实现无对齐监督的序列转写，输出包含 `<blank>` 与 `<space>`。

用课程式训练从易到难：S1 数字 → S2 字母 → S3 词 → S4 句。

2. 架构概览
PGM/PPM 图片
   └─ Spike Encoders（TTFS/Poisson, ON/OFF 双通道）
      └─ Early Visual (简化 Gabor/DoG → 本项目用位图/卷积替代)
         └─ SNN Backbone（若干 LIF 1D/2D 卷积块 + 残差）
            └─ OTC（只在高度维下采样至 H′≈1，控制列 token 数 W′）
               └─ Sequence Head（1D 深度可分卷积 + 轻量注意力 可选）
                  └─ CTC Logits（按列输出字符分布 + `<blank>`）
                     └─ 解码（贪心 / 小束宽 Beam）


仅标准库约束下的取舍：

卷积/注意力全部用 Python 循环实现（小通道、小核、小层数）。

重点在可跑通训练闭环：S1/S2 支持分类损失，S3/S4 支持 CTC（含前向-后向、束搜索解码）。

3. 数据合成（无第三方）
3.1 位图字体渲染

内置 bitfont.py：为 0–9、A–Z、a–z、标点（.,!? 和空格）提供 5×7 或 7×9 二值位图。

文本 → 画布：将字符位图按内/外间距拼接为一行，支持：

缩放（整数倍、就地最近邻），

斜体/透视近似（整数像素级水平错切），

粗细（3×3 最大值膨胀若干次），

噪声（椒盐/随机点、对比度扰动），

微扫视（全图±1像素平移），

背景（程序化噪声/条纹；或纯色）。

输出 PGM（灰度 0–255）或 PPM（彩色）：

```
P5\n{W} {H}\n255\n<二进制像素...>
```
或
```
P6\n{W} {H}\n255\n<RGB字节...>
```

3.2 课程式样本

S1：单字符（0–9），统一尺寸（如 28×28），高对比，轻噪声。

S2：单字符（A–Z/a–z），字体/扰动↑。

S3：词（长度 3–12），仅 `<space>` 作为词间空白，不含标点。

S4：句（长度 15–60），含 `<space>` 与少量标点（.,!?）。

所有阶段样本通过 synth.py 一键生成到 data/ 目录（含标签 JSONL）。

4. 脉冲编码（静态图 → 时域）

ON/OFF 双流：对帧的亮度变化进行正/负通道编码；静态图通过合成微扰获得时域事件。

TTFS（首次放电时间）：像素强度越大越早放电，S1/S2 默认 T=4。

Poisson 率编码：S3/S4 用 T=6~8；像素强度→发放率，用 random.random() 采样。

多 τ 混合：不同层使用不同膜时间常数（通过衰减系数 alpha 实现）。

5. SNN 基元与训练
5.1 LIF（软复位）更新

记 v[t] 为膜电位、i[t] 为输入电流、阈值 v_th、衰减 alpha=exp(-Δt/τ)（此处直接设定常数）。

前向：

v[t] = alpha * v[t-1] + i[t] - r[t-1] * v_th      # 软复位
r[t] = H(v[t] - v_th)                              # 硬阈发放 (0/1)


替代梯度（训练用）：将 H'(x) 用三角/分段线性近似（阈值附近给定斜率，远离阈值为 0）。

5.2 TET（Temporal Efficient Training）

将跨时步的损失加权平均或后期加权，抑制首时步偏置、提高泛化。

5.3 OTC（视觉 Token 压缩）

仅在高度维做池化/合并，直到 H′≈1；宽度 W 不池化，保序列性。

自适应门控：以每列像素/特征的方差或熵近似为信息量指标，低信息列可合并，得到 W′≪W。

5.4 CTC 前向-后向（纯 Python）

给定每列的字符分布（含 `<blank>`），通过动态规划在对数域计算 -log P(target|inputs)。

贪心/束搜索（Beam Width 5–10）做推断；不使用语言模型。

6. 训练日程（Curriculum）
阶段	任务	编码/时步	头	损失	备注
S1	数字 0–9 分类	TTFS, T=4	GAP + 线性	CE + TET	先把底座训稳
S2	字母 52 类	TTFS, T=4	同上	CE + TET	混入 S1 重放 5–10% 防遗忘
S3	词级 CTC	Poisson, T=6–8	序列头（1D DWConv ×2）	CTC + TET	冻前层、训后层
S4	句级 CTC	Poisson, T=6–8	同上	CTC + TET + 时域平滑	含 `<space>` 与少量标点

优化器：手写 SGD/Adam（标准库实现），lr 余弦退火。
批处理：小 batch（1–8），Grad Clip，周期性评估 CER/WER。

7. 目录结构（生成物约定）
snn_ocr_minimal/
  README.md
  LICENSE
  snn_ocr/
    __init__.py
    bitfont.py        # 内置位图字体与绘制原语
    pgm.py            # PGM/PPM 读写
    render.py         # 文本→位图→PGM/PPM，含增强/抖动/噪声
    synth.py          # 按阶段合成数据集，生成 JSONL 标签
    spikes.py         # TTFS/Poisson 编码，ON/OFF 双流
    lif.py            # LIF 神经元与层、替代梯度、TET
    otc.py            # 光学 Token 压缩（高度合并→H′≈1）
    seq.py            # 序列头（1D DWConv/简注意力 可选）
    ctc.py            # CTC 前向-后向、贪心/Beam 解码
    train.py          # 四阶段训练脚本（可断点续训）
    eval.py           # 指标计算：Top1/CER/WER、能耗估计（脉冲计数）
    cli.py            # 命令行封装（synth/train/eval/preview）
    utils.py          # 随机数、调参、日志、进度条（纯标准库）
  data/
    s1_digits/...
    s2_letters/...
    s3_words/...
    s4_sentences/...
  runs/
    s1/ckpt.json
    s2/ckpt.json
    ...
  tests/
    test_bitfont.py
    test_pgm.py
    test_lif.py
    test_ctc.py

8. 使用方法（CLI）

所有命令均仅依赖标准库，可直接 python -m snn_ocr.cli ...

生成 S1 数据：

python -m snn_ocr.cli synth --stage S1 --n 2000 --out data/s1_digits --width 28 --height 28 --seed 7


训练 S1（数字分类）：

python -m snn_ocr.cli train --stage S1 --data data/s1_digits --epochs 5 --lr 0.003 --batch 8


生成并训练 S2（字母分类，带重放防遗忘）：

python -m snn_ocr.cli synth --stage S2 --n 5000 --out data/s2_letters
python -m snn_ocr.cli train --stage S2 --data data/s2_letters --replay data/s1_digits --epochs 8


S3/S4（CTC 转写）：

python -m snn_ocr.cli synth --stage S3 --n 8000 --out data/s3_words
python -m snn_ocr.cli train --stage S3 --data data/s3_words --epochs 10 --beam 5

python -m snn_ocr.cli synth --stage S4 --n 12000 --out data/s4_sentences
python -m snn_ocr.cli train --stage S4 --data data/s4_sentences --epochs 12 --beam 8


评估/可视化：

python -m snn_ocr.cli eval --stage S4 --data data/s4_sentences --ckpt runs/s4/ckpt.json
python -m snn_ocr.cli preview --text "HELLO WORLD!" --out /tmp/hello.pgm --ascii

9. 指标与能耗（估算）

S1/S2：Top‑1 准确率。

S3/S4：CER/WER；

能耗代理：统计总脉冲数、平均每像素/每列脉冲数、有效时步占比。

OTC 效果：记录 W → W′ 的压缩率与 CER 变化。

10. 已知限制与扩展

纯 Python 实现较慢；建议用于小规模演示与算法理解。

位图字体简单，复杂连笔/花体会更难；可在 bitfont.py 扩充更多字形。

如需性能，可在不改接口的前提下，将内部运算替换为 numpy/numba/torch（另分支）。

11. 许可证

代码与 5×7/7×9 位图字体：MIT。

生成数据仅用于研究与测试。
