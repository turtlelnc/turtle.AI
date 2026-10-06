# 维基百科 CPU 实训记录

2026-10-06 在 Linux 本地环境实际运行 MLX CPU 后端，完成训练、检查点恢复、
留出集验证和生成。详细配置、命令和结果见
[wikipedia_cpu_2026-10-06.json](wikipedia_cpu_2026-10-06.json)。

## 数据与模型

- 数据：`wikimedia/wikipedia`，`20231101.en`，Dataset Viewer 第 1000–1999 行。
  连续选取 1,000 篇文章，每第 10 篇留作验证；训练和验证文章 ID 没有交集。
- 训练：900 篇，18,849,502 字节；留出集：100 篇，2,113,671 字节。
  来源和许可证为 [Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia)，
  CC BY-SA 3.0 / GFDL。下载清单保存每篇文章的 ID、URL 和分组。
- 模型：AR，维度 64、上下文 128、循环上限 2、1 个 wide block、4 个专家、
  top-2 路由，实际词表 1,039，FP16 参数 511,704 个。
- 每步 4 个窗口，峰值学习率 0.001，种子 7，图编译和 packed batch 开启，
  GaLore 关闭，优化器状态常驻内存。
- 环境：MLX 0.31.2、GCC 14.2、Linux x86_64；容器限额 2 个 CPU、8 GiB 内存。

## 实测结果

| 检查 | 结果 |
|---|---|
| 初次训练 | 完成 600 步，含分词准备耗时 529.18 秒 |
| 恢复训练 | 从 600 恢复到 620 步，耗时 19.54 秒 |
| 权重更新 | 恢复前后模型权重 SHA-256 不同，完成步数正确 |
| 训练任务 loss | 最初 5 个日志采样点均值 6.6841，最后 5 个均值 4.0314 |
| 生成 | 3 个提示词各生成 80 Token，均正常返回 END |
| 无 KV cache 生成 | 同样正常完成；未验证与 KV cache 数值等价 |
| 非有限值 | 训练、恢复、生成和验证日志均未出现 NaN/Inf |
| 检查点 | 7,164,718 字节，包含参数、优化器状态和架构元数据 |

留出集固定选择 64 个文本窗口，共 8,192 个有效 Token。两次验证使用同一词表、
种子、数据和上下文，关闭路由噪声，报告纯交叉熵，不包括训练时的 confidence
penalty 或 MoE 辅助损失：

| 模型 | 留出集交叉熵 | 困惑度 |
|---|---:|---:|
| 随机初始化，seed=7 | 6.9413 | 1034.08 |
| 训练后，第 620 步 | 4.0320 | 56.37 |

这些结果支持训练与恢复链路正常，并且模型学到了这个子集中的统计规律。
生成仍不能形成连贯回答，例如提示 `The history of mathematics` 的输出开头是：

```text
 mport the to Prlew the Hada the and was the Gigjolet, in who s to rically lpublic
```

本次只处理约 31.7 万个训练目标位置，没有遍历全部语料；也没有评估完整留出集。
数据是连续子集，包含相近主题，不能将这些指标外推为通用语言或代码能力。
小模型按字节组合的生成还可能产生无效 UTF-8 序列。

## 复现与产物

从仓库根目录执行，先按照 [MLX 构建说明](../README.md) 构建程序：

```sh
python3 mlx/tools/download_wikipedia.py
python3 mlx/tools/train_wikipedia.py
```

下载和训练工具拒绝复用已有输出，以避免误恢复旧实验。重复运行时分别指定新的
`--output`，训练工具的 `--data` 要指向新的数据目录。模型和数据没有提交到 Git：

- `build-mlx/wikipedia/manifest.json`：文章来源、分组、数据文件校验值。
- `build-mlx/wiki-small/model.ckpt`：第 620 步模型。
- `build-mlx/wiki-small/before-resume.ckpt`：第 600 步模型。
- `build-mlx/wiki-small/tokenizer.bpe`：对应词表。
- `build-mlx/wiki-small/summary.json` 和 `*.log`：完整本地实验记录。

训练数据 SHA-256：
`37f6e7043b19fce3e1a38882a8fbf5c9baba50ec10b56933f6a614a9f29144db`。
留出数据 SHA-256：
`82a7219059d3fc30ed4d018026c12084defa2566dac29acc50aaa8647a18b21a`。

训练主体使用提交 `aa5dbad`；验证入口、AR 提示词边界修复和本文记录在后续提交。
训练面板原将含置信度惩罚的任务 loss 标成纯交叉熵，本次修正了名称。另一个
监控问题是路由探针读取固定零 Token，而不是训练批次：现已改为实际采样输入，
并明确标为嵌入输入探针；它仍不等于各循环层实际隐层上的专家分派统计。

## 下一步

优先延长训练并扩大、随机化验证样本，增加连贯性与任务评估；补齐实际隐层的
专家分派和容量丢弃指标。随后保存完整 RNG/GaLore 刷新状态，验证恢复轨迹一致性，
再到 Apple Silicon 上完成 Metal 与 CPU 的数值对照。

本次恢复改变了目标总步数，因此学习率计划也改变；RNG 状态尚未恢复，不能声称
它等价于连续运行 620 步。ELF/T5 有独立合成数据烟雾测试，此次真实语料实验
只验证 AR 流程。原 CPU 回归、ASan/UBSan 和 MLX 运行时回归均通过。
