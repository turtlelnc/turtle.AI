# MLX / OpenMythos 后端

这是上传的本地 `main.cpp`、`BPE.h`、`BPE.cpp` 接入后的独立后端。根目录
`train.cpp` 继续使用原来的 CPU 模型。两个程序的架构、词表和检查点不兼容。

保留的本地功能包括：MoE 与负载均衡损失、循环层/ACT、RMSNorm、RoPE、滑动窗口
和稀疏全局注意力、KV cache、packed batch、编译训练图、SNR 优化器、动态 GaLore、
内存/SSD 优化器状态、ELF 条件生成、T5 潜变量读取和 EMA。

## 构建

需要 C++20、CMake 3.25+ 和 MLX。验证依赖固定为 MLX 0.31.2，commit
`68cf2fddd8de5edd8ab3d926391772b2e2cedad8`。macOS 的 GPU 路径要求 Apple
Silicon 和相应的 macOS/Xcode SDK；Linux 可使用 MLX CPU 后端。该应用暂不支持
Windows，Windows 用户仍可构建根目录 CPU 程序。

从源码获取固定 MLX 版本：

```sh
cmake -S mlx -B build-mlx -DCMAKE_BUILD_TYPE=Release -DTURTLE_FETCH_MLX=ON
cmake --build build-mlx --parallel 2
ctest --test-dir build-mlx --output-on-failure
```

Linux 还需要 BLAS/LAPACK 开发文件，例如 Debian/Ubuntu 的 `libblas-dev`、
`liblapack-dev`、`liblapacke-dev`。macOS 使用 Accelerate。CMake 首次配置会下载
固定版本的 MLX 及其上游依赖。如果已有 MLX 的 CMake 安装包，可省略
`TURTLE_FETCH_MLX`，通过 `CMAKE_PREFIX_PATH` 指定安装目录。

只检查分词器和数据读取，无需安装 MLX：

```sh
make test-mlx-bpe
```

完整 `make test` / `make sanitize` 也包含这些独立检查。CMake 的
`TURTLE_BUILD_MLX=OFF` 可以构建同样的分词器测试。运行时烟雾测试使用 Python 3。

## 小规模训练与生成

以下命令从仓库根目录运行，输出与根目录 CPU 模型分开存放：

```sh
mkdir -p build-mlx/run
./build-mlx/turtle-mlx --device auto --data-dir examples \
  --dim 16 --seq-len 16 --max-loop 2 --wide-blocks 1 \
  --experts 2 --moe-topk 1 --window-size 8 --global-topk 4 \
  --batch-size 2 --steps 20 --vocab-size 320 --seed 7 \
  --galore-off 1 --model-out build-mlx/run/ar.ckpt \
  --tokenizer-out build-mlx/run/tokenizer.bpe

./build-mlx/turtle-mlx gen 'Hello turtle' --device auto \
  --dim 16 --seq-len 16 --max-loop 2 --wide-blocks 1 \
  --experts 2 --moe-topk 1 --window-size 8 --global-topk 4 \
  --model build-mlx/run/ar.ckpt --tokenizer build-mlx/run/tokenizer.bpe \
  --max-tokens 20
```

这是验证程序可运行的小模型示例，尚不能生成有用代码。`--device auto` 优先使用
可用 GPU，否则使用 CPU；`--device gpu` 在没有 GPU 后端时明确报错。
`--dim` 必须是 16 的倍数。恢复/生成时模型尺寸和模式必须与检查点一致。
输出目录需事先创建。`--steps` 是目标总步数，恢复到第 20 步后传入 30 会继续
训练 10 步。默认保存最终模型，`--save-final 0` 可关闭。

`--data-dir` 接受目录或单文件。单文件按 2000 字节分块、保留 400 字节重叠；
短文件和尾部保留原始换行。分词器保存数据集标识；更换数据后，使用新的分词器
路径，或用 `--allow-dataset-change 1` 明确复用既有词表进行后续训练。

`--optimizer-memory auto|resident|swap` 控制状态存储。
`--swap-dir` 指定 SSD 状态目录；并行运行多个训练进程时应使用不同目录。
`--resident-optimizer` 保留本地版的常驻优化器数组开关。
GaLore 参数保留 `--galore-rank`、`--galore-min-size`、`--galore-refresh`、
`--galore-min-interval`、`--galore-cos-thresh`、`--galore-fixed-refresh`。
`--compile-train-graph 0` 可关闭图编译，`--packed-batch 0` 使用梯度累积。
容量 MoE 默认启用，可在 CMake 配置时用 `TURTLE_CAPACITY_MOE=OFF` 构建旧的
BatchedMoE 实现；该替代路由仍需独立质量评估。

## 文件兼容性

MLX 分词器保留六个控制 Token：UNK/BOS/EOS/PAD 为 0–3，文件边界为 4–5，
基础字节从 6 开始。代码关键字词表保持本地版的初始 ID。新增保存格式
`OM_BPE_V2` 使用固定长度整数，保存完整词表、额外特殊 Token、合并规则、
lexer 版本和数据集标识。有效的 `OM_BPE_V1` 文件可加载，并保留旧 lexer 的
行为；损坏文件或重复/冲突 Token ID 会被拒绝。旧版训练器曾可能覆盖已有关键字
的 ID，这类文件不能安全自动迁移，需要重新训练词表并使用新的模型输出路径。

MLX 模型新增 `TRTLMX01` 元数据头，后面的参数和优化器布局保留本地 `GLR2` /
`ELF7` 格式。元数据检查模型尺寸、模式、容量 MoE 开关、字节序和分词器标识。
保存采用临时文件替换；读取先验证所有张量，再提交参数。
无元数据的旧 MLX 检查点可以加载，但只能检查参数布局，无法验证旧词表语义。
新文件不能直接被原本的本地程序读取。CPU 的 `TRTLMODL` 文件不适用于这里。

检查点步数表示已完成更新数。优化器状态会保存；GaLore 子空间仍按原设计在恢复
后重建，随机数生成器状态尚未保存，因此恢复训练的轨迹不保证与连续运行一致。

## ELF / T5

使用 `--model-mode elf` 和独立 `--model-out` 路径启用 ELF。最终模型同时保存
`.ema`，EMA 用于推理。已有的 `--elf-*` 选项保留，包括时间分布、prefix、
self-conditioning、CFG、decoder 分支、EMA 和采样设置。

T5 需要外部准备的 `--elf-t5-latents` 与 `--elf-t5-unembedding` 二进制文件。
潜变量文件包含 7 个原生 uint32：magic=`0x5435454C`、version=1、记录数、
序列长度、潜变量维度、词表大小、PAD ID；每条记录随后存储 int32 token IDs 和
FP16 潜变量。Unembedding 文件包含 4 个 uint32：magic=`0x45553554`、
version=1、词表大小、潜变量维度，随后是 FP16 矩阵。必须使用同一编码器/词表
生成这些文件。程序检查长度、维度、Token ID 与有限数值。

当前 T5 生成模式使用 `--elf-eval-record` 指定已有潜变量记录；尚未接入 T5
编码器来处理任意新提示词。合成数据的测试只检查 I/O 和训练流程，不能说明
真实 T5/ELF 的生成质量。

后续工作与验证范围见 [ALIGNMENT.md](ALIGNMENT.md)。
