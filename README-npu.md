# NPU (Ascend) 环境说明

本机为 Ascend 910（aarch64，CANN 9.0.0，2 卡 × 61.27 GiB HBM）。项目上游默认面向
NVIDIA CUDA，这里的 `.venv312` 已完成 NPU 适配，并且**五种任务全部端到端验证通过**。

| 项 | 版本 |
| --- | --- |
| Python | 3.12.16（deadsnakes PPA） |
| torch | 2.14.0+cpu（PyTorch 官方 CPU 源） |
| torch_npu | 2.14.0 |
| torchvision | 0.29.2+cpu（**必需**，见下文） |
| CANN | 9.0.0 |
| 其余依赖 | 与 `requirements.txt` 完全一致 |

## 快速使用

```bash
cd /workspace/Pika-H3-Flex
source .venv312/bin/activate

pika-h3-pro --task t2va --prompt "a red ball bouncing on a wooden table" \
  --revision 0ddf55ad7067ca1e76812253bd7b4600df87f759 \
  --base-path models/MiniMax-H3-ref \
  --adapter-path models/Pika-H3-Flex/fl2va \
  --resolution 480p --duration 5 --seed 42 --low-vram \
  --output outputs/sample.mp4
```

**NPU 上必须加 `--low-vram`**：文本编码器 62.13 GiB、transformer 61.70 GiB 都超过
单卡 61.27 GiB，不做卸载无法运行。详见下文。

双卡可以用 `--tensor-parallel 2` 走 HCCL 权重张量并行，480p/5s 下比单卡快 30.7%，
详见 [双卡张量并行](#双卡张量并行tp)。
## 实测结果（480p / 5s / 124 帧 / 24fps）

单卡（默认线程数）：

| 任务 | 权重分支 | 数值策略 | 总耗时 | 条件峰值 | 采样峰值 | 解码峰值 |
| --- | --- | --- | --- | --- | --- | --- |
| `t2va` | transformer | bf16 | 131 s | 1.45 GiB | 6.22 GiB | 11.65 GiB |
| `i2va` | transformer | bf16 | 211 s | 10.66 GiB | 6.43 GiB | 11.65 GiB |
| `l2va` | transformer | bf16 | 182 s | 10.66 GiB | 6.43 GiB | 11.65 GiB |
| `fl2va` | transformer | bf16 | 155 s | 10.66 GiB | 6.61 GiB | 11.65 GiB |
| `ref2va`（图像参考） | transformer_ref | fp32 | 176 s | 10.66 GiB | 6.70 GiB | 11.65 GiB |
| `ref2va`（图+视频+音频） | transformer_ref | fp32 | 279 s | 15.53 GiB | 10.63 GiB | 11.65 GiB |

单卡限制线程后重测 `t2va`：16 线程 **125.0 s**（线程争抢主要影响加载阶段，
17.5 s → 4.9 s）。

### 去噪阶段满载率与功耗（2026-10-31 实测）

`--low-vram` 的 block offload 在 NPU 上现在走 side-stream 预取（不再每次同步卸载到
pageable 内存）。480p/5s `t2va`，msprof task 时间线 + npu-smi 1 Hz：

| 指标 | 改动前 | 改动后 |
| --- | ---: | ---: |
| 去噪总时长 | 109.97 s | **39.9 s** |
| 稳态步长 | 10.77 s | 6.26 s |
| NPU 计算忙碌率 | 46.8 % | **89.2 %** |
| 满载占比（100 ms 分箱 ≥90 %） | 9.8 % | **77.7 %** |
| 满载占比（npu-smi 1 Hz ≥90 %） | 32.7 % | **83–90 %** |
| 平均功耗 / 峰值 | 265 W / 494 W | **458 W / 545 W** |
| 每帧能量 | 235 J | **147 J** |

输出与改动前**逐位相同**（同 seed 的 mp4 md5 一致）。纯计算下限实测约 6.3 s/步
（全驻留、零拷贝），即搬运已基本完全隐藏在计算之后，剩余时间是 kernel 本身。

代价是 transformer 加载阶段多花约 7 s（把 62 GiB 权重在主机侧 pin 一份），
一次性投入、6 步内即回本。768p 实测采样 181.3 s → **138.8 s**（−23 %），
采样峰值 11.33 → 12.56 GiB。详见 [docs/npu-denoise-utilization.md](docs/npu-denoise-utilization.md)。

`PIKA_FUSED_RMSNORM=1` 可再快 9.8 %（融合 RMSNorm），但会破坏逐位一致，默认关闭。

采样阶段的满载率与空闲归因（msprof + probe 实测）见
[docs/npu-sampling-idle-profile.md](docs/npu-sampling-idle-profile.md)：稳态 kernel
忙碌率 **92.8–96.4 %**，block 权重预取 **96 % 与计算重叠**、同步 D2H 为 0；剩下的空档
集中在 **10 个非 block 模块（AdaLN 投影 / norm_out / rope 等，1.69 GiB fp32）的同步
H2D**（45–244 ms/步，随机器状态波动）与 float64 `position_ids` 的逐前向 cast
（13–24 ms/步）。去噪均值功率 449 W、峰值 534 W、每帧 145 J。

### 视频 VAE 解码（2026-11-02）

解码原先是整段 fp32、全部空间 tile、全部时间 chunk 一次性算完再拼成整段像素：
832×480/5 s 要 16 s（加 2.6 s 编码），1920×1088/10 s 要 **157 s**（加 11 s 编码），峰值 22.4 GiB，而且这段里
第二张卡完全空闲。现在改成流式 + 分片：

| 改动 | 效果 |
| --- | --- |
| fp16 autocast（**上游官方 recipe**：官方 pipeline 就是这样把 fp32 权重转成 fp16 算的） | 解码 2.2× |
| 空间 tile 按 batch 送进 ViT（默认 4） | fp16 下再快 20–25 % |
| 按时间 chunk 流式产出，边解码边 x264 编码 | 峰值显存 22.4 → **17.6 GiB**（1080p），编码 11 s 完全藏进解码 |
| 生成器内进入 `no_grad` | 此前 `@torch.no_grad()` 只覆盖创建生成器的调用，整个 decode 仍建 autograd 图 |
| TP 下按 chunk 把 clip 切给各 rank，帧经 HCCL 送回 rank 0 | 解码阶段也变成 2 卡并行 |

| 场景（解码 + 编码墙钟） | 改动前 | 单卡 | TP2 |
| --- | ---: | ---: | ---: |
| 832×480 / 5 s | 18.6 s | **7.8 s** | **6.8 s** |
| 1344×768 / 5 s | 40.3 s | **17.0 s** | **13.3 s** |
| 1920×1088 / 10 s | 168 s | **68.0 s** | **59.5 s** |

纯解码（不含编码、不含模型加载）：480p/5s 16.0 → **6.7 s**，768p/5s 36.7 → **15.3 s**，
1080p/10s 157.3 → **65.7 s**；1080p 峰值显存 22.4 → **17.6 GiB**。

（1080p 的 TP2 收益（−12.5 %）小于 768p（−22 %）：1080p 时单卡解码
已经和 x264 完全重叠，TP 反而要额外搬 ~0.9 GiB uint8 帧（HCCL 实测 70 GB/s，
3 s 左右），而且两卡的 chunk 数只有 7/7，负载不均的尾巴更明显。）

**精度**：fp16 不是逐位一致。随机 latent 上，与 fp32 逐像素比较 max diff ≤ 2/255、
平均 0.04/255、PSNR 62 dB；**真实 latent 的端到端实测更保守一些**：max diff **7/255**、
平均 0.031/255、PSNR **63.3 dB**（99 % 的像素差 ≤ 7，仅 0.02 % 的像素差 > 16）。
tile batch 只影响第 8 位的舍入，≤ 1/255。
需要和旧版**逐位相同**的产物时用 `--vae-decode-dtype fp32`（会关闭 tile batch，
因为 per-tile 的 fp32 路径与上游 bit-exact，已实测 480p/768p/1080p 全部一致，
真实 latent 上 max diff 也是 **0**）。
WebUI 的“VAE 解码精度”下拉框对应同一开关；`PIKA_VAE_DECODE_DTYPE` 环境变量也可覆盖，
`PIKA_VAE_TILE_BATCH` 调整 tile 批量（`1` 即逐 tile，与上游逐位一致）。
输出 JSON 记录 `vae_decode_dtype`、`vae_decode_tile_batch`、`vae_decode_chunks` 与
`vae_decode_chunk_ranges`，便于事后确认某次产物用的是哪条路径。

TP2 的 md5 与单卡**完全相同**（同一份 latent，只是把 chunk 分给两张卡），
所以分片本身不引入任何数值差异。

**端到端实测（1344×768 / 5 s / seed 42，无 profiler，基线 = VAE 优化前的 HEAD）**：

| 阶段 | 优化前 | 优化后 | 加速比 |
| --- | ---: | ---: | ---: |
| 采样 | 138.90 s | 138.38 s | 1.00×（与 VAE 无关） |
| **解码 + 编码** | **43.41 s** | **22.60 s** | **1.92×** |
| 端到端 | 212.82 s | 189.33 s | **1.12×** |

同一份 latent 只换解码器：`--vae-decode-dtype fp32` 与旧版**逐像素相同**（max diff 0）；
默认 fp16 的帧级差异 max 7/255（PSNR 63.3 dB）。端到端 mp4 的 max 68 是 x264 对
输入微扰的级联放大，不是解码误差。

音频 VAE 仍然整段 fp32 解码（0.56 GiB，1–2 s），视频帧先入 MP4，音频在
`_Mp4Writer.close()` 里统一 mux，保证音轨从 0 开始对齐。

**全程峰值 15.53 GiB，单卡 61.27 GiB 余量充足。** 输出均为 832×480 H.264 +
AAC 32 kHz 双声道。

已验证的画面正确性：`i2va` 首帧与输入图一致；`l2va` 尾帧一致；`fl2va` 首帧闭盖、
尾帧开盖且中间自动过渡；`ref2va` 保留参考主体外观；混合参考能迁移参考视频的镜头
角度与场景光照。

## HBM 不足时的 CPU 卸载

### 为什么必须卸载

| 组件 | 权重大小 | 单卡可用 | 能否常驻 |
| --- | --- | --- | --- |
| Qwen3-VL 文本编码器 | 62.13 GiB | 61.27 GiB | ❌ 单独就装不下 |
| transformer / transformer_ref | 61.70 GiB | 61.27 GiB | ❌ 装下亦无激活空间 |
| 视频 VAE (fp32) | 9.70 GiB | 61.27 GiB | ✅ 但会挤占采样空间 |
| 音频 VAE (fp32) | 0.56 GiB | 61.27 GiB | ✅ 常驻 |

主机内存约 2 TB，CPU 侧存放全部权重无压力。

### `--low-vram` 做了什么

一个开关同时启用三条卸载路径（`--cpu-offload` 现在等价于 `--low-vram`）：

| 组件 | 策略 | 实测峰值 |
| --- | --- | --- |
| Qwen3-VL | `apply_group_offloading(..., offload_type="leaf_level")`，逐层流式 | 1.45 GiB |
| transformer | `enable_group_offload(..., offload_type="block_level", num_blocks_per_group=1)`，**NPU 上用 side stream 预取** | 6.2–10.6 GiB |
| 视频 VAE | 解码前整模块 `.to(device)`，解码后 `.to("cpu")` + `empty_cache` | 11.65 GiB（含解码激活，旧版整段解码；流式后 480p 11.3 / 1080p 17.6 GiB） |
| 音频 VAE | 常驻设备（太小，搬运不划算） | 0.86 GiB |

不带 `--low-vram` 时行为与上游完全一致（全量驻留），CUDA 路径不受任何影响。

transformer 的流式卸载在 NPU 上默认启用 side stream 预取：权重在主机侧保留一份 pinned
副本，卸载只是把 `param.data` 指回去（没有 D2H），上载在独立 stream 上与该块计算重叠。
这是去噪从 110 s 降到 40 s 的原因，详见
[docs/npu-denoise-utilization.md](docs/npu-denoise-utilization.md)。

### 两个必须避开的坑

1. **accelerate 的 `cpu_offload_with_hook` 在这里不可用。**
   参考实现 `/workspace/vdn-minimax-h3` 用它处理 VAE，但 module hook 只挂
   `Module.forward`，而 diffusers 的 VAE 通过 `decode()` / `encode()` 调用，内部直接
   访问子模块，hook 从不触发，于是权重留在 CPU、输入已在 NPU：
   `RuntimeError: Expected all tensors to be on the same device`。

2. **音频 VAE 不能做 leaf 级卸载。**
   它用 `torch.nn.utils.weight_norm` 参数化卷积权重，实际计算用的 `weight` 是运行时
   算出来的，不是 leaf parameter，leaf hook 搬运 `weight_g`/`weight_v` 而漏掉
   `weight`，同样触发设备不一致。该 VAE 只有 0.56 GiB，直接常驻设备最省事。

transformer 也不能按 leaf 卸载：其融合 kernel 直接读取子模块权重而不调用子模块，
leaf hook 不会触发。这与上游 `docs/diffusers.md` 的描述一致。

## 双卡张量并行（TP）

`--tensor-parallel 2` 用 HCCL 把 transformer 按权重切成两份，实现参考
`/workspace/vdn-minimax-h3` 的 `ascend-npu-a3` 分支（权重 TP，不是 Ulysses 序列
并行——后者依赖 Triton kernel，NPU 上不可用）。

> **2026-10-30 修复**：早期版本的 TP 切分丢弃了 LoRA 适配器，`--tensor-parallel 2`
> 实际运行的是未微调的 base 模型，ref2va 等任务画质严重劣化。现已修复（见
> [一致性](#一致性)）。若手上有更早的 TP 产物，请重新生成后再评估质量。

```bash
.venv312/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m pika_h3.cli --task t2va --prompt "a red ball bouncing" \
  --revision 0ddf55ad7067ca1e76812253bd7b4600df87f759 \
  --base-path models/MiniMax-H3-ref \
  --adapter-path models/Pika-H3-Flex/fl2va \
  --resolution 480p --duration 5 --seed 42 \
  --tensor-parallel 2 --low-vram \
  --output outputs/tp.mp4
```

### 切分方式

| 位置 | 切分 | 通信 |
| --- | --- | --- |
| Q / K / V | 按 head（56 → 28/卡） | 无 |
| attention 输出投影 | 按输入通道 | 1 次 FP32 `all_reduce` |
| FFN 第一层 | SwiGLU 两个半区同步切 | 无 |
| FFN 第二层 | 按输入通道 | 1 次 FP32 `all_reduce` |
| AdaLN 投影 | 按输出 | 小表 `all_gather` |
| **LoRA `lora_B`** | 随 base 按输出行切 | 无 |
| **LoRA `lora_A`** | 随 base 按输入列切 | 随所在层的 reduce |
| 残差流 | 每卡完整 | 无 |

每层 2 次 all-reduce，没有序列级通信。每卡权重 **34.93 GB（含适配器，原模型的 51.6%）**。

适配器必须随 base 一起切分：PEFT 的 `LoraLayer` 在 `forward` 里计算
`base(x) + scaling * B(A(x))`，而切分后的 shard 模块直接读 `.weight`，所以
`_LoraFactors` 把 A/B 因子挂到 shard 上并由 shard 自己应用，否则适配器会丢失
（见上文「一致性」的修复记录）。

### 实测性能

#### 480p / 5s / 124 帧

| 阶段 | 单卡 640 线程 | 单卡 16 线程 | TP + 广播 + 16 线程 |
| --- | ---: | ---: | ---: |
| 资产校验 | 2.5 s | 2.2 s | 2.1 s |
| 条件编码 | 27.9 s | 19.7 s | **17.8 s** |
| transformer 加载 | 17.5 s | 4.9 s | 7.0 s |
| 采样 | 77.8 s | 77.2 s | **38.4 s** |
| 解码 + 编码（旧） | 20.1 s | 21.2 s | 21.5 s |
| 解码 + 编码（流式后） | **7.8 s** | **7.8 s** | **6.8 s** |
| **总计（流式后）** | **133.6 s** | **111.6 s** | **71.9 s** |

| 显存峰值 | 单卡 | TP |
| --- | ---: | ---: |
| 条件编码 | 1.45 GiB | 1.45 GiB |
| 采样 | 6.21 GiB | **4.20 GiB** |
| 解码（旧整段 fp32） | 11.65 GiB | 11.65 GiB |
| 解码（流式 fp16） | 15.7 GiB | 15.7 GiB |

**TP 相对公平基线（单卡 16 线程）快 30.7%，采样加速 2.01×。**

#### TP 自适应权重放置（2026-10-31）

TP 分片后每卡只有 34.93 GB 权重（51.6 %），**多数请求下不需要逐块卸载**。
`--low-vram` 现在自适应：权重 + 本次请求的激活装得下就常驻，装不下才流式。
两个分支的算术完全相同，**输出逐位一致**（同 seed 四次运行 mp4 md5 相同）。

判定必须同时看激活，只看权重会翻车：`1920x1088 / 10s` 权重 32.5 GiB 看着宽裕，
但激活约 22 GiB，采样第 3 步就 OOM。激活系数由两个实测点反推（14.3 / 14.5，取 15），
预算按 `min(free, total) × 0.90`。

480p / 5s / t2va：

| 指标 | TP + 流式卸载 | TP + 自适应（常驻） |
| --- | ---: | ---: |
| 采样 | 27.2–27.9 s | **22.0–23.1 s**（−18 %） |
| 采样峰值 | 5.04 GiB | 34.79 GiB |

大画布仍会自动回退到流式卸载（实测 `1920x1088 / 10s` → `transformer_offloaded: true`，
811 s 正常完成、峰值 24.13 GiB、无 OOM），见
`docs/npu-denoise-utilization.md` 的自适应判定表。

> 上述耗时是在适配器丢失 bug 存在时测的，但**耗时数据仍然有效**：切分/通信的
> 工作量由 base 权重决定，适配器只增加 0.87 GB 参数（34.06 → 34.93 GB）和
> 每层两次小 GEMM，占比不足 1%，不改变加速比。修复后如需精确数字可重测。

#### 768p / 5s / 124 帧（1344×768）

| 阶段 | 单卡 | TP | 变化 |
| --- | ---: | ---: | ---: |
| 资产校验 | 2.6 s | 2.4 s | −6.7% |
| 条件编码 | 20.8 s | 38.1 s | +83.1% |
| transformer 加载 | 4.8 s | 9.1 s | +88.9% |
| 采样 | 181.3 s | **94.7 s** | −47.8% |
| 解码 + 编码（旧） | 41.6 s | 42.4 s | +2.0% |
| 解码 + 编码（流式后） | **17.0 s** | **13.3 s** | −58.9% / −68.6%（对比旧单卡 / 旧 TP） |
| **总计（流式后）** | **226.4 s** | **157.6 s** | **−30.4%** |

| 显存峰值 | 单卡 | TP |
| --- | ---: | ---: |
| 条件编码 | 1.45 GiB | 1.45 GiB |
| 采样 | 11.33 GiB | **7.18 GiB** |
| 解码（流式 fp16） | 16.1 GiB | 16.1 GiB |

采样加速 **1.91×**。768p 端到端收益（25.6%）反而低于 480p（30.7%）：采样的收益更大，
但广播开销随 packed token 数线性增长，条件编码从 20.8 s 涨到 38.1 s（+17.3 s），
抵消了一部分。

#### 分辨率与 TP 收益

| 分辨率 | 采样占比 | 采样加速 | 端到端提升 | 采样峰值（单卡 → TP） |
| --- | ---: | ---: | ---: | --- |
| 480p | 58% | 2.01× | **−30.7%** | 6.21 → 4.20 GiB |
| 768p | 72% | 1.91× | −25.6% | 11.33 → 7.18 GiB |

#### `ref2va`

同样画布下 175.7 s → 151.3 s（**−13.9%**，采样加速 1.58×，采样峰值
6.70 → 4.48 GiB）。它比 `t2va` 收益低，原因有二：`fp32_conditioning_v1`
让广播数据量翻倍，参考图还要额外经过 VAE 编码；参考行拉长了序列，all-reduce
通信占比随之升高。

### 两个必要的配套改动

1. **条件编码只在 rank 0 执行，然后广播。** 文本编码器和 VAE 是复制的，
   两卡各跑一遍纯属浪费（实测 57.2 s → 19.0 s）。广播走逐元素 HCCL
   `broadcast`，接收端拿到的值与源端**逐位相同**，不是重新编码。
   注意 HCCL 只能广播设备张量，payload 需临时搬到 NPU 再取回。

2. **必须限制每个 rank 的 CPU 线程数。** `torchrun` 只设 `OMP_NUM_THREADS`，
   PyTorch 自己的 intra-op 池仍按可见核心数创建（本机 640 线程）。两个 rank
   合计 1280 线程争抢 CPU，加载从 8 s 恶化到 87.9 s、采样从 38.9 s 恶化到
   76.3 s。`_limit_cpu_threads()` 取「`PIKA_CPU_THREADS`（默认 16）、CPU 亲和性、
   cgroup 配额」三者的最小值。该限制在单卡路径同样生效——不限制时单卡加载也要
   17.5 s，限制后 5.0 s。

   ```bash
   PIKA_CPU_THREADS=32 pika-h3-pro ...   # 换设备后按可用核数调整
   ```

### 一致性

TP 相对单卡的数值误差来自 BF16 局部 GEMM 在 FP32 规约前的舍入，量级取决于是否
携带适配器：

| 场景 | 相对平均误差 | 说明 |
| --- | ---: | --- |
| 无适配器的 base 模型 TP2 vs TP1 | 1.29% | 纯 split-K 底噪 |
| **带 LoRA 适配器 TP2 vs TP1（修复后）** | 1.6e-3 | 与底噪一致，适配器已精确切分 |
| 带适配器 TP2 vs TP1（修复前，**有 bug**） | **9.66%** | 适配器被丢弃 |

> **重要修复记录**：早期版本的 `install_tensor_parallel()` 只切分 `linear.weight`，
> 而 ref2va 的投影是 PEFT `LoraLayer`，其真实权重由 `forward` 里的
> `base + scaling*B(A(x))` 组成，`.weight` 只返回 base。切分后 shard 模块替换了
> wrapper，**适配器被静默丢弃**，TP=2 实际运行的是未微调的 base 模型。
>
> 结构化证据：LoRA 参数 `624 → 24`（只有不参与切分的 `token_refiner` 幸存）。
> 数值证据：旧 TP2 输出距离「无适配器 base 模型」比距离正确输出近 5.6 倍。
> 视觉证据：同一 prompt 下 480p 首帧亮度 50.0 vs 正确 73.6，画面扭曲。
>
> 现在 `_LoraFactors` 会把 `B` 按输出行、`A` 按输入列随 base 一起切分，走同一
> FP32 all-reduce，数学上精确；DoRA / LoRA-FA / 带 bias 的变体会被明确拒绝而不是
> 近似处理。**修复前所有 `--tensor-parallel 2` 的产物都不反映模型真实能力。**

这与参考仓库的结论一致：其文档明确说明默认 parity **未通过近无损门限**
（`FAILED the near-lossless gate`），只有 opt-in 的 `row_gather`/`row_stream`
模式才做到 bitwise。本实现属于默认 parity 路线。

同种子在 TP 下可复现（两卡初始噪声逐位相同已实测验证）。

## NPU 适配原理

项目不依赖全局 monkey-patching。所有设备访问统一经过
`src/pika_h3/core/accelerator.py`（`accelerator`），把设备选择、`empty_cache`、
显存统计、stream、`manual_seed`、fork RNG、精度标志路由到选定后端：

1. 环境变量 `PIKA_DEVICE` = `cuda` | `npu` | `cpu`（大小写不敏感）优先；
2. 未设置时，若 `torch_npu` 可导入且 `torch.npu.is_available()`，选 `npu`；
3. 否则若 `torch.cuda.is_available()`，选 `cuda`；
4. 否则 `cpu`。

`torch_npu` 的导入有 try/except 保护；纯 CUDA 环境不安装 torch_npu 时静默失败，
CUDA 路径照常工作。**CUDA 行为保持不变**：后端解析为 `cuda` 时，
`torch.backends.cuda.matmul.allow_tf32=False`、`torch.backends.cudnn.allow_tf32=True`、
CUDA stream 与 `enable_group_offload(use_stream=True)` 都与原实现一致。

```bash
PIKA_DEVICE=npu pika-h3-pro ...   # 强制 Ascend NPU
PIKA_DEVICE=cuda pika-h3-pro ...  # 强制 CUDA（不可用时直接报错）
PIKA_DEVICE=cpu  pika-h3-pro ...  # 纯 CPU（仅用于导入/CLI 检查）
```

旧的 `torch_npu.contrib.transfer_to_npu` 全局钩子改为显式 opt-in，仅在
`PIKA_NPU_SHIM=1` 时启用（兼容尚未迁移的第三方代码）：

```bash
export PIKA_NPU_SHIM=1     # 启用旧的 CUDA->NPU 全局映射
export PIKA_DISABLE_NPU=1  # 即使 PIKA_NPU_SHIM=1 也强制关闭
```

不设置时钩子不做任何事，`torch.cuda.is_available()` 在本机保持 `False`，
代码可以真实区分 CUDA / NPU。

## 从零复现

### 1. Python 3.12

Ubuntu 22.04 官方源没有 3.12，使用 deadsnakes PPA：

```bash
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.12 python3.12-venv python3.12-dev git patch ffmpeg
```

### 2. 创建 venv + 镜像

```bash
cd /workspace/Pika-H3-Flex
/usr/bin/python3.12 -m venv .venv312
printf '[global]\nindex-url = https://mirrors.cloud.tencent.com/pypi/simple\ntimeout = 60\n' > .venv312/pip.conf
```

### 3. torch / torch_npu / torchvision（关键）

**必须使用 PyTorch 官方 CPU 源**。PyPI 上 `torch==2.14.0` 是 CUDA 13 构建，
与 `torch_npu` 同时加载会因 allocator 配置键重复而直接 abort：

```
ValueError: Duplicated key 'pinned_reserve_segment_size_mb'
```

```bash
source .venv312/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0+cpu"
pip install --no-deps "torch_npu==2.14.0"
pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "torchvision==0.29.2+cpu"
```

两个 `--no-deps` 都不可省略：否则依赖解析会把 PyPI 的 CUDA 版 torch 拉回来。

**torchvision 是必需依赖**，虽然项目源码没有直接 import，但 transformers 的
`Qwen3VLVideoProcessor` 在加载 processor 时强制要求它，缺失会报：

```
Qwen3VLVideoProcessor requires the Torchvision library but it was not found
```

### 4. 其余依赖

```bash
pip install -r requirements-npu.txt
pip install https://codeload.github.com/huggingface/diffusers/tar.gz/3a2f35d4efa4c059c8bfb3bc0d6c906264895c81
python tools/apply_diffusers_patches.py
pip install --no-deps -e .
```

### 5. NPU 启动钩子（默认禁用）

```bash
# .venv312/lib/python3.12/site-packages/pika_npu_setup.py
# .venv312/lib/python3.12/site-packages/pika_npu.pth  ->  import pika_npu_setup
```

钩子默认不做任何事，只有 `PIKA_NPU_SHIM=1` 时才导入
`torch_npu.contrib.transfer_to_npu`。项目本身通过 `accelerator` 显式选择设备，
不需要该钩子。

注意：**不要**把钩子写成 `.venv312` 内的 `sitecustomize.py`——`/usr/lib/python3.12/sitecustomize.py`
会优先被加载，venv 内的同名文件被静默忽略。因此这里用 `.pth` 文件。

## 模型权重

`models/MiniMax-H3-ref` 约 196 GB，`models/Pika-H3-Flex` 约 5.5 GB。

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 适配器（注意：每个 pattern 必须单独一个 --include）
hf download Pika-Labs/Pika-H3-Flex \
  --revision 0ddf55ad7067ca1e76812253bd7b4600df87f759 \
  --include "fl2va/*" --include "ref2va/*" \
  --local-dir models/Pika-H3-Flex --max-workers 8

# 基座（ref2va 需要 transformer_ref，其余任务需要 transformer）
hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include "transformer/*" --include "transformer_ref/*" \
  --include "text_encoder/*" --include "tokenizer/*" --include "processor/*" \
  --include "vae/*" --include "audio_vae/*" --include "*.json" \
  --local-dir models/MiniMax-H3-ref --max-workers 8
```

`hf` CLI 的 `--include` 一次只接受一个值；写成 `--include "a" "b"` 会让 `b` 变成
位置参数、整条 `--include` 被忽略（`UserWarning: Ignoring --include since filenames
have been explicitly set`）。实测 hf-mirror 约 10 MB/s。

不要下载顶层 `FL2VA/`、`Ref2VA/`（各 144 GB），项目不使用。

## 验证

```bash
source .venv312/bin/activate

python -c "import torch; print(torch.cuda.is_available(), torch.npu.is_available())"
# False True    ← 显式代码选择 NPU，未被全局改写

python -c "from pika_h3.core import accelerator; print(accelerator.backend(), accelerator.device_count())"
# npu 2

PIKA_NPU_SHIM=1 python -c "import torch; print(torch.cuda.is_available())"
# True

python -c "import torch_npu; print(torch_npu.npu.utils.get_cann_version())"
# 9.0.0

python tools/apply_diffusers_patches.py --check
# Diffusers H3 patches verified (0 applied)

pika-h3-pro --help
```

TP 切分自检（不需要跑生成，确认适配器随切分保留）：

```bash
.venv312/bin/python - <<'PY'
import torch
from diffusers import MiniMaxH3Transformer3DModel
from pika_h3.core.loader import apply_lora, load_lora_weights
from pika_h3.core.tensor_parallel import install_tensor_parallel

m = MiniMaxH3Transformer3DModel.from_pretrained(
    "models/MiniMax-H3-ref", subfolder="transformer_ref",
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
apply_lora(m, rank=128, alpha=128, master_dtype=torch.bfloat16)
load_lora_weights(m, "models/Pika-H3-Flex/ref2va/adapter.safetensors")

before = sum(1 for mod in m.modules()
             if getattr(mod, "lora_A", None) is not None or hasattr(mod, "lora_A"))
install_tensor_parallel(m, 0, 2)
carried = sum(1 for mod in m.modules() if getattr(mod, "lora", None) is not None)
print(f"切分前带适配器的层 {before} -> 切分后携带因子的 shard {carried}")
PY
# 期望：312 -> 300（token_refiner 的 12 层不参与切分，仍留在原地）
```

注意不能在切分后用 `"lora_" in name` 计数：`_LoraFactors` 把参数重命名为
`lora.a` / `lora.b`，过滤器匹配不到。切分后每 rank 的适配器元素数约为原来的
2/3（A 按输入列、B 按输出行各取一半），这是正确的。

## 已知注意事项

1. **启动告警**：torch_npu 2.14 打印
   `Warning: the current CANN and Soc versions require processing for 32 padding size`，
   以及 transformers 关于 `min_frames`/`max_frames` 未写入 docstring 的 ERROR 日志，
   均为上游日志噪声，不影响功能。
2. **CANN 版本**：torch_npu 2.14 的部分新特性（如 `FORCE_OVERFLOW_CHECK`）要求
   CANN 9.1.0+；当前 9.0.0 下不可用，主流程不受影响。
3. **CPU 往返开销**：`--low-vram` 下文本编码条件耗时明显高于纯 GPU 实现
   （t2va 19.5 s，i2va 68.3 s，主因是视频 VAE 在 CPU/NPU 间搬运），属于预期代价。
4. **torch 2.14 vs 上游 2.13**：数值行为可能与官方 H100 结果有细微差异；
   同 seed 在 NPU 上可复现。
5. **仅验证 480p / 5 s**：768p 与 15 s 未实测，显存有余量但耗时更长。
6. **pip 镜像**：系统 `/etc/pip.conf` 指向清华源，当前返回 403。`.venv312` 已写入
   腾讯云镜像配置（`pip.conf`），venv 内装包无需额外指定 `-i`。
