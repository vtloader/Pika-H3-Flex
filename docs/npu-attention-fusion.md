# Attention 优化：为什么内核到顶了，以及唯一还能无损拿到的 3–5 %

> 2026-11-02。目标：在**不改变任何输出位**的前提下降低采样阶段的 attention 开销。
> 测试口径 1344×768 / 5 s / seed 42（768p），单卡与 TP2 各测。

## 1. 结论

attention 内核本身**没有余量**了。唯一还能无损拿到的，是它周围的 per-head QK RMSNorm：

| | 单卡采样 | TP2 采样 | 产物 |
| --- | ---: | ---: | --- |
| 关闭融合 | 138.20 ± 0.67 s | 83.11 s | — |
| 开启融合（默认） | **134.17 ± 0.69 s**（−2.9 %） | **78.99 s**（−5.0 %） | md5 **完全相同** |

单卡数字是两次运行的平均（138.67 / 137.73 → 134.66 / 133.68）。
两种并行度下开/关的 mp4 都逐位相同（逐像素 max diff 0）。

## 2. attention 内核到顶的证据

### 2.1 吞吐恒定，与形状无关

`FlashAttentionScore` 在 768p 的真实形状（S=37643, H=56, D=128, bf16）上：

| 变量 | 取值 | 吞吐 |
| --- | --- | ---: |
| head 数 | 8 / 16 / 24 / 28 / 32 / 48 / 56 / 72 / 96 | **159.9 – 163.7 TFLOPS** |
| 序列长度 | 2048 / 8192 / 14852 / 24576 / 37643 | **153 – 169 TFLOPS** |
| head_dim | 64 / 128 | 86 / 160 TFLOPS |
| dtype | fp16 / bf16 | 157.7 / 159.7 TFLOPS |
| `inner_precise` | 0 / 1 / 2 / 3 | 158.4 – 160.2 TFLOPS |

吞吐是一条平线。如果瓶颈是分块或负载不均，改 head 数（8 → 96，跨 12 倍）一定会有
台阶——没有。所以限制在 kernel 内部。

### 2.2 但同一颗芯片 matmul 能跑 323 TFLOPS

| 形状 | TFLOPS |
| --- | ---: |
| 37643 × 7168 × 5376 | **323.3** |
| 37643 × 14336 × 5376 | 314.9 |
| 16384³ | 251.1 |
| **37643 × 37643 × 128**（FA 的 QK^T 就是这个形状） | **52.7** |

最后一行是关键：**K=128 的 matmul 本身只有 52.7 TFLOPS**，而 FA 在同样 K=128 的
两个乘加上做到 159 TFLOPS —— 已经是等价 matmul 的 **3 倍**。缺的这 50 %（相对
323 TFLOPS）来自 D=128 时 cube 无法被喂饱，不是调度问题。

`cube_utilization` 在 profiler 里是 **98 %**，也印证了这一点：cube 没有闲着，
是每次乘加的 K 太小。

### 2.3 换 CANN 入口也没用

| 入口 | 768p 耗时 | 相对 SDPA |
| --- | ---: | ---: |
| `F.scaled_dot_product_attention`（当前） | 256.1 ms | 1.00× |
| `npu_fusion_attention` BNSD, pre=max | 256.1 ms | 1.00× |
| `npu_prompt_flash_attention` | 263.0 ms | 0.97× |
| `npu_fusion_attention` BSND（diffusers 的 `_native_npu` 路径） | 329.6 ms | 0.78× |
| `npu_fusion_attention_v3` | 329.1 ms | 0.78× |
| `npu_fused_infer_attention_score` | 255.2 ms | 1.00×，但**差 2.4e-4** |

diffusers 默认的 `native` 后端（走 SDPA）恰好就是最快的。它的 `_native_npu` 后端
反而慢 22 %，所以**不要**设 `DIFFUSERS_ATTN_BACKEND=_native_npu`。

### 2.4 双流拆 head 也不行

softmax 跑在 AIV 上、cube 跑矩阵乘，直觉上把 head 拆成两组放到两条 stream 上能让
B 组的 cube 填 A 组的 softmax 空档：

| | 耗时 | 相对 |
| --- | ---: | ---: |
| 1 次调用 56 heads | 252.9 ms | 1.00× |
| 2 组 / 2 stream | 253.2 ms | 1.00× |
| 4 组 / 2 stream | 250.7 ms | 1.01× |
| 2 组 / 串行（对照） | 254.5 ms | 0.99× |

没有收益，且结果仍逐位相同。说明 kernel 内部已经把 AIV 和 cube 流水起来了。

## 3. 唯一可用的融合：per-head QK RMSNorm

### 3.1 为什么是它

一个 block 的耗时分解（768p，真实权重）：

| 部分 | 耗时 | 占 block |
| --- | ---: | ---: |
| 整个 block | 416.7 ms | 100 % |
| attention 子模块 | 327.0 ms | 78.5 % |
| └ FA kernel | 257.5 ms | 61.8 % |
| └ to_q/to_k/to_v | 26.9 ms | 6.5 % |
| └ **norm_q（ATen）** | **8.4 ms** | 2.0 % |
| └ **norm_q（融合）** | **0.92 ms** | 0.2 % |
| feed-forward | 65.5 ms | 15.7 % |
| norm1 / norm2 | 6.4 / 6.4 ms | 1.5 / 1.5 % |

`nn.RMSNorm` 在 Ascend 上会拆成 5 个 AIV kernel（`Pows`、`ReduceMean`、`Muls`、
`Rsqrt`、`Mul`），profiler 里能看到它们各占一档；`npu_rms_norm` 合成一个，
**23.4 → 0.96 ms（24.4×）**。每个 block 有 norm_q 和 norm_k 两个，50 层 × 6 步
共 600 次。

### 3.2 为什么它逐位一致，而全宽 norm 不是

这是本次最重要的发现。同一个 `npu_rms_norm`，在两种宽度上表现完全不同：

| 归一化宽度 | 调用 | 与 ATen 的关系 |
| --- | ---: | --- |
| **128**（head_dim） | 23.4 → 0.96 ms | **逐位相同**，24/24 随机张量 + 真实 Q/K |
| 5376（hidden_size） | 6.4 → 0.88 ms | 差 **0.0312**（约 1 个 bf16 ulp） |

128 通道时两种实现的累加顺序恰好一致；5376 通道时归约树不同，落在一个 ulp 上。
六步一致性采样会把 1 ulp 放大成肉眼可见的差异，所以全宽 norm 必须继续留在
`PIKA_FUSED_RMSNORM`（默认关闭）后面。

验证强度：

- 24 组随机 bf16 张量（12 个 seed × norm_q/norm_k）：`torch.equal` 全 true，max diff 0
- 真实前向里的 Q/K（投影后的真实分布）：`torch.equal` true，max diff 0
- 3 层 block 串联：`torch.equal` true，max diff 0
- **50 层完整前向**（真实 latent 的 768p packed 形状）：`nonzero = 0 / 202,368,768`
- **端到端 mp4**：单卡与 TP2 各自的开/关 md5 完全相同

### 3.3 实现

`core/fused_ops.py::fuse_attention_norms` 只替换同时满足以下条件的模块：

- 模块有 `head_dim` 属性，且
- 有 `norm_q` / `norm_k`，且
- `normalized_shape == (head_dim,)`

所以它碰不到 VAE、text encoder 和 hidden-size norm。`pipeline.py` 里默认调用，
`PIKA_QK_NORM=0` 仅供 A/B 测量。输出 JSON 增加 `fused_attention_norms`（本模型是
50 × 2 = 104）。

### 3.4 为什么 TP2 收益更大（5.0 % vs 2.9 %）

TP2 把 56 个 head 拆成每卡 28 个，FA kernel 的时间减半，但那 5 个小 AIV kernel 的
**启动开销不随 head 数缩小**。分母变小、分子不变，占比自然从 2.0 % 升到约 4 %。

## 4. 试过但放弃的

都不满足"无损"，或者反而更慢。记下来避免重复踩：

| 方案 | 结果 | 原因 |
| --- | --- | --- |
| RoPE → `npu_rotary_mul` | 慢 1.8×，差 1 ulp | 算子要求 D ∈ {64,128}，模型 rotary_dim=96；permute 通道对齐后仍差 0.03125，且 index_select 的开销抵消收益 |
| RoPE → `npu_apply_rotary_pos_emb` | 直接报错 561002 | 同上，D 不合法 |
| SwiGLU → `npu_swiglu` | 快 2.9×，差 0.0625 | value/gate 顺序与 diffusers 相反；swap 后仍不逐位 |
| 残差+门控 → `addcmul` | 快 1.26×，差 0.125 | 融合改变了乘加的结合顺序 |
| AdaLN modulation → `addcmul` | 快 1.19×，差 0.125 | 同上 |
| QKV 投影融合成一次 matmul | 慢 6 % | 3 次 (S,5376)×(7168,5376) 比 1 次 ×3 宽更贴合 cube |
| 双流拆 head | 1.00× | 见 §2.4 |
| `index_select` → `embedding` | 1.03× | 收益在噪声内 |

## 5. 复现

脚本在 `/tmp/opencode/perf/attn/`（`bench_attn.py`、`sweep_attn.py`、
`bench_ceiling.py`、`bench_heads.py`、`bench_streams.py`、`bench_block.py`、
`bench_ablate.py`、`bench_rope.py`、`bench_swiglu.py`、`verify_qk.py`、
`verify_full.py`、`run768.sh`、`run768tp.sh`）。

```bash
# A/B：同一台机器、背靠背
PIKA_QK_NORM=0 bash run768.sh off  /workspace/Pika-H3-Flex/src out_off
PIKA_QK_NORM=1 bash run768.sh on   /workspace/Pika-H3-Flex/src out_on
md5sum out_off/gen/off.mp4 out_on/gen/on.mp4   # 必须相同

# TP2
bash run768tp.sh tp2off   /tmp/out_tp2off   0
bash run768tp.sh tp2fused /tmp/out_tp2fused 1
```

> 注意：TP2 与单卡的 md5 本来就不同（all_reduce 改变了归约顺序），这是既有行为。
> 无损性必须在**同一种并行度**下比开/关。
