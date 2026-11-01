# NPU 去噪阶段满载时间占比与功耗

**结论：去噪时 NPU 不满载的主因不是 kernel 算不动，而是权重卸载路径把 50 % 的时间花在了
同步、pageable 的 D2H 拷贝上。** 把 diffusers 的 stream 预取路径接到 NPU 上之后：

| 指标 | 改动前 | 改动后 | 变化 |
| --- | ---: | ---: | ---: |
| 去噪总时长（6 步） | 109.97 s | **39.9–40.7 s** | **−63 %** |
| 稳态步长 | 10.77 s | 6.20–6.28 s | −42 % |
| NPU 计算忙碌率 | 46.8 % | **89.2 %** | +42.4 pp |
| 满载时间占比（100 ms 分箱 ≥90 %） | 9.8 % | **77.7 %** | +67.9 pp |
| 满载时间占比（npu-smi 1 Hz ≥90 %） | 32.7 % | **83–90 %** | +50–57 pp |
| 去噪平均功耗 | 265 W | **454–458 W** | +71 % |
| 每步能量 | 4859 J | **3040 J** | **−37 %** |
| 每帧能量 | 235 J | **147 J** | −37 % |
| 输出一致性 | — | **逐位相同（md5 相同）** | 无损 |

功耗上去了，能量反而下来了：去噪时间缩短的幅度（2.76×）远大于功率上升的幅度（1.73×），
而 170 W 的 idle floor 被摊在更短的窗口里。

数据：`/tmp/opencode/perf/denoise/T1`（基线）、`T2`（offload 优化）、`T3`（kernel 深挖）、
`T4`（TP2 与自适应常驻）。测量工具：`/tmp/opencode/perf/denoise/tools/run_profile.py`。

> **一句话回答"还能不能更快"**：搬运已经藏进计算里了（§4），单步贴着 6.3 s 的纯计算下限。
> 下一步只能动 kernel：AIV 向量算子占 28 % 且 cube 利用率 0（§4.7），把它们融合掉可以再快
> 约 18 %，**但会破坏逐位一致**。这是当前唯一剩下的实质空间。

---

## 1. 测量方法

三个时钟都对齐到 `time.time()` 微秒（实测 msprof `task_start(us)` 与该时钟同源）：

1. **msprof**（`--task-time=l1 --ai-core=on --aic-mode=task-based --ascendcl=on
   --runtime-api=on --aic-metrics=PipeUtilization`）给出设备 task 时间线与 op 级 cube/mte 比例。
2. **npu-smi 1 Hz**（`npu-smi info watch -i 5 -c 0 -s pnua -d 1`）给出 Pwr、AI Core %、
   NPU Util %、AI Cube %，每行打 `time.time()` 时间戳。采样器在生成进程之外，不干扰步边界。
3. **进程内探针** `tools/probe_entry.py` 包装 `H3DiffusionWrapper.forward`（一次 = 一步），
   前后 `torch.npu.synchronize()`。同时包装 `ModuleGroup.onload_/offload_`：
   阻塞卸载用同步 `aclrtMemcpy`，**不会**在 `task_time` 里产生 `MEMCPY_*` 行，
   不包这一层就完全看不到权重搬运。

派生量（结果里都是显式字段，不是估算）：

- `busy_s` = `AI_CORE` / `AI_VECTOR_CORE` / `MIX_*` task 区间并集；
- `h2d_copy_s` / `d2h_copy_s` = 阻塞 hook 窗口减去其中与计算重叠的部分（真正的拷贝时间）；
- `drain_s` = hook 窗口内等计算跑完的部分（是 `busy_s` 的子集，不是额外开销）；
- `idle_s` = 既无计算也无拷贝的纯 host 停滞；
- 满载占比同时给两种口径：100 ms 分箱的 task 时间线（细）与 1 Hz npu-smi（粗，±1 样本≈10 pp）。

---

## 2. 为什么不满载（改动前的定量拆解）

`--low-vram` 走 diffusers 的**阻塞** block offload：每块 forward 前 H2D、forward 后 D2H，
全部串行。109.97 s 的去噪拆成：

| 项 | 秒 | 占比 |
| --- | ---: | ---: |
| 计算 | 35.10 | 31.9 % |
| 权重 H2D | 18.61 | 16.9 % |
| 权重 **D2H** | 55.22 | **50.2 %** |
| 纯 host 空转 | 1.04 | 0.9 % |

两个叠加的问题：

1. **D2H 落回 pageable 内存。** 阻塞路径每次 `param.data = param.data.to("cpu")`
   都重新分配主机内存（实测 `is_pinned() == False`），卸载带宽只有 **11.5 GB/s**；
   pinned 副本能跑到 42 GB/s。每步 51 次卸载共 57.4 GiB，单这一项就是 ~5.3 s/步。
2. **完全没有重叠。** 卸载是同步的，下一步的 H2D 和计算都得等它做完。

顺带确认：host 空转只有 0.9 %，所以不是 Python/调度开销的问题。

---

## 3. 改动（已落地）

### 3.1 NPU 上用 stream 预取卸载 —— `core/accelerator.py` + `pipeline.py`

`diffusers` 的 `apply_group_offloading(use_stream=True)` 只给 CUDA/XPU 建流，其它后端直接
raise；于是 `supports_streams()` 之前对 NPU 返回 False，永远走阻塞路径。但下游机制本身与后端
无关：`ModuleGroup` 从 `torch.accelerator.current_accelerator().type` 取加速器模块
（此处就是 `torch.npu`），只用到 `stream(...)`、`current_stream()`、`synchronize()`。

新增 `accelerator.enable_group_offload()`：CUDA 仍走 diffusers 自己的入口（行为不变），
NPU 则用 `torch.npu.Stream` 自行构造 `GroupOffloadingConfig` 后交给同一个私有入口
`_apply_group_offloading`，从而复用它的 prefetch 流水线。

stream 模式带来的结构性变化：

- 权重在 CPU 侧保留**一份 pinned 副本**（`low_cpu_mem_usage=False`，一次性 pin ~2 s）；
- 卸载 = 把 `param.data` 重新指回该副本，**D2H 归零**；
- 上载用 `non_blocking=True` 在 side stream 上做，第 i+1 块的 H2D 与第 i 块的计算重叠
  （微基准已确认 NPU 上 H2D 与 matmul 可以真重叠）。

### 3.2 融合 RMSNorm（**默认关闭**，`PIKA_FUSED_RMSNORM=1` 开启）

`nn.RMSNorm` 在 Ascend 上会拆成 `Pows` / `ReduceMean` / `Muls` / `Rsqrt` / `Mul` 五个 AIV
算子；`torch_npu.npu_rms_norm` 一个算子完成，生产形状（15730×5376 bf16）实测
**2.62 ms → 0.36 ms（7.3×）**。

但它是**有损**的：单算子对比与 ATen 差 1e-9，放进真实 block 后相对差 0.2 %
（约 0.5 个 bf16 ulp），而六步一致性采样会把这个 ulp 放大：同 seed 下成片像素平均差 20 %、
逐帧相关 0.90。项目对外承诺逐位一致，所以**默认不启用**，只作为显存/时间更紧时的可选加速。

开启后（`T3/runs/t3_rms_on`）：稳态步长 6.26 → 5.65 s（−9.8 %），cube 利用率 67.9 → 74.2 %。

### 3.3 部分常驻（hybrid）—— 测了，但没有收益

把前 k 个 block 长驻显存、其余照常流式（`T2/patch_hybrid.py`）：

| k | 去噪 | 稳态步长 | 额外显存 |
| ---: | ---: | ---: | ---: |
| 0（纯 stream） | 38.66 s | 6.07 s | 0 |
| 8 | 39.12 s | 6.18 s | +9 GiB |
| 40 | 38.78 s | 6.09 s | +45 GiB |

一旦 H2D 被完全隐藏，常驻就不再改变稳态步长 —— **不推荐**，白占显存。

---

## 4. 现在到顶了吗

单独测"全驻留、零拷贝"时单个 block 的前向成本（`T3/probe_ceiling.py`，真实权重 + 真实
480p packed 布局）：

| blocks | 前向 | 每块 | 折算 50 块 |
| ---: | ---: | ---: | ---: |
| 10 | 1.278 s | 127.8 ms | 6.39 s |
| 25 | 3.169 s | 126.8 ms | 6.34 s |

**纯计算下限 ≈ 6.3 s/步**，而 stream 路径的稳态步长是 6.07–6.26 s —— 也就是说
**搬运已经几乎完全藏进计算里了，剩下的 6 s 就是 kernel 本身**。

单步 op 构成（`T3` 的 op_statistic，稳态步）：

| 类别 | 时间 | 占比 | cube 利用率 |
| --- | ---: | ---: | ---: |
| MatMul（含 attention 的 QKV/O） | 1.93 s | 33 % | 99.0 % |
| FlashAttentionScore | 2.00 s | 34 % | 98.7 % |
| 向量类（Add/Mul/Cast/Slice/Muls/Transpose...） | 1.40 s | 24 % | 0 % |
| 其余（Concat/Gather/Swish...） | 0.5 s | 9 % | — |

所以下一阶段的抓手只有两个，且都在 kernel 层：

1. **把向量算子从关键路径上拿掉**（占 24 %，且 cube 利用率 0）：融合 RMSNorm、
   融合 `add+norm`、减少 `Cast`/`Slice`/`Transpose`；
2. **提高 cube 利用率**：目前 cube 67.9 % 是因为 AIV 段串行插在 cube 段之间。
   融合之后 cube 升到 74.2 %，与 3.2 的 −9.8 % 步长改进吻合。

> 上表是**单卡稳态步**的口径（24 %）。TP2 下同样的 AIV 总量被摊到更长的整轮时间里，
> 占比升到 28 %，绝对秒数见 §4.7 —— 两个数字不矛盾，分母不同。

---

### 4.5 768p 端到端复核（WebUI 实跑）

通过 WebUI 提交同一 prompt（832×480 请求被 WebUI 按 MP 自动选到 p768 档 1344×768），
走的是默认 `--low-vram` 路径（`fused_rmsnorm_modules: 0`）：

| 阶段 | 旧单卡基线（README） | 本次实测 | 变化 |
| --- | ---: | ---: | ---: |
| 资产校验 | 2.6 s | 2.44 s | −6 % |
| 条件编码 | 20.8 s | 21.7 s | +4 % |
| transformer 加载 | 4.8 s | 12.7 s | +165 %（见下） |
| **采样（去噪）** | **181.3 s** | **138.8 s** | **−23.4 %** |
| 解码 + 编码 | 41.6 s | 42.0 s | +1 % |
| **总计** | **251.0 s** | **236.0 s** | **−6.0 %** |

- 768p 的采样收益（−23 %）小于 480p（−63 %）：768p 每步计算量约为 480p 的 2.5 倍，
  搬运本来就被摊得更薄，可隐藏的部分相对更少。
- 采样峰值 **12.56 GiB**（旧基线 11.33 GiB）：stream 路径多常驻一份 pinned 主机副本，
  设备侧多 1 个预取组（约 1.1 GiB）。仍远低于 61.27 GiB。
- **transformer 加载 12.7 s vs 4.8 s 是本次改动带来的真实代价。** stream 卸载要在
  加载阶段把 62 GiB 权重在主机侧 pin 一份（`_init_cpu_param_dict`，`low_cpu_mem_usage=False`），
  实测 pin 吞吐约 32 GiB/s，约 +2 s；其余差额来自同进程内 pinned 分配与 page cache 压力。
  这是**一次性**成本：换来的是每步省下 4.5 s 的同步 D2H。480p 下 6 步即回本 4 倍以上。
  （早期把差额归因于机器负载是错的：同配置三次运行加载分别是 10.5 / 13.4 / 12.6 s，
  都显著高于阻塞路径的 5.4 s。）
- 输出校验：1344×768 / 124 帧 / 24 fps / AAC 32 kHz 双声道 / 5.167 s，全部符合预期。

### 4.6 TP 路径的自适应权重放置

TP 分片后每卡只有 **32.5 GiB** 权重（51.6 %），所以"`--low-vram` 必须逐块卸载"对 TP 并不成立。
但也不能一律常驻 —— 取决于请求尺寸。

#### 判定必须同时看权重和激活

只用权重判断会翻车。实测 `1920x1088 / 10s`：权重 32.5 GiB 看着宽裕，采样到第 3 步时
`Tried to allocate 3.95 GiB (49.77 GiB already allocated, 1.92 GiB free)` 直接 OOM。
激活峰值约 **22 GiB**。

用两个实测点反推激活系数（`activations / (seq × hidden × 2B)`）：

| 实测点 | seq | 权重 | 激活 | 系数 |
| --- | ---: | ---: | ---: | ---: |
| 480p/5s 常驻成功 | 15 730 | 32.5 GiB | 2.29 GiB | 14.54 |
| 1920×1088/10s OOM | 148 580 | 32.5 GiB | ≥21.2 GiB | 14.26 |

两个完全不同的形状给出同一个系数 ≈14.4，说明模型可信。代码取 **15**（上界 + 余量），
并按 `min(free, total) × 0.90` 作为预算 —— `free` 已经把进程早期 reserve 的部分算掉了。

#### 判定表（TP2，61.27 GiB/卡）

| 画布 | 5 s | 10 s | 15 s |
| --- | --- | --- | --- |
| 832×480 | 常驻 | 常驻 | 常驻 |
| 1344×768 | 常驻 | 常驻 | 常驻 |
| 1920×1088 | 常驻 | 卸载（54.8 GiB 超出预算，实测 OOM） | 卸载 |
| 2560×1440 | 常驻 | 卸载 | 卸载 |

单卡（非 TP）权重 61.7 GiB，任何尺寸都走卸载，行为不变。

**回退路径已实测**：`1920x1088 / 10s` 判定为卸载（`transformer_offloaded: true`），
采样 811 s 正常完成、峰值 24.13 GiB、无 OOM；而在判定只看权重时，同一请求会在
第 3 步 OOM。这说明"同时看激活"这一条是必需的，不是保守过头。

#### 收益与一致性

480p / 5s / t2va：

| 指标 | TP + 流式卸载 | TP + 常驻 |
| --- | ---: | ---: |
| 采样 | 27.2–27.9 s | **22.0–23.1 s**（−18 %） |
| 采样峰值 | 5.04 GiB | 34.79 GiB |
| 输出 md5 | `fab2cc7f…` | `fab2cc7f…`（**逐位相同**） |

**为什么能逐位相同**：常驻与流式只改变权重放在哪里，不改变切分方式、集合通信顺序或
任何 kernel 的输入输出。已验证三种组合（流式 / 常驻 / 带 msprof）的 mp4 md5 完全一致。

#### TP 的剩余瓶颈不在搬运

rank0 的 msprof 时间线（480p/5s，**整轮** 84.7 s，其中采样窗口约 16 s）：

| 项 | 秒 | 说明 |
| --- | ---: | --- |
| 整轮计算区间并集 | 36.72 | 含条件编码、采样、解码；**不是采样单独的时间** |
| 采样窗口计算区间并集 | **14.97** | 62–78 s 窗口，busy_ratio **93.6 %** |
| 通信/同步区间并集 | 21.45 | 整轮口径，含与计算重叠的部分 |
| 其中与计算重叠 | **19.40**（90 %） | 只有约 2 s 露在关键路径上 |

> 早期版本把 36.72 s 标成了"采样"，实际它是整轮。采样阶段本身的计算并集是 14.97 s，
> 与 27.2 s 的 `sampling_seconds` 相比 busy_ratio 约 55 %——差额来自每步开头的
> 权重预取与 HCCL 等待，不是 kernel 空闲。

通信已被计算遮住九成，剩下的差距来自 HCCL 长尾。**注意这里有两个口径，别混用**：

| 口径 | 来源 | 601 次 `allReduce` | 含义 |
| --- | --- | ---: | --- |
| 通信域挂钟 | `communication_statistic.csv` 的 `Total Time` | 1.71 s（均 2.85 ms，max 41 ms） | 集合操作从发起到完成，含各 rank 等待与 notify |
| 单卡 kernel | `hccl.db` 的 `HCCLOP.begin/end` | **8.95 ms**（中位 14.6 µs） | 本卡实际在 HCCL kernel 里的时间 |

两者差约 190 倍，因为 all-reduce 的代价主要花在**等待最慢的 rank** 上，而单卡 kernel
时间只统计自己那段。判断"通信是否拖慢关键路径"要用挂钟口径，判断"单卡算力浪费"要用
kernel 口径。

挂钟口径下 601 次共 1.71 s，而采样总时长 27.2 s —— 通信本身不是主要瓶颈（6.3 %）。
真正的长尾是少数尖峰：`max 41 ms`，以及 `allGather` 里单次 **47.9 ms** 的异常值
（中位数只有 213 µs，即 225 倍）。这些尖峰集中在每步的固定位置，会直接拉长关键路径。

要再快，方向是**削尖峰**而不是减总量：把每层两次 all-reduce 合并成一次、给 AdaLN 的
`all_gather` 加 `async_op` 与下一层计算重叠、或换更小的通信域。继续优化权重搬运已经没有
空间了（§4 已到计算下限）。

### 4.7 下一个真正的瓶颈：AIV 向量算子

搬运优化到顶之后，单步里仍有 **13.1 s** 是 cube 利用率为 0 的纯向量算子（占所有 op
累计时长的 33.8 %）。它们串行插在 cube 段之间，是 cube 平均利用率只有 67.7 % 的直接原因。

> 口径说明：这里的 13.1 s 是 `op_summary` 里所有 `cube_utilization==0` 的 task 时长**直接
> 相加**（239 157 个 task），会重复计算同一 stream 上并行的时间，所以它大于墙钟。
> 它的用途是**横向比较各 op 的相对权重**，不能当成"省下这 13 s 就能快 13 s"。
> 真实可回收量见本节末尾的估计。

TP2 / 480p 的 op 级分解（`tp2_prof` 的 `op_summary`，整轮 239 157 个 task）：

| op | 次数 | 累计时长 | 占 AIV | 备注 |
| --- | ---: | ---: | ---: | --- |
| `Mul` | 46 639 | 2.413 s | 18.4 % | 主要是 AdaLN 的 `norm*(1+scale)+shift` |
| `Add` | 30 845 | 2.210 s | 16.8 % | 残差相加 |
| `Pows`+`ReduceMean`+`Rsqrt` | 40 966 | 1.518 s | 11.6 % | **就是 RMSNorm 拆开的 5 个算子** |
| `Cast` | 6 683 | 1.477 s | 11.2 % | bf16↔fp32 边界 |
| `Slice` | 29 248 | 1.447 s | 11.0 % | 大多来自 SwiGLU 的两半切分 |
| `Transpose` | 13 859 | 0.631 s | 4.8 % | QKV 布局 |
| 其余 | — | 3.444 s | 26.2 % | ConcatD/Swish/SliceV2/Muls/Neg… |

按形状聚合，最值钱的三个（`[14852, 5376]` 是 480p 的 packed 宽度）：

| 模式 | 次数 | 时间 |
| --- | ---: | ---: |
| `Add[14852,5376]` | 2 412 | 1.186 s |
| `Cast[14852,5376]` | 3 018 | 1.103 s |
| `Mul[14852,5376]` | 2 418 | 0.994 s |

**为什么不能简单融合**：直觉上 `x*(1+s)+g` 可以写成一次 `addcmul`，实测也快 27 %
（0.580 → 0.422 ms）。但两者**不是逐位等价**的——`1.0+s` 提前算改变了舍入位置，
真实形状上 max abs diff 达到 **27.6**（bf16 下 `1.0+s` 先舍入再乘，与先乘后加不同）。
残差那一路的 `addcmul(res, g, x)` 差异小得多（max 0.0625），但仍然不是 bitwise。

所以这条路和融合 RMSNorm 是同一类取舍：**要速度就得放弃逐位一致**。
项目当前的选择是保持逐位一致，因此本节只作为"下一个可选加速项"记录，不默认开启。

如果将来接受非逐位一致，按收益排序应该做：

1. `PIKA_FUSED_RMSNORM=1`（已有，−9.8 % 步长）——吃掉 `Pows`/`ReduceMean`/`Rsqrt` 1.52 s；
2. AdaLN 调制融合成 `addcmul` —— 吃掉 `Mul`+`Add` 里的一部分，估计 −1.0 s/轮；
3. 残差 `add` 与下一个 norm 融合（`npu_add_rms_norm`，实测 4.571 → 0.998 ms，4.6×）
   —— 但 max diff 0.03125，同样非逐位。

三项全做预计能把单步从 6.1 s 压到 ~5.0 s（−18 %），代价是输出不再可复现。

## 5. 复现

```bash
cd /workspace/Pika-H3-Flex
source .venv312/bin/activate
flock /tmp/opencode/perf/denoise/NPU.lock -c '
  ASCEND_RT_VISIBLE_DEVICES=0 PIKA_CPU_THREADS=16 \
  python /tmp/opencode/perf/denoise/tools/run_profile.py \
    --label repro --task t2va --canvas 832x480 --duration 5 --seed 42 \
    --offload-mode low-vram-stream --profile full --extra-arg=--low-vram \
    --pipeline-patch /tmp/opencode/perf/denoise/T3/noop_patch.py \
    --outdir /tmp/opencode/perf/denoise/repro --overwrite'
```

`result.json` 里 `summary` 即本页所有数字；`steps.csv` 是逐步明细；`power.csv` 是 1 Hz 功耗。

---

## 6. 风险与未覆盖

- NPU stream 路径依赖 diffusers 的私有 `_apply_group_offloading`（0.40.0.dev0 固定 commit）。
  升级 diffusers 需复核；仓库已用 `tools/apply_diffusers_patches.py --check` 固定版本。
- 已覆盖 `t2va` 的 480p（单卡 + TP2）与 768p（单卡），见 §4.5 / §4.6。
  **`ref2va`（多图 / 视频 / 音频参考）未做满载率测量**：它的条件编码与 `transformer_ref`
  fp32 路径不同，采样占比也不一样，本页数字不能直接外推。
- TP2 的收益主要来自**自适应常驻**（§4.6），不是流式卸载本身：480p 下常驻比流式再快 18 %。
  但常驻会让采样峰值从 5.04 GiB 涨到 34.79 GiB，大画布仍必须回退到流式。
- **首步仍偏慢**（8.20 s vs 6.07–6.12 s，多 2.1 s）：lazy prefetch 在第一次 forward
  才建立执行顺序，且首次全部 `non_blocking=False`（`LazyPrefetchGroupOffloadingHook`
  的 `initialize_hook` 会临时关掉异步）。逐步数据（`T2/runs/t2_stream/steps.csv`）：

  | step | wall | busy | onload | offload | h2d |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 0 | 8.199 | 5.847 | **1.909** | 5.789 | 1.777 |
  | 1 | 6.123 | 5.871 | 0.173 | 5.846 | 0.116 |
  | 2–5 | 6.07–6.10 | 5.87–5.88 | 0.11 | 5.85–5.89 | 0.06 |

  可回收量 ≈ **1.8 s / 次生成**（首步 onload 1.909 − 稳态 0.11）。做法是首次 forward 后
  hook 已自动移除并建好 `next_group` 链，所以只要让"第一次 forward"发生在计时之外即可；
  但那等于多跑一次完整前向（6 s），净亏。更省的做法是直接遍历 hook 注册表手工建链、
  跳过执行顺序追踪——**未实现**，因为收益（1.8 s）相对单次 480p 的 40 s 去噪只有 4.5 %，
  而它要动 diffusers 的 hook 注册表，与"不碰私有内部"的现状相冲突。
- 融合 RMSNorm 在 480p 上 −9.8 %，但会破坏逐位一致；默认关闭。
- 功耗随机器状态浮动：同一配置三次运行去噪均值 456–458 W、峰值 494–545 W，
  稳态步长 ±0.2 s。表中数字取单次 msprof 全量运行，"稳态步长"取步 1–5 中位数以避开首步。
