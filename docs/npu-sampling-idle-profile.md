# 采样阶段 NPU「偷懒」行为：msprof 实测

**结论：稳态 5 步里 NPU 的 kernel 忙碌率是 92.8–96.4 %（两次采集），看起来已经满载，
但设备时间线上确实存在三处「NPU 在等、不在算」的真空**，合计 **167–400 ms/步**，
另加首步一次性的 ~2.4 s。全部能对上具体模块，不是玄学。

| 现象 | 位置 | 代价（稳态每步） | 可修 |
| --- | --- | ---: | --- |
| **10 个非 block 模块（1.69 GiB fp32）用同步 H2D 上载** | 步首 +1.3 ms 处 | **45–244 ms 纯空转** | ✅ |
| 同组的**同步 D2H 卸载**落在步末 | 步尾 | 85–204 ms（跨步边界，部分不算空转） | ✅ |
| `position_ids` 是 **float64**，rope 每前向 `.to(float32)` | 前向内部 | **13–24 ms**（AIV 队列里的 f64 cast） | ✅ 零风险 |
| 首个 block 的 pinned 副本首次 H2D 未完全隐藏 | 步首 | **~15–20 ms** | 部分（预热） |
| 其余稀疏空档（<1 ms × 数百） | 全程 | ~30 ms | 收益太小，不做 |
| **首步**：执行顺序懒建立 + 首次同步上载 | step 0 | **2.4 s**（一次性） | 见 denoise 文档 §6 |

两次独立采集（同配置同 seed）：

| | T5 第一次 | T5 第二次（含功耗） |
| --- | ---: | ---: |
| 稳态 kernel 忙碌率 | **96.4 %** | **92.8 %** |
| NPU 完全无任务 | 0.83 s / 5 步 | 2.05 s / 5 步 |
| 该组同步 onload | 45–66 ms/步 | **169–244 ms/步** |
| 稳态步长 | 6.07–6.13 s | 6.29–6.40 s |
| 去噪总时长 | 38.68 s | 39.97 s |
| 去噪均值功率 / 峰值 | —（采样器未取到） | **449 W / 534 W** |
| 每步能量 / 每帧能量 | — | **2994 J / 145 J** |

差异全部来自那一次同步 H2D 的耗时波动（45 → 244 ms），说明它是**阻塞式、对系统状态
敏感**的路径——这正是它该被修掉的理由。

数据/脚本：`/tmp/opencode/perf/denoise/T5/runs/{t5_full,t5_full2}`、
`/tmp/opencode/perf/denoise/T5/{report_metrics,stall_sites,per_stream_first,gap_timeline,verify_fixes}.py`。

---

## 1. 测量口径

单卡（`ASCEND_RT_VISIBLE_DEVICES=0`），**832×480 / 5 s / 124 帧 / seed 42 / t2va**，
默认 `--low-vram`（与仓库默认一致，走 side-stream 预取卸载）。

1. **msprof**：`--task-time=l1 --ai-core=on --aic-mode=task-based --ascendcl=on
   --runtime-api=on --aic-metrics=PipeUtilization`。117 006 条 device task / 103 275 条 op，
   4 条 stream：**44 = 权重预取 H2D，45 = 事件等待，46 = 计算**，49 = 杂项。
2. **probe**：`H3DiffusionWrapper.forward` 前后 `torch.npu.synchronize()` 取真实步边界；
   给 `ModuleGroup.onload_/offload_` 打点，记录 `stream` / `async` / 字节数 / 模块类名。
3. **npu-smi 1 Hz**：功耗。第一次采集用的 harness 把卡号写死成 5，本机是 4，采到空；
   已修成自动探测（见 §5），第二次拿到了完整功耗。

术语：

- **kernel 忙碌**= `AI_CORE ∪ AI_VECTOR_CORE ∪ MIX_AIC ∪ MIX_AIV ∪ AI_CPU` 的区间并集；
- **NPU 完全无任务**= 步长 − 上述并集 − 当时正在跑的 H2D（H2D 与计算并行时 NPU 没在算，
  但那是设计上的搬运窗口，不算「空等」）。这两个数分开报。

---

## 2. 三处真实的空转

### 2.1 主角：10 个非 block 模块，每步 1.69 GiB **同步**上载

`group_events` 里每步有 **50 次** `stream=True / async=True` 的 block 预取（50 层，正常），
外加**恰好 2 次阻塞事件**：

```
onload  45.12 ms  1694.2 MiB  n_mod=10  stream=False async=False
        cls = Linear, MiniMaxH3AdaLayerNormOut, MiniMaxH3RotaryPos…（AdaLN 投影、norm_out、rope…）
offload 85.18 ms  1694.2 MiB  n_mod=10  同上        ← 步末
```

1.69 GiB 正好等于「1 个 AdaLN 投影 (`time_embed_dim→6·hidden·3`, 1.32 GiB fp32)
+ `norm_out.linear` (147 MiB) + `context_embedder` (140 MiB) + `time_embedder` 等」
之和（`adaln_check.py` 复算）。它与 block 组**分开**，因为
`pipeline.py` 只给 `block_level` 的组传了 side stream：

```python
enable_group_offload(..., num_blocks_per_group=1, use_stream=True, ...)
```

根因在 diffusers 自己的实现里，不在本仓库：`_apply_group_offloading_block_level`
会把所有未匹配的模块收进一个 `unmatched_group`，并且**硬编码**同步语义
（`.venv312/.../diffusers/hooks/group_offloading.py:793-808`）：

```python
unmatched_group = ModuleGroup(
    modules=unmatched_modules,
    ...
    non_blocking=False,     # <-- 同步
    stream=None,            # <-- 没有 side stream
    ...
)
```

`_apply_group_offload_direct` 传下去的 `stream` 只对 block 组生效，这个
`unmatched_group` 永远拿不到。于是这 10 个模块每步**先同步 D2H 回 CPU，下一步前再
同步 H2D 回设备**。

设备时间线上的证据（`gap_timeline.py` / `stall_sites.py`）：上载窗口内 stream 46
（计算）完全无任务，只有 stream 44 在搬 block 权重：

```
step1  Transpose -> Cast 之间空档  49.59 ms     (T5-1)
step2                              46.76 ms
step3                              65.79 ms
step4                              50.27 ms
step5                              49.00 ms
step1-3 (T5-2)                    169–244 ms
```

它落在**步首 +1.3 ms** 处，而 stream 46 上真正的计算要等到 +60 ms 才开始——这就是
「NPU 偷懒」的主体。

**为什么能修**：

| 变体（`verify_fixes.py`，同一计算队列） | 耗时 |
| --- | ---: |
| 只有计算 | 50.73 ms |
| 只有同步 H2D（1.654 GiB pinned） | 30.02 ms |
| 串行「H2D + 计算」 | **80.87 ms** |
| 计算 + side stream 上的 H2D | **51.98 ms** |

1.65 GiB 的搬运可以**几乎完全藏进计算**（回收 29 ms/次），量级与 profile 的空档一致。
两条落地路径（都不需要改 diffusers）：

1. **把 `block_modules` 扩展到这些模块**（`GroupOffloadingConfig.block_modules` 支持显式
   列出），让它们进 block 组、拿到 side stream；或
2. **直接常驻**：这 10 个模块 1.69 GiB，480p 采样峰值才 12.6 GiB，61 GiB 的卡绰绰有余
   ——事后 `.to(device)` 它们、并把它们从 `_gather_parameters_with_no_group_offloading_parent`
   之外单独管理即可。

两者都能消掉 45–244 ms/步的步首同步上载，顺带消掉步末的同步 D2H。

### 2.2 `position_ids` 是 float64，rope 每前向转一次

msprof op 表里每步**有且仅有 1 个** `AI_CPU` 类型的 op：

```
Cast  AI_CPU  in="14852,3"  out="14852,3"  0.269 ms
```

`14852` = 480p 的 packed 序列长度，`3` = `(t,h,w)`——正是
`pika_h3/core/packing.py::build_layout` 产出的 `position_ids`：

```
position_ids  torch.Size([14864, 3])  torch.float64
```

transformer 的 rope 第一行就是 `position_ids.to(torch.float32)`
（`transformer_minimax_h3.py:99`）。float64 在 Ascend 上没有向量指令，只能走 `AI_CPU`
标量路径；它本身 0.27 ms，但排在 AIV 队列末尾，设备必须等它完成，于是每步开头出现
13–24 ms 的 AIV 空档（`stall_sites.py`：`Add → Swish` 每步 1 次，13.2–24.9 ms）。

**修法**：`build_layout` 直接产出 float32。`position_ids` 是 `[-1,1)` 的归一化坐标，
float32 的 24 位有效位足够，且 rope 本来就会立刻转成 float32，所以**数值完全不变**。

微基准（`verify_fixes.py`，同一 AIV 队列）：float64+cast **100.78 ms** vs float32
**76.86 ms** → 每前向 **24 ms**。

### 2.3 首个 block 的 pinned 副本首次上载

step 2 的第一个 block onload 窗口是 `50.81→70.67 ms`（19.87 ms），而同一组其余 249 次
只需要 0.75 ms/次。这 20 ms 里设备只跑了少量 AIV op。原因是 pinned host 副本**懒建立**
（`low_cpu_mem_usage=False` 只保证 CPU 侧已 pin，首次 `copy_` 仍要建 DMA 描述符）。
修法：warmup 里对 group 0 做一次 `copy_(..., non_blocking=True)` 建链，可回收 ~15 ms/步。

---

## 3. 不是「偷懒」的部分（已满载）

| 项 | 数值 | 说明 |
| --- | --- | --- |
| 稳态 kernel 忙碌率 | **96.4 % / 92.8 %**（两次） | 100 ms 分箱 ≥90 % 占 78–80 %，≥50 % 占 91–96 % |
| npu-smi `AI Core` | 均值 **94.8 %**，≥90 % 占 **95 %**（38/40 样本） | 1 Hz 口径，仅作交叉验证 |
| `AI Cube` 均值 | **60.8 %** | cube 有空转是 AIV 段串行插入所致，见 §4 |
| 权重预取 H2D | 1.23 s/步，**1.18 s（96 %）与计算重叠** | stream 44 与 46 真并行 |
| 同步 D2H（block 组） | **0** | stream 路径只重指 `param.data` |
| cube 活跃 op 时间 | 4.045 s / 步（68.8 %） | MatMul cube 99 %、FlashAttention cube 98 % |
| AIV 向量 op 时间 | 1.833 s / 步（31.2 %） | Mul/Add/Cast/Slice/Transpose/Pows… 这是下一层瓶颈 |
| `AI_CPU` op | 5 步共 5 个（每步 1 个） | 只有 position_ids 的 cast |

stream 45 每步 3.66 s 的 `EVENT_WAIT` 是 HCCL/事件同步的等待载体，与计算并行，不构成
额外空转；`event_wait` 本身也不占 AI core。

---

## 4. 下一步的优先级

1. **把这 10 个非 block 模块纳入 side-stream 预取或直接常驻**——唯一能稳定拿到
   **0.7 % 以上**（最坏情况 4 %）的点，且顺带消掉步末的 85–204 ms 同步 D2H。
2. **`position_ids` float64 → float32**——零风险，13–24 ms/步。
3. **预热首个 block 的 pinned 副本**——~15 ms/步。
4. 首步 2.4 s 的 lazy prefetch 建链：收益 4.5 %，但要把执行顺序追踪挪出计时窗口，
   优先级最低。

> 注：msprof 自身有开销（`StreamSynchronize` 6639 次共 36.8 s、`MemCopySync` 5984 次
> 共 17.9 s，含插桩），所以带 profiler 的步长比干净的 6.07 s 略高；口径内的**相对**
> 占比不受影响。

---

## 5. 顺手修掉的 harness bug

`tools/run_profile.py` 把 `npu-smi info watch -i` 的卡号写死成 `5`（原机器的编号），
本机是 `4`，于是功耗采样全程 `Invalid card id`，`power.csv` 只有表头。已改为自动探测：

```python
def card_id() -> int:
    """$NPU_SMI_CARD, else the lowest valid card id from `npu-smi info`."""
```

（按「有 Health 列的行才是 card 行」识别，避免把 chip 序号 0/1 当成卡号。）
修复后第二次采集拿到了完整的 Pwr / AI Core% / NPU Util% / AI Cube% 序列。

---

## 6. 端到端加速比（VAE 优化前后，768P / 5s）

配置：`1344x768 / 124 帧 / seed 42 / t2va / --low-vram`，同一 prompt，无 profiler。
基线 = `git archive HEAD src`（VAE 优化前），新版 = 当前工作区，各跑 2 / 5 次。

| 阶段 | 基线均值 | 新版均值 | 加速比 | 时间变化 |
| --- | ---: | ---: | ---: | ---: |
| 资产校验 | 2.72 | 2.77 | 0.98× | +1.8 % |
| 条件编码 | 16.68 | 14.11 | 1.18× | −15.4 % |
| transformer 加载 | 11.11 | 11.25 | 0.99× | +1.3 % |
| **采样** | **138.90** | **138.38** | **1.00×** | **−0.4 %** |
| **解码 + 编码** | **43.41** | **22.60** | **1.92×** | **−47.9 %** |
| **端到端合计** | **212.82** | **189.33** | **1.12×** | **−11.0 %** |

- 采样阶段与 VAE 无关，1.00× 是预期的（也是这次对比的"无副作用"证据）。
- 解码阶段单独看是 **1.92×**；`decode_peak` 13.81 → 16.37 GiB（fp16 autocast 的
  workspace 更大），仍远低于 61 GiB。
- 端到端只有 1.12×，因为采样占了总时长的 65 %——VAE 优化动的是剩下那条腿。
- 条件编码 1.18× 不是本次改动的功劳，是两次基线之间机器负载的抖动（基线两次
  18.58 / 14.79 自己就差 26 %）。

**逐位一致性**（同一份 latent，只换解码器，`compare_decoders.py`）：

| 对比 | max diff | 平均 | PSNR | 逐像素相同 |
| --- | ---: | ---: | ---: | --- |
| 旧 fp32 vs 新 fp32（`--vae-decode-dtype fp32`） | **0** | 0 | ∞ | ✅ **是** |
| 旧 fp32 vs 新 fp16（默认） | **7** | 0.031 | 63.3 dB | 否 |

注意端到端 mp4 的差异（max 68、mean 1.29、PSNR 42 dB）**不是解码器直接造成的**：
把两版解码器产出的 uint8 帧各自过同一个 x264（同一份 `write_mp4`，已验证该 box 上
x264 是确定性的），编码后差异就从 7 放大到 68 —— 这是有损编码器对输入微扰的级联放大。
帧级统计：99 % 的像素差 ≤ 7，0.83 % 的像素差 > 8，> 16 的只占 0.02 %。
