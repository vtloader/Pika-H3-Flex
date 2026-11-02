# Pika H3 Flex 浏览器 WebUI

纯 Python 标准库（`http.server` + `email.parser` + `subprocess` + `threading`）实现的本地
WebUI，用于在 Ascend NPU 上驱动 `pika-h3-pro` 生成带音频的视频。无需安装 FastAPI /
Flask / Gradio，也没有 Node 构建步骤。

## 启动

```bash
bash scripts/launch_webui.sh
# 打开 http://127.0.0.1:3000
```

脚本优先使用仓库内的 `.venv312/bin/python`（可用 `PYTHON_BIN` 覆盖），并设置
`PYTHONPATH=src`、`PIKA_CPU_THREADS=16`。也可以直接启动：

```bash
.venv312/bin/python -m webui.server
HOST=0.0.0.0 PORT=8080 .venv312/bin/python -m webui.server
```

| 环境变量 | 作用 | 默认 |
| --- | --- | --- |
| `HOST` / `PORT` | 监听地址 | `127.0.0.1` / `3000` |
| `PIKA_WEBUI_DRY_RUN=1` | 不加载模型，模拟进度并写入占位 MP4/JSON | 关闭 |
| `PIKA_WEBUI_RUNS_DIR` | 运行目录（日志、上传、输出） | `webui/runs` |
| `PIKA_WEBUI_BUDGET` | 默认计算预算（MP·帧） | `260` |
| `PIKA_WEBUI_NO_DEVICE=1` | 跳过 `npu-smi` 探测 | 关闭 |
| `PIKA_WEBUI_DRY_RUN_DELAY` | 试运行每步之间的等待秒数 | `0.35` |
| `PIKA_BASE_PATH` / `PIKA_ADAPTER_PATH` | 表单默认权重目录 | `models/MiniMax-H3-ref` / `models/Pika-H3-Flex` |

## 四种生成模式

页面用四张卡片呈现，服务端再把关键帧卡片映射到正确的 CLI 任务：

| 卡片 | `data-mode` | 上传 | CLI `--task` |
| --- | --- | --- | --- |
| 文本 | `text` | 无 | `t2va` |
| 关键帧 | `keyframes` | 首帧 / 尾帧 / 两者 | 首帧 → `i2va`，尾帧 → `l2va`，两者 → `fl2va` |
| 首尾帧 | `firstlast` | 首帧 + 尾帧 | `fl2va` |
| 多模态参考 | `references` | 图片 / 视频 / 音频 | `ref2va` |

API 也接受 `first+last` 作为 `firstlast` 的别名，以及原始任务名
`t2va` / `i2va` / `l2va` / `fl2va` / `ref2va`（映射到对应卡片）。

首帧、尾帧都是单值控件，带**替换**与**删除**；参考素材按模态分组、保持上传顺序，
对应提示词里的 `<Picture n>`、`<Video n>`、`<Audio n>`，每项都有**替换 / 删除**，
并提供**清除图片 / 清除视频 / 清除音频**。

### 参考素材限制（服务端与前端同时校验）

- 参考图片 ≤ **9**，参考视频 ≤ **3**，独立参考音频 ≤ **3**，合计 ≤ **12**；
- 至少 **1 张图片或 1 个视频**（音频不能单独使用）；
- 每个视频 / 独立音频 **2–15 秒**；
- 参考视频总时长 ≤ **15 秒**，独立参考音频总时长 ≤ **15 秒**；
- 视频内嵌音轨随视频一起作为参考，不额外计入独立音频。

上传大小：图片 ≤ 20 MB，视频 ≤ 256 MB，音频 ≤ 64 MB，单次请求 ≤ 512 MB。
服务端用 `ffprobe`（缺失时回退 PyAV）读取真实时长，不信任浏览器端的时长。

## MP 模式的画布与时长计算

模型只接受 32 px 网格的画布和 `17n+5` 帧的时间网格（24 fps）。MP 模式让用户只输入
**目标百万像素数**和**宽高比**，其余由服务端推导：

1. 由 `MP × 1e6` 和宽高比算出目标**短边**；
2. 按短边就近选择 `buckets.yaml` 里的画布档位。档位名义短边是
   **480 / 768 / 1088 / 1440 / 2176**，同一档位下 11 种比例都有对应画布：
   2.0 MP / 16:9 → 短边 1061 → **p1080 档 1920×1088**；
   2.0 MP / 1:1 → 短边 1414 → **p1440 档 1440×1440**。
   超宽比例（21:9、9:21）的档位行本身短边更小，改用该行实际短边比较，
   避免 2.0 MP 的 21:9 掉回 768p；
3. 宽高比被夹在 `[0.25, 4.0]`（即 1:4–4:1），画布始终是档位表里的合法行；
4. 时长自动选择：从 15 s 往下，取第一个满足
   `width × height × frames ≤ 预算（MP·帧）` 的整数秒；预算默认 260 MP·帧
   （约等于 768p × 10 s）。**若 5 s 都放不下，默认直接报错**，不会悄悄替换画布；
   只有勾选「预算不足时自动缩小画布」（`allow_canvas_shrink`）才会等比缩小到
   能放下 5 s 的尺寸；
5. 返回并展示最终的 `W×H`、所属档位、`17n+5` 帧数、实际秒数（帧数 / 24）与预算占用。

**为什么是档位而不是精确 MP**：档位表是模型验证过的画布族，也与 ComfyUI 的
H3 分辨率选择一致。按面积精确搜索会得到 1888×1056 这类陌生尺寸；按档位对齐后
2.0 MP 得到标准的 1920×1088。代价是实际面积与输入 MP 可能有偏差（0.4 MP / 3:2
会落到 704×480，−15.5%），因此界面会显示「已对齐到 p1080 档位 1920×1088
（2.089 MP，与输入 2 MP 相差 +4.4%）」。

常见 16:9 结果：

| 输入 MP | 档位 | 画布 | 实际 MP |
| --- | --- | --- | --- |
| ≤ 0.6 | p480 | 832×480 | 0.399 |
| 0.7 – 1.5 | p768 | 1344×768 | 1.032 |
| 1.6 – 2.8 | p1080 | **1920×1088** | 2.089 |
| 2.9 – 5.8 | p1440 | 2560×1440 | 3.686 |
| ≥ 5.9 | p2160 | 3840×2176 | 8.356 |

例（预算 260）：1.0 MP / 16:9 → **1344×768 / 243 帧 / 10.125 s**。
2.0 MP / 16:9 → **1920×1088 / 124 帧 / 5.167 s**（面积更大，自动时长更短）。
0.4 MP / 16:9 → **832×480 / 362 帧 / 15.083 s**。

**预设路径**保留给不想算 MP 的用户：选择 480p / 768p + 宽高比 + 整数秒时长，画布直接取
`src/pika_h3/core/buckets.yaml` 的桶，行为与 CLI 默认路径完全一致。

**手动覆盖**复选框允许强制指定画布（仍校验 32 px 网格与 1:4–4:1）和/或时长；
超出预算只提示不阻止，便于用户自行承担风险。

服务端在 `/api/plan` 与提交时使用同一套 `webui/geometry.py`，并把最终画布通过
`--canvas WIDTHxHEIGHT` 传给 CLI；`pika_h3.pipeline.resolve_canvas_args()` 在显式画布
存在时跳过 bucket 表的 tier 查找，直接使用给定宽高。

## 任务、进度与日志

提交后立即返回任务 id，请求进入 **FIFO 队列**；同一时间只运行 **1 个**生成进程。
排队中的任务显示队列位置，可取消；运行中的任务取消时会终止整个进程组
（`start_new_session=True` + `killpg`）。

子进程命令形如：

```bash
.venv312/bin/python -u -m pika_h3.cli --task t2va --prompt-file <run>/prompt.txt \
  --canvas 1344x736 --duration 10 --seed <seed> --output <run>/output.mp4 \
  --base-path ... --adapter-path ... --low-vram
```

勾选“双卡张量并行”时改用
`python -m torch.distributed.run --standalone --nproc_per_node=2 -m pika_h3.cli ... --tensor-parallel 2`。

“VAE 解码精度”下拉框把 `--vae-decode-dtype` 传给 CLI：默认 `fp16`（约 2× 快，
与 fp32 逐像素差 ≤ 2/255），`fp32` 与旧版逐位相同。该选择随任务记录写入
`state.json` 与输出 JSON 的 `vae_decode_dtype` 字段。

`pika_h3/pipeline.py` 会输出机器可读标记：

```
[progress] stage=loading pct=5 what=references
[progress] stage=conditioning pct=12 what=text-encoder
[progress] stage=sampling pct=71 step=3 total=6
[progress] stage=decode pct=88 what=video
[progress] stage=encode pct=95 what=mp4
[progress] stage=done pct=100 output=...
```

所有标记都带 `flush=True`，原有 `[cond] Qwen3-VL loaded …`、`[cond] VAEs loaded …`、
`Saved <path>` 等打印保持不变；若标记缺失，服务端按阶段名回退进度。日志逐行写入
`<run>/task.log` 并通过 `GET /api/tasks/<id>/log` 轮询展示（页面每 1.2 s 刷新）。

成功后页面显示内嵌播放器、MP4 下载和 JSON 元数据下载；元数据里记录本次实际使用的
种子、画布、帧数与预算。

## 种子控制

- 默认 **随机**：输入框显示一个 0 … 2^63−1 的整数（页面加载时抽签），POST 返回后
  立即换新值并重新渲染，不会在页面停留期间重复使用同一个自动种子；
- **+1 / −1** 步进（−1 下限为 0，上限 2^63−1）；**固定 / 随机**切换；“换一个”立即重新抽签；
- 手动输入数字或点击 +1/−1 会自动切换到 **固定**，提示文案同步变化，输入框边框变为琥珀色；
- 提交时发送的就是输入框里显示的整数；服务端校验 `0 <= seed < 2^63`（非法值直接 400），
  未提供时用 `os.urandom` 自行抽取，并把它写进任务记录与输出 JSON（“本次种子: N”）；
- 随机/固定模式与固定值写入 `localStorage`，刷新页面后保留；随机模式在加载时重新抽签。

## 测试

```bash
.venv312/bin/python -m unittest webui.tests.test_webui -v   # 42 项：MP 数学、参考限制、--canvas、种子、进度标记、VAE 解码
node webui/tests/seed_flow_test.js                          # 随机/固定、提交后刷新、持久化
bash webui/tests/e2e_dry_run.sh 3112                        # 自带 dry-run 服务器的端到端检查
```

`e2e_dry_run.sh` 会启动一个 `PIKA_WEBUI_DRY_RUN=1` 的服务，提交任务并轮询进度/日志，
校验 MP4 与 JSON 可下载，验证随机种子的两次提交不同、固定种子两次提交相同，以及
10 图 / 4 视频 / 13 个参考 / 1 秒视频 / 纯音频无视觉素材这几种非法请求都被拒绝。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | 主页面 |
| `GET` | `/logs/<id>` | 单任务日志页 |
| `GET` | `/static/<file>` | 静态资源（白名单） |
| `GET` | `/api/config` | 宽高比、桶表、限制、默认权重路径 |
| `GET` | `/api/status` | NPU HBM 占用 + 队列快照 |
| `GET` | `/api/tasks` | 任务列表（含进度、种子、画布） |
| `GET` | `/api/tasks/<id>` | 单任务状态 |
| `GET` | `/api/tasks/<id>/log?tail=262144` | 纯文本日志（默认尾部 256 KiB） |
| `GET` | `/api/tasks/<id>/video` | 输出 MP4（支持单段 Range） |
| `GET` | `/api/tasks/<id>/metadata` | 输出 JSON（附件下载） |
| `POST` | `/api/plan` | 用同样的表单字段预计算画布（JSON） |
| `POST` | `/api/tasks` | 提交任务（multipart 或 JSON，返回 `202 {id, seed, plan}`） |
| `POST` | `/api/tasks/<id>/cancel` | 取消排队/运行中的任务 |

## 目录与安全

- 每次提交创建唯一目录 `webui/runs/<YYYYmmdd-HHMMSS-xxxxxxxx>/`，包含
  `prompt.txt`、`inputs/`、`task.log`、`state.json`、`output.mp4`、`output.json`；
- 文件名只保留 basename 并过滤控制字符/路径符号，客户端路径永不被信任；
- 静态文件来自固定白名单，运行文件只通过任务 id 正则 + `resolve().is_relative_to()`
  校验后提供，不存在 `..` 穿越；
- 服务默认只绑定 `127.0.0.1`。

## 试运行

```bash
PIKA_WEBUI_DRY_RUN=1 .venv312/bin/python -m webui.server
```

试运行不会启动模型子进程：它按固定节奏写入 `[progress]` 标记与日志，并用 `ffmpeg`
生成 1 秒占位 MP4（无 ffmpeg 时写入最小 MP4 头）以及 JSON。用于验证 API、队列、
取消、种子与页面交互。
