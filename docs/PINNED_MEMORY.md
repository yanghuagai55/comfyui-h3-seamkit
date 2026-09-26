# PINNED_MEMORY.md — aimdo 上限与多档补丁（A 主机 pinned · B 设备 cast 预留）

> 工具：`tools/pinned_memory_patch.py` · 备份：`_hardcut_work/patches/model_management.py.orig`
> 本文是 README 里引用的那份说明（2026-09-25 补写，同日扩为 A+B 双数值管理）。

**本工具管两个独立数值，都可持续切档、随时复原：**

| | A 主机 pinned 上限 | B 设备侧 cast 预留 |
|---|---|---|
| 位置 | `model_management.py:1593` | `model_management.py:1378` |
| 原版 | `MAX_PINNED_MEMORY = ram * 0.40` | `DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE = 16 * 1024 ** 3` |
| 档位 | 0.45 标准 · **0.50 缓存一采** | **4 GiB 保守** · 2 GiB 极小 |
| 修的报错 | `hostbuf.c:46 hostbuf_grow: ... beyond reserved` | `host_buffer.py:109 read_file_slice failed` |
| 影响的域 | **主机**内存 | **设备**显存 |

## 1. A：上限是怎么算的

```python
# comfy/model_management.py
MAX_PINNED_MEMORY = ram * 0.40            # ← Windows 分支，原版
pinned_hostbuf_size = min(size, MAX) * 2  # ← 注意这个 x2
# comfy/model_patcher.py
HostBuffer(0, 8 * 1024 * 1024, pinned_hostbuf_size(model.model_size()))
```

在 `model_management.py:1589-1604` 附近。**有效上限 ≈ 2 × ratio × 物理内存**。

撞上限时 aimdo 这样报（**这是真实的上限拒绝，不是假报错**）：

```
aimdo: src/hostbuf.c:46:ERROR:hostbuf_grow: requested 30635134976 bytes beyond reserved host buffer 30631395328
```

净差常常只有 **3.5~235 MiB**，所以表现极具迷惑性：上限在**建模型那一刻定死**，
而请求大小随内容浮动 → **前面几段能跑完，后面某一段突然撞线**，与段长无关。

> ⚠️ 别和 `aimdo memory compile error` 搞混：那个是**假报错**（真身是 `result=2` =
> `cudaErrorMemoryAllocation`，真 CUDA OOM），重启即可恢复；而 `hostbuf_grow` 是
> **真实上限拒绝，重启不会自愈**，必须改参数。

## 2. 档位表（本机物理内存 31.63 GiB）

| 档位 | 上限 | 定位 |
|---|---|---|
| `ram*0.40` | 25.30 GiB | ComfyUI 原版；**复原档** |
| `ram*0.45` | 28.47 GiB | 标准档，日常跑片 |
| `ram*0.50` | 31.63 GiB | **缓存一采档**，一采缓存 / HIT 链路专用 |

## 3. 缓存一采档为什么存在

`MiniMaxH3FirstPassSampler` 走 **一采缓存 HIT** 时，一采采样被整体跳过，
但**二采仍要加载 32.4 GB 主模型、并第一次建立 host buffer**。

2026-09-25 14:38 本机实测（HIT 轮二采）：

```
requested 30635134976 bytes beyond reserved host buffer 30631395328   ← 只差 3.57 MiB
requested 30741573632 bytes beyond reserved host buffer 30631395328   ← 差 105 MiB
```

- 需求峰值 = `30741573632 B` = **28.63 GiB**
- 0.45 档 reserve = `30631395328 B` = **28.53 GiB** → 差 105 MiB 被拒
- 0.50 档 reserve ≈ **31.63 GiB** → 余量 ~3.0 GiB

**代价**：reserve 只是"预留地址空间 + 按需 extend"（初始 8 MB、prewarm 64 MB），
所以**抬 cap 的实际代价 ≈ 超出量** —— 本场景只会比 0.45 多占 ~105 MiB，
不是一次性吃满 31.63 GiB。**但** reserve ≈ 1.00 × 物理内存，万一某轮真吃满会挤压系统，
所以这档定位为专用档，**日常跑片建议切回 `0.45`**。

### 为什么不直接用 `--high-ram`

`--high-ram` 把上限**整个去掉**（`pinned_hostbuf_size` 直接 `return size * 2`，
32.4 GB 模型 → **64.8 GiB**），还会连带 `cache_classic=True`。
pinned 内存不可换页，护栏拆掉后失败模式会从"干净报错"变成"掉页 / 整机卡死"。
**护栏留着** —— 0.50 档只是把护栏抬到物理内存量级，不是拆掉。

## 4. 用法

### 4.1 ★ 配置预设（推荐，一键切完整组合）

工具现在能**同时**改 `model_management.py` 的两个常量 **和** `启动ComfyUI.bat` 的附加参数：

| key | 名称 | A 主机 pinned | B 设备 cast | bat 附加参数 |
|---|---|---|---|---|
| `official` | 官方默认 | `ram*0.40` | 16 GiB | （无）→ bat 801 字节，出厂原样 |
| `std045` | 0.45 默认 | `ram*0.45` | 16 GiB | （无） |
| **`stable`** | **新设置（稳定跑通）★** | `ram*0.45` | 16 GiB | `--disable-async-offload` |

```bash
... pinned_memory_patch.py --preset stable -y    # 切到实测跑通的那套
... pinned_memory_patch.py --preset official -y  # 完全还原出厂
```

菜单里对应 1) / 2) / 3)。

**★ `stable` 是 2026-09-25 实测跑通的配置**：`--disable-async-offload` 让
`get_offload_stream()` 返回 `None`，cast buffer 从 aimdo 的 `VRAMBuffer`（常驻不还）
改走 `torch.empty`（PyTorch 分配器，用完可回收）—— 二采 4/4 段全过、25分33秒出片，
且**每段耗时与开异步时基本持平**。

**⚠️ 预设会改两个文件，都必须重启 ComfyUI 才生效。**

**bat 改写的安全设计**：只替换 `set COMMANDLINE_ARGS=` 行里 `BAT_BASE` 之后的**附加段**，
基础部分不是预期内容就拒绝；全程 **GBK 字节级**（bat 含 `D:\共享`），
并做三重校验：GBK 可解码 / CRLF 行数不变 / `共享` 的中文字节数不减。
原始 bat 备份在 `_hardcut_work/patches/启动ComfyUI.bat.orig`。

### 4.2 单项操作

```bash
# 交互菜单（推荐）
"D:/comfyui/comfyenv/python.exe" \
  "D:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/tools/pinned_memory_patch.py"

# 非交互
... pinned_memory_patch.py --status                 # 只看 A+B 状态
... pinned_memory_patch.py --to 0.50 -y             # A 切到缓存一采档
... pinned_memory_patch.py --to 0.45 -y             # A 切回标准档
... pinned_memory_patch.py --cast-to 4g -y          # B 切到保守档（4 GiB）
... pinned_memory_patch.py --cast-to 2 -y           # B 切到极小档（2 GiB）
... pinned_memory_patch.py --restore-a -y           # 只复原 A
... pinned_memory_patch.py --restore-b -y           # 只复原 B
... pinned_memory_patch.py --restore -y             # A+B 整文件复原
```

菜单：

```
  ── A. 主机 pinned 上限 ──────────────────────────────────
    1) 标准档        ram*0.45   上限 ≈ 0.90x 物理内存   （日常跑片）
    2) 缓存一采档 ★  ram*0.50   上限 ≈ 1.00x 物理内存   （一采缓存 HIT 链路专用）
  ── B. 设备侧 cast 预留 ──────────────────────────────────
    3) 保守档        4 GiB      ← 8 GB 卡推荐
    4) 极小档        2 GiB      （保守档仍失败时用）
  ── 复原 ─────────────────────────────────────────────────
    5) 复原 A        6) 复原 B        7) 全部复原
    0) 退出
```

**生效方式：必须重启 ComfyUI。** A 看启动日志：

```
Enabled pinned memory 16193        # = 物理内存 x 0.50 取整到 MiB
                                   #   0.45 -> 14574 / 0.40 -> 12954
```

B 是模块级常量，同样要重启才重新求值；实际显存占用发生在首次权重转换时。

## 5. 安全设计（改这个文件前先看）

- 打补丁前完整备份；`.orig` **只在"当前确实是原版"时写入一次，绝不覆盖**
- 正则精确匹配目标行做**单次**替换，命中数 != 1 就拒绝
  → A 锚定在 `if WINDOWS:` 上，不会误改下一行 Linux 分支里那个**同样写着 `ram * 0.40`** 的表达式
- 状态机 `orig / profile / unknown`：**unknown 一律拒绝，不硬来**
  （例如 ComfyUI 升级换写法、或这行被人手改成 0.42 / 8 GiB）
- 切档时若缺 `.orig` 且当前**不是原版** → **拒绝**（保证任何状态都退得回原版）
- 复原走**逐字节拷贝**（`--restore`）或精确回写原版值（`--restore-a/-b`）
- 写盘后做"新文本 + 落盘"**双重校验**，失败自动从本次备份回滚
- 行尾安全：`Path.read_text/write_text` 在 Windows 上 `\r\n → \n → \r\n` 往返无损，
  已验证改后 `CRLF=2131 / 纯 LF=0`、字节数仅按位数增减
- **`--backup-dir` 一经指定即锁定**，不再"回退去找有 `.orig` 的目录"
  （2026-09-25 事故：修复前试跑会把 bak 写进真实 `patches/`）

## 6. B：设备侧预留（`read_file_slice failed` 的真凶）

`host_buffer.py:109` 那个报错的根因**不在 pinned**，而是：

```
comfy/model_management.py:1378  DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE = 16 * 1024 ** 3   ← 16 GiB
comfy/model_management.py:1416  get_aimdo_cast_buffer() -> VRAMBuffer(DEFAULT_..., device.index)
comfy_aimdo/vram_buffer.py:32   VRAMBuffer.__init__ → lib.vrambuf_create(device, 16 GiB)   ← 设备侧预留
comfy/ops.py:160-161            cast_buffer.get(buffer_size, offset) → aimdo_to_tensor(...)
comfy/ops.py:466                cast_bias_weight   ← 32B TE 前向第一个权重去量化时就崩
```

即 aimdo 为**权重去量化/类型转换**申请一块 **16 GiB 的设备显存 VRAMBuffer**；
本机只有 8187 MB → `vrambuf_grow` 拿不到 → 往那块 VRAM 拷权重时 `read_file_slice failed`
→ Sticky error → `CUDA error: out of memory`。

**崩点固定、确定性可复现**（2026-09-25 三次实测，栈逐行相同，与 HIT/缓存无关）：

```
nodes.py:340 → conditioning.py:543 → sd.py:410 → text_encoders/minimax.py:100
  → qwen3vl.py:100 → llama.py:906/892/685/651 → ops.py:466 cast_bias_weight
  → ops.py:377/234/232/223 → model_management.py:1525 cast_to_gathered
  → memory_management.py:70 → comfy_aimdo/host_buffer.py:109
```

**B 档位（本机显存 8187 MB）**

| 值 | 定位 | 说明 |
|---|---|---|
| `16 GiB` | 原版 | ComfyUI 默认；8 GB 卡上必然失败 |
| `4 GiB` | **保守档** | 8 GB 卡推荐，覆盖绝大多数单权重转换 |
| `2 GiB` | 极小档 | 只在保守档仍失败时用 |

**语义要点（实测确认，2026-09-25 15:05）**：
用 comfyenv 直连 aimdo C 库实测（显存全空的干净环境）：

| 操作 | 显存变化 | 说明 |
|---|---|---|
| `VRAMBuffer(4 GiB, 0)` | **+0 MiB** | `create` 只记预留上限，**不占显存** |
| `.get(256 MiB)` | **−256 MiB** | `grow` 真实占用 |
| 累计 `.get()` 到 1 GiB | **−1024 MiB** | 按 16 MB chunk 向上取整 |
| `.get(6 GiB)`（超 4 GiB 上限） | 抛 `RuntimeError: VRAM grow failed: 6442450944 bytes` | **N 就是真实需求值** |

所以 max_size 是**给 grow 设的天花板**：压小它不会立刻省显存，
但会让"某次转换请求超限"从**静默 CUDA OOM** 变成**明确的 `VRAM grow failed: N bytes`**。
**看到这个报错就是拿到了真实需求量**，照 N 往上取一档即可。

**未排除的次因**：`ops.py:190` 里 `cast_buffer_offset += buffer_size` 是**累加**的，
若一次前向里累计超过 max_size，即使单权重不大也会 grow 失败 ——
所以别一上来就试 2 GiB。

> ⚠️ 该机制原作者标注 *"this is temporary and will be removed in a future comfy.
> Not supported for custom node use."* —— 本工具只改**数值**，不动逻辑；
> ComfyUI 升级后状态会变 unknown，届时按 §9 处理。

## 7. ★ 更优先的官方开关：`--vram-headroom`

改源码之前**先试这个** —— 它是 ComfyUI 官方启动参数，零侵入、易复原。

```
--vram-headroom FLOAT   Set the amount of vram in GB for DynamicVRAM to maintain as extra
                        headroom above default. ComfyUI will try and keep this much VRAM
                        completely free and unused, even counting VRAM from other apps.
                        （comfy/cli_args.py:175，默认 0）
--reserve-vram FLOAT    Set the amount of vram in GB you want to reserve for use by
                        your OS/other software.（默认 None = 按 OS 自动）
```

**为什么关键**：默认 `--vram-headroom 0` 意味着 DynamicVRAM **尽量用满显存**，
连桌面/其他程序占的都不让 —— 8 GB 卡上贴边跑长序列时极易撞墙。

本机 2026-09-25 实测症状（与此相符）：
- 崩在 32B TE 前向，`Sticky error detected`
- PyTorch `Memory summary` 显示 **`CUDA OOMs: 0`**（分配器自己没 OOM！）
- 进程总占 6813 MiB，而 PyTorch reserved 只 2240 MiB → 差额来自 aimdo

**处置**（2026-09-25）：`启动ComfyUI.bat` 的 `COMMANDLINE_ARGS` 追加 `--vram-headroom 1`。
原始 bat 备份在 `_hardcut_work/patches/启动ComfyUI.bat.orig`（该目录只放本机备份）。

**改 bat 的注意**：`启动ComfyUI.bat` 是 **GBK 编码**（含 `D:\共享`），
必须**字节级**处理（按行找 `set COMMANDLINE_ARGS=` 再追加），绝不能按 UTF-8 读写。

## 8. ⚠️ 别把三类 OOM 混为一谈（2026-09-25 实测教训）

本机出现过**三种**都报 "out of memory" 的失败，**修法完全不同**：

| 报错 | 位置 | 性质 | 修法 |
|---|---|---|---|
| `hostbuf_grow: requested N beyond reserved M` | `hostbuf.c:46` | **主机** pinned 上限拒绝 | A 切档（0.45 / 0.50） |
| `hostbuf_read_file_slice: device copy failed result=2` | `hostbuf.c:283` | **设备**显存不足 | 抬 A **没用**，要降 B |
| `HostBuffer.read_file_slice failed` → `Sticky error detected` | `host_buffer.py:109` | 同上的 Python 侧表象 | 同上（降 B） |

**实测对照**：本机 A 已切到 0.50（`Enabled pinned memory 16193.0` 确认生效）后，
第三种**仍然崩在同一处** → 证明 A→B 是两条独立的链，**抬 A 治不了 B**。

## 9. ComfyUI 升级后

工具会因状态变成 `unknown` 而拒绝动手，这是**预期行为**。
处理方式：

1. 找到新版 `comfy/model_management.py` 里的两行：
   - Windows 分支的 `MAX_PINNED_MEMORY = ram * x`（A）
   - `DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE = N * 1024 ** 3`（B）
2. 照着 §5 的规则，必要时更新 `PAT_ANY` / `PAT_SUB` / `PAT_CAST_ANY` / `PAT_CAST_SUB` 四个正则
3. 重新跑 `--status` 确认 A、B 状态识别都正常，再切档

## 10. 验证记录（2026-09-25）

`tools/` 侧全部实测通过（在临时副本上做，**未碰真文件**；其间发现并修掉一个隔离缺陷）：

| 用例 | 结果 |
|---|---|
| A：0.45 ↔ 0.50 双向切档 | ✅ 文本正确、CRLF/字节数不变 |
| B：16 → 4 → 2 → 16 GiB | ✅ |
| A、B 组合（A=0.50 + B=4 同时生效） | ✅ |
| 只复原 A（B 保持）· 只复原 B（A 保持） | ✅ |
| 全复原后与 `.orig` **逐字节一致** | ✅ sha256 `83420988a417` |
| 幂等：已是该档再切 | ✅ 拒绝空操作 |
| 越界 `--to 0.90` / `--cast-to 0` / `99` / `abc` | ✅ 全部拒绝 |
| `unknown` A 值（0.42） | ✅ 拒绝（实测发现原实现会放行，已修） |
| `unknown` B 值（人工改成 8 GiB） | ✅ 拒绝 |
| 缺 `.orig` + 当前非原版 | ✅ 拒绝 |
| Linux 分支 `ram * 0.40` | ✅ 未被误伤 |
| ⚠ **`--backup-dir` 隔离失效** | ✅ **已修**：显式指定后禁用"回退找 `.orig`" |

> **事故记录**：修复隔离缺陷前，试跑把 6 个 bak 写进了真实 `patches/` —— 已全部清除，
> `.orig` 与历史备份完好。这是"隔离机制本身不可信"的典型，所以现在一经指定就锁定。

## 11. 上机现状（2026-09-25 15:06）

```
model_management.py:1378  DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE = 4 * 1024 ** 3   ← B 保守档
model_management.py:1593  MAX_PINNED_MEMORY = ram * 0.50                                ← A 缓存一采档

启动ComfyUI.bat            COMMANDLINE_ARGS += --vram-headroom 1                        ← 新增
```

**本轮进展（重要）**：B 档 16 → 4 GiB 之后，崩溃点**从 `host_buffer.py:109 read_file_slice failed`
后移到了 `memory_management.py:32 copy_from`** —— 说明 B 确实修掉了原来那条路，
只是又撞上了下一层。下一层的关键线索是 `PyTorch CUDA OOMs: 0`（分配器自己没 OOM），
指向 PyTorch 之外的 aimdo 分配，故加 `--vram-headroom 1`。

**待验证**：重启后看是否能越过 `qwen3vl.py:100 → ops.py:466 cast_bias_weight`。
若仍崩 → 下一轮加 `--verbose DEBUG`（`main.py:284` 会转成 `comfy_aimdo.control.set_log_debug()`）
抓 aimdo C 层日志。
