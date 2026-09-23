# 环境要求：pinned-memory 上限（aimdo host buffer）

> 这份文档记录**一个必须对 ComfyUI 核心做的改动**。
> 核心文件不在本仓库内，**ComfyUI 升级后改动可能被覆盖，照本文重打即可**。
>
> 交互式工具：`tools/pinned_memory_patch.py`（菜单 `1` 打补丁 / `2` 复原）

---

## 1. 现象

二采跑着跑着，某一段采样中途报（会反复刷）：

```
aimdo: src/hostbuf.c:46:ERROR:hostbuf_grow: requested 27434295296 bytes
                             beyond reserved host buffer 27235385344
```

迷惑之处：**前面几段能正常跑完，后面某一段才撞线**，而且和段长无关
（实测 119 帧的段过了、102 帧的段反而挂）。

⚠️ 别和另一个错误混：`aimdo memory compile error` 是**假报错**（真身是 CUDA OOM），
重启就能缓解；**这个是真实的容量拒绝**，重启不会自愈。

## 2. 为什么

`aimdo` 为模型权重准备的 **页锁定（pinned）主机内存**上限这么算：

```python
# comfy/model_management.py
MAX_PINNED_MEMORY = ram * 0.40                      # Windows 分支（ram = 物理内存）
def pinned_hostbuf_size(size):
    return max(0, int(min(size, MAX_PINNED_MEMORY) * 2))     # ← ★这个 x2

# comfy/model_patcher.py  （每个模型建 4 个 HostBuffer，reserve 相同）
HostBuffer(0, 64 * 1024 * 1024, pinned_hostbuf_size(self.model_size()))
```

⇒ **上限 ≈ 0.80 × 物理内存**

| 项 | 本机（32 GB） |
|---|---|
| 物理内存 | 31.63 GiB |
| `MAX_PINNED_MEMORY` | 12.65 GiB（启动日志打印 `Enabled pinned memory 12954.0`） |
| host buffer 上限 = ×2 | **25.30 GiB**（实测 reserve 25 365 385 344 B） |
| H3 二采实测需求 | **25.38 – 25.60 GiB** |
| **净差** | **15 – 235 MiB** |

上限在**建模型时定死**，而每次请求的大小随内容/激活浮动 → 每段都是掷骰子。

## 3. 补丁

`comfy/model_management.py` 约 1593 行，**只改一行**：

```diff
@@ -1590,7 +1590,7 @@
     if is_nvidia() or is_amd():
         ram = get_total_memory(torch.device("cpu"))
         if WINDOWS:
-            MAX_PINNED_MEMORY = ram * 0.40  # Windows limit is apparently 50%
+            MAX_PINNED_MEMORY = ram * 0.45  # Windows limit is apparently 50%
         else:
             MAX_PINNED_MEMORY = max(ram * 0.40, min(ram * 0.90, ram - 4 * 1024 ** 3, ram + swap - 16 * 1024 ** 3))
```

改后上限 ≈ **28.47 GiB**，需求 25.6 GiB 有 ~2.9 GiB 余量，
同时仍给系统留 10% 物理内存。

### 为什么不用 `--high-ram`

`pinned_hostbuf_size()` 的另一个分支是 `size * 2` —— **上限整个去掉**。
在 32 GB 机器上跑 32.4 GB 的模型时，这等于拆掉护栏：

- **pinned 内存不能换页** → 要多了不是"干净报错"，而是掉页 / 整机卡死
- 它还会连带 `comfy/cli_args.py:287` 的 `args.cache_classic = True`，改变缓存行为

**所以选择只把上限抬 12.5%，护栏留着。**

## 4. 内存门槛（同一公式推出来的）

| 机器内存 | 可用(≈) | 上限 @0.40 | 上限 @0.45 | vs 需求 25.6 GiB |
|---|---|---|---|---|
| 64 GB | 63 GiB | 50.4 | 56.7 | ✓ 轻松 |
| 32 GB | 31.6 GiB | 25.30 | **28.47** | 0.40 差 0.3 ✗ / 0.45 余 2.9 ✓ |
| 24 GB | 23.5 GiB | 18.8 | 21.2 | ✗ 差 4.4 |
| 16 GB | 15.6 GiB | 12.5 | 14.0 | ✗ 差 11.6 |

（"可用" ≈ 标称内存减约 0.4 GiB 硬件保留）

**结论：二采建议 ≥32 GB 内存，并且要打这个补丁。**

## 5. 打 / 验 / 退

```bash
D:\comfyui\comfyenv\python.exe <本仓库>\tools\pinned_memory_patch.py
# 目标文件不在默认位置时，把路径当参数传进去：
#   ...python pinned_memory_patch.py D:\some\ComfyUI\comfy\model_management.py
```

工具行为：

- 目标文件与备份目录**从脚本自身位置推导**（不硬编码），可用参数覆盖目标
- 正则**锚在 `if WINDOWS:` 上**单次替换；命中数 ≠ 1 就拒绝 —— **不会误改 Linux 分支那行同样的 0.40**
- 状态机 `unpatched` / `patched` / `unknown`；`unknown`（例如升级过）**一律拒绝，不硬来**
- 备份：`model_management.py.orig` **只在未打补丁时写一次、绝不覆盖**，另存带时间戳的本次备份
- 复原 = **逐字节拷贝**，不靠反向替换
- 备份目录优先用"已经有 `.orig` 的那个"，否则用 `<ComfyUI>/user/pinned_memory_patch/`

### 验证

**重启 ComfyUI**，启动日志应出现：

```
Enabled pinned memory 14573.0        <- 原来是 12954.0
```

（数字 = 物理内存 × 0.45，取整到 MiB。物理内存不同数字也不同。）

### 回退

工具的 `2`，或手工把 `.orig` 覆盖回去。**`.orig` 不要删。**

## 6. 打完补丁还在报同样的错怎么办

说明需求超过了 25.6 GiB，那就不是"差一点"，方向要改成**减小需求**：

- 降二采画布 `second_megapixels`（只压激活，省得有限）
- 换更小的底模 / 更低的精度
- 缩短最长段（`target_segment_seconds`）

每条都要实测，不要靠推断。
