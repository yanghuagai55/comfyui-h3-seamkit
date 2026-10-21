# 提示词模板（官方六段骨架）

> 母本：官方 `VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`。
> 本文件为**官方标准六段结构骨架**，不带任何特定剧情的假想示例。
> 按槽位括号 `<...>` 填入实际内容即可。

---

## 一、官方六段骨架模板

```text
subject_definitions:
<Subject 1> is <角色/主体> in <Picture 1>, <外形特征描述>.
<Subject 2> is <场景/道具> in <Picture 2>, <特征描述>.

summary:
[reference generation] The target video is a <时长>-second clip with native stereo sound,
executed as a <N>-shot sequence with hard cuts at <MM:SS.mmm>[, <MM:SS.mmm> ...],
with every segment no longer than <target_segment_seconds> seconds.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2], ...): fully_preserved - <具体保持不变的特征，如五官/服装/体型>.
<Subject 2> (appears in [Shot 1], [Shot 2], ...): fully_preserved - <具体保持不变的特征，如环境/色调/风格>.

detailed_description:
<风格/画质/光影定调 —— 必须在 [Shot 1] 之前>.
[Shot 1] <构图/景别 + 主体站位与动作 + 环境光照 + 镜头运动 + 声音/对白>.
[Shot 2] At <MM:SS.mmm>, <机位/景别变化 + 动作接续 + 新镜头运动 + 声音/对白>.
[Shot 3] At <MM:SS.mmm>, <再次换景别/机位 + 动作收束>.

overall_soundscape:
<环境音 + 物理动作音效，跨切点自然连贯不中断>.

non_diegetic_music:
<配乐风格/乐器/节奏，或填 N/A>.
```

> 💡 **绿幕模式（`green_screen_mode=true`）简写说明**：
> 如果在上游节点勾选了绿幕模式，不需要分镜：
> - `detailed_description` 中只需写 `[Shot 1] <人物在绿幕前的连贯动作与神态描述>`，**无需写 `[Shot 2] At ...` 时间戳**；
> - `summary` 中的 `executed as ... with hard cuts at ...` 亦无需写切点，上游直接放行。

---

## 二、官方硬规则与槽位填写要点

### 1. `detailed_description` 分镜与 `Shot At` 规则
- `[Shot 1]` **不带时间戳**。
- `[Shot 2]` 起必须以 **`At MM:SS.mmm, `** 开头（如 `[Shot 2] At 00:04.958, <内容>`）。
  - 时间戳直接参考上游规划节点报告里的 `cut points`；
  - `At` 后面写：**换什么机位/景别（如全景切特写） + 人物接续做什么动作 + 镜头怎么动**。
- ⚠ **`[Shot N]` 是行首识别前缀**：只在行首作为镜头声明使用。句中引用前面镜头请改用文字（如 `the opening wide shot`），不要再出现 `[Shot 1]`。

### 2. 媒体标签必须与节点连线一一对应
未连线而在提示词里写了标签会导致报错：
| 提示词标签 | 对应的节点输入槽位 | 说明 |
|---|---|---|
| `<Subject N>` / `<Picture N>` | `ref_images.ref_image_{N-1}` | 角色/物体/场景参考图 |
| `<Audio N>` | `ref_audios.ref_audio_{N-1}` | 音频参考（没接音频线就**不要写 `<Audio N>`**） |
| `<Video N>` | `ref_videos.ref_video_{N-1}` | 视频参考 |

### 3. retention 官方合法标签
- **画面类**（`<Subject>` / `<Picture>` / `<Video>`）：
  `fully_preserved` · `partially_preserved` · `attribute_transfer` · `weak_reference`
- **音频类**（`<Audio>`）：
  `fully_copy` · `partially_copy` · `reference` · `weak_reference`

### 4. blocking（站位与朝向必须写死）
- 两个分段窗口是独立采样的，主体人物的站位、面部朝向、视线方向必须明确写死（例如 `facing the camera directly, shoulders square to the lens`），防止分段拼接处出现姿态跳变。

---

## 三、快速开机检查

- [ ] 六段结构完整，大小写与冒号正确
- [ ] 媒体标签都有实际连线（没接音频参考线则删掉 `<Audio N>`）
- [ ] **硬切分镜模式**：`[Shot 2]` 起带行首 `At MM:SS.mmm, `，相邻镜头有景别变化，无 `continuous take` 措辞
- [ ] **绿幕模式**：开启 `green_screen_mode`，只需写连贯动作，无需写 `At` 时间戳
