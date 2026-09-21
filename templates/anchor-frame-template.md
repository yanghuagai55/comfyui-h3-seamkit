# 硬切锚定帧 · 画面生成（喂 LLM 就复制这一份）

> 给它**片长 + 切点帧号 + 剧情** → 它输出**每个锚定帧的静态画面描述**。
> 你拿这些描述去图像模型出图 → 每张图用 `MiniMaxH3AddGuide` 锚到对应帧 → **切点被钉死**。
> 姊妹文档：六段式视频提示词用 `auto-prompt-template.md`，两份可以一起喂给同一个 LLM。

---

## 一、为什么需要锚定帧

**提示词里的时间戳只能"请求"模型在某帧切镜**——实测换模型后会偏 5~7 帧，而且不可控。

`MiniMaxH3AddGuide`（ComfyUI 0.34 原生，无需插件）换了个机制：

> 把一张图（或一小段视频）**锚**在第 N 帧，它会变成 keyframe condition latent，
> **每一步重新注入、永不去噪** —— **模型必须照它画**。

于是"切点准不准"从**模型的意愿**变成**你的硬约束** ✓

**代价**：你得提供那张画面，而且它必须**严格等于新镜头的第一帧**。这份模板就是让 LLM 把这张画面写清楚。

---

## 二、你要给 LLM 的输入

```text
片长：15 秒
切点帧号：85, 187, 272        ← 从 #56 的报告里抄（planned_cuts 那一行），不要自己算
剧情要求：<你的故事；如果已经有六段提示词，直接贴六段>
角色定义：<和提示词里的 subject_definitions 保持一致，逐字>
```

> 帧号必须是 **17 的倍数**（0.708s 的整数倍）。拿不准就先跑
> `python hardcut_math.py 15 "4.25,8.5,12.75" --mp 1.5`，报告里的 `cut frames` 直接抄。

---

## 三、给 LLM 的指令（整段复制）

```text
你现在是分镜师 + 图像提示词工程师。

任务：为下面这个片子写"锚定帧画面描述"。这些描述会拿去图像模型生成单帧图片，
再用 MiniMaxH3AddGuide 锚进视频的指定帧——所以每一张都必须是那个镜头的第一帧。

硬性规则：
1. 每个切点写一张（个数 = 给的帧号个数）。若给了 0 号帧需求，再加一张。
2. 只描述"这一帧的静止画面"。禁止动作过程、禁止时间词
   （然后 / 接着 / 逐渐 / 开始 / 继续）。
3. 每张必须写全六要素：机位角度 · 景别 · 构图 · 人物姿态与表情 · 光线 · 环境细节。
4. 必须与该镜头开头一致——它就是这个镜头的第一帧，不是中间帧。
5. 与上一镜至少有一项大改（机位或景别），否则切点在成片里看不出来。
6. 角色外观描述与「角色定义」逐字一致，全片不得漂移。
7. 40~80 词，逗号分隔的短语堆叠（图像模型友好），不要完整句、不要连词。
8. 不出现元信息："第 N 镜"、"cut"、"hard cut"、"时间戳"。

输出严格 JSON，不要解释：

{
  "clip_seconds": <片长>,
  "cuts": [<帧号...>],
  "anchors": [
    {"frame_idx": <帧号>, "shot": "[Shot 2]", "image_prompt": "<静态画面描述>"},
    ...
  ]
}
```

---

## 四、画面描述写什么（对照表）

| 要素 | ✗ 太虚 | ✓ 可出图 |
|---|---|---|
| 机位 | 看向她 | 低角度仰拍，机位在她膝盖高度 |
| 景别 | 近景 | 过肩中景，焦点在肩线前方，背景虚化 |
| 构图 | — | 人物偏左三分线，右侧留出天空与落花 |
| 姿态表情 | 她跑着 | 身体前倾，双臂摆到最高点，嘴微张，眉头紧 |
| 光线 | 白天 | 侧逆光，雨后湿地面反光，发梢有暖色轮廓光 |
| 环境 | 校园 | 樱花树下柏油小径，落花贴地，远处教室玻璃反光 |

**一句话判据**：把描述丢给图像模型，**不看上下文也能画出正确的一帧**，才算合格。

---

## 五、完整示例（15 秒 / 3 切点）

**输入**

```text
片长：15 秒
切点帧号：85, 187, 272
剧情要求：雷姆（蓝短发、女仆装）与高木（棕长发、校服）在雨后校园小径并排走，
         中途雷姆停下回头看镜头，最后切到高木的极近特写。
角色定义：<Subject 1> is Rem, a young woman with short sky-blue hair and a blue maid outfit.
         <Subject 2> is Takagi, a girl with long brown hair in a school uniform.
```

**输出**（LLM 应给出的 JSON）

```json
{
  "clip_seconds": 15,
  "cuts": [85, 187, 272],
  "anchors": [
    {
      "frame_idx": 85,
      "shot": "[Shot 2]",
      "image_prompt": "frontal medium two-shot at chest height, both girls fully in frame side by side, Rem in her blue maid outfit on the left, Takagi in school uniform on the right, bodies turned slightly toward each other, walking paused, wet asphalt path with fallen petals, overcast sky, soft even light, blurred sakura branches behind"
    },
    {
      "frame_idx": 187,
      "shot": "[Shot 3]",
      "image_prompt": "low-angle over-the-shoulder from behind Takagi, Rem in the left third stopping mid-step and turning her head back toward camera, sky-blue hair swinging, blue maid outfit, mouth slightly open, rain-wet path reflecting grey sky, shallow focus on her face, cold rim light along her hair edge"
    },
    {
      "frame_idx": 272,
      "shot": "[Shot 4]",
      "image_prompt": "extreme close-up of Takagi's face filling the frame, brown hair strands crossing her cheek, eyes wide and unblinking looking slightly off-camera, lips pressed, dark blurred background of wet branches, single soft side light, fine skin texture with raindrops"
    }
  ]
}
```

注意三张的共同点：**机位各不相同**（中景侧向 → 过肩仰角 → 极近特写）、**都是静止画面**、**每张都能独立成图** ✓

---

## 六、拿到描述之后怎么用

1. **出图**：把每张 `image_prompt` 丢给图像模型（Krea2 / 你的图像链），**一帧一张**
   - 分辨率随意，`AddGuide` 会自动缩到画布尺寸
   - 建议按画布比例出（16:9），构图才不会被裁
2. **接线**（每个切点一个 AddGuide，串起来）：

```text
条件节点(#8 / #41) ──positive──► [AddGuide frame_idx=85] ──► [AddGuide frame_idx=187] ──► … ──► guider
                                    ├─ vae    ← video VAE
                                    ├─ latent ← H3 AV latent（一采或二采的）
                                    └─ image  ← LoadImage（该帧的图）
```

3. **校验**：`frame_idx` 必须**逐字等于** `#56` 报告里的 `planned_cuts`
4. **二采**：如果想让二采也遵守锚点，高分辨率条件链上也要串一遍

---

## 七、检查清单

- [ ] 帧号与 `#56` 报告的 `planned_cuts` **完全一致**（抄，不要自己算）
- [ ] 每张描述都是**静止画面**（无动作词、无时间词）
- [ ] 相邻两张的**机位或景别差异明显**
- [ ] 角色外观描述**全片统一**（与 subject_definitions 逐字一致）
- [ ] 每张描述**不看上下文也能出图**
- [ ] 张数 = 切点数（如需锚首帧再 +1）
