# 长图切分识图（astrbot_plugin_long_image_reader）

> **本插件的全部代码与文档由 DSH（DeepSeek Harness）生成完成。**

## 安装

把整个目录放进 AstrBot 的 `data/plugins/`，重启 AstrBot 即可，不需要额外依赖（只用 Pillow）。

```bash
cd AstrBot/data/plugins
git clone https://github.com/APHu19/astrbot_plugin_long_image_reader.git
```

## 它解决什么问题

AstrBot 的 AI 配置页有「启用图片压缩」，下面还有一项「最大边长」（默认 1280）。
开启后框架对发给模型的图片做的是**按最长边等比缩放**：

```python
working_img.thumbnail((max_size, max_size), PILImage.Resampling.LANCZOS)
```

一张 750×20000 的长截图会被压成 48×1280 —— 文字彻底糊掉，模型只能瞎猜。
截图越完整，信息丢得越干净。

本插件不碰框架的压缩设置，而是在模型自己认为需要读长图时，把**原图**重新按
「纯色 / 边框行」切成若干段，每段最长边不超过 1280，交给识图模型逐段转成文字，
再把文字结果交回对话模型。

## 怎么触发

只有模型的工具调用才会触发，用户说「帮我看看这张图」不会自动生效：

1. 插件挂了 `on_llm_request`。当本条消息的图片实测是长图（超过阈值 + 长宽比达标）时，
   往 `req.extra_user_content_parts` 追加一行提示，告诉模型「图被压过了，要准确读字请调
   `read_long_image`」。
   **不要改成改 `system_prompt`**：system 排在历史之前，动它会让整段历史的前缀缓存失效。
   实测 60 条历史、1.6 万 token 的会话，改 system 后全价输入 15,789 token，
   走 `extra_user_content_parts` 只有 174 token。
2. 模型自己决定是否调用。调用才会切片。
3. 切片由插件直接送给识图模型（`provider.text_chat(image_urls=[...])`），
   **这条路不经过 `prepare_model_image`，切片不会被二次压缩。**

普通图片、没调工具的情况，插件完全不介入。

### 顺带掐掉框架的整图转述

框架自带的「图片转述」（`default_image_caption_provider_id`）会在 `on_llm_request`
**之前**执行，把整张压缩后的长图丢给转述模型读一遍，产出一段 `<image_caption>`。
对长图来说这一步既贵又读不准：压缩后 1080×8100 会变成 171×1280，字全糊了。

插件在 `initialize()` 里包住 `astr_main_agent._ensure_img_caption`，检测到长图时
直接返回、不做转述，同时清空 `req.image_urls`（否则框架的
`_select_image_chat_provider` 会把整轮请求切给视觉模型）并补一条占位说明。
普通图片仍走原来的转述路径。开关是 `skip_caption_for_long_image`，默认开。

## 为什么要用识图模型而不是把图塞回对话

早期版本把切片以 `ImageContent` 返回给工具调用方，让对话模型自己看。实测有两个问题：

- 框架在 `tool_loop_agent_runner.py:1054-1087` 把工具返回的图拼进上下文时**不切换 provider**，
  永远交给当前对话模型。对话模型不支持图片输入（`modalities` 不含 `image`）时，
  这些图还会被 `sanitize_contexts_by_modalities` 替换成 `[Image]` 占位文本，
  模型就只能跳过识图直接编。
- 每张切片以 `data:image/jpeg;base64,...` 内联进上下文，单片 200–400KB，十片就是几 MB。
  上下文被顶爆后触发压缩，历史被替换成摘要 —— 表现就是「文本被压缩」，
  而且模型一次能看的图数量会肉眼可见地变少。

所以现在默认走「识图模型转文字」：对话模型只收到一段文字，上下文占用从几 MB 降到几千字符。


## 切分算法

以竖向长图为例（横向长图先做一次转置，切完再转回来）：

1. 先按短边缩放到最大边长（短边超过它才缩），这样每片都足够清晰又不超限。
2. 把图转灰度、宽度缩到 256（只影响切点计算），逐行算灰度标准差。
   标准差 ≤ 阈值（默认 3.0），并且上下各 2 行也都如此，这一行才算「纯色行」。
   连续 5 行都均匀才算切点 —— 对应「识别某一块是否足够像纯色」。
3. 从第 0 行起，在窗口 `(当前, 当前+最大边长]` 内**从后往前**找第一个纯色行当切点
   （越靠下切得越少，片数最少）。窗口内没有纯色行时，取窗口内标准差最小的一行当切点，
   相当于硬切但尽量不压在字上。
4. 切完的起点回退 2% 宽度做重叠，避免切缝正好压在一行字上导致两头都缺半个字。
5. 重复到剩余高度不足一个窗口为止，最后一片直接包含到最后一行。

竖切、横切共用同一套代码：横向长图 `transpose(TRANSPOSE)` 后按竖向处理，切完转回来
（`TRANSPOSE` 自逆，`T(T(img)) == img`）。

## 两种交付方式

| 模式 | 行为 | 适用 |
| --- | --- | --- |
| 默认（`two_stage_enabled: true`） | 插件调识图模型逐片转文字，再调汇总模型合并，只把文字返回当前模型 | 几乎所有情况 |
| 直传（`two_stage_enabled: false`） | 切片以图片形式返回，当前模型一次看到前 `direct_max_images` 片 | 当前模型支持图片，且片数不多 |

两段式的并发数、超时、两段提示词都可配置。识图模型留空时会自动挑一个 `modalities` 含 `image`
的已启用模型；都挑不到就明确报错，不会把图硬塞给文本模型。

汇总结果过长时会按 `summary_chunk_chars` 分段汇总再合并一次，避免汇总请求自己撑爆上下文。

## 配置项

| 项 | 默认 | 说明 |
| --- | --- | --- |
| `enable` | 开 | 总开关 |
| `hint_enabled` | 开 | 是否向模型注入长图提示 |
| `skip_caption_for_long_image` | 开 | 长图跳过框架的整图转述（省一次视觉模型调用） |
| `max_side_override` | 0 | 单片最大边长，0 = 跟随框架的「最大边长」 |
| `slice_threshold` | 0 | 触发切分的边长阈值，0 = 与单片最大边长相同 |
| `min_aspect_ratio` | 1.5 | 长宽比门槛，正方形大图不切 |
| `max_slices` | 64 | 片数上限，超出整张放弃并告知模型 |
| `two_stage_enabled` | **开** | 两段式识别 |
| `vision_provider_id` | 空 | 识图模型，空 = 自动选第一个支持图片的模型 |
| `summary_provider_id` | 空 | 汇总模型，空 = 当前对话模型 |
| `summary_chunk_chars` | 12000 | 汇总分段长度 |
| `include_message_text` | 开 | 把用户消息全文传给汇总模型 |
| `direct_max_images` | 6 | 直传模式最多回传几张 |
| `vision_concurrency` | 4 | 两段式并发数 |
| `vision_timeout` | 90 | 两段式单次调用超时（秒） |
| `vision_total_budget` | 100 | 两段式逐片识别总预算（秒），超出的片标记为超时 |
| `slice_prompt` | 空 | 逐片识别提示词，空 = 内置默认 |
| `summary_prompt` | 空 | 汇总提示词，空 = 内置默认 |
| `uniform_stddev_threshold` | 3.0 | 纯色行判定阈值，越小越严格 |
| `overlap_ratio` | 0.02 | 相邻片重复比例 |
| `max_scan_width` | 256 | 判定纯色行时的扫描宽度 |
| `output_format` | jpeg | jpeg / png |
| `temp_file_policy` | keep | keep / delete / delete_after_24h |

两个提示词的占位符：`slice_prompt` 支持 `{index}` / `{total}`；
`summary_prompt` 支持 `{message}`（用户提出识图要求时的**消息全文**）/ `{parts}`（逐片识别结果）/ `{total}`。

汇总模型拿到的是完整消息而不是只有「用户想看图」这句话，所以像
「把第三段和第五段的时间戳和金额读给我」这种带明确诉求的要求能被正确照顾到。
模型调用工具时另外传了 `focus` 的话，会追加在消息全文后面，不会被吞掉。

不想传全文就把 `include_message_text` 关掉：这时 `{message}` 只保留 `focus`，
群聊里消息很长时能省下不少 token，代价是汇总模型看不到用户的具体诉求。

## 不重复付费

模型调用工具是「自己决定」的，同一张图很容易被连着调好几次（每次都完整重跑一遍
逐片识图）。插件从两个方向挡住重复计费：

- **同一条消息内**：第一次识别完就在事件上打标记，再调直接返回
  「这张长图在本条消息里已经识别过」，不再产生任何模型调用。
- **跨消息**：按「会话 + 图片内容哈希 + focus」缓存识别结果，最多 32 张。
  用户隔几分钟再发同一张图，或模型在新一轮里又调一次，直接复用上次结果。

缓存只存文字结果，不占多少内存；进程重启后失效。

## 临时文件目录

切片写到 **AstrBot 的 data 目录**下的 `temp/long_image/<日期>/<会话摘要>_<时间戳>/`，
位置取自配置文件 `cmd_config.json` 所在目录，与启动时的工作目录无关。

早期版本用 `get_astrbot_temp_path()`，那个函数跟随进程 cwd，从别处启动 AstrBot 时
会把切片写到 `gsuid_core/data/temp/long_image` 之类的错误位置。已经清掉了旧目录。

## 已知限制

- **工具调用总时长默认 120 秒**（`agent_runner.config.misc.tool_call_timeout`）。
  两段式处理很多片时可能超时，这时调大该配置，或减少片数（`max_slices`）。
- 片数超过 `max_slices` 时**整张放弃**，不降采样也不截断 —— 这是有意的：
  一张 10 万像素高的图切出来没人读得完，提示用户分几张发更实际。
- 只处理当前消息里的图片。历史消息里的图片、引用消息里的图片不会切。
- 切点判定基于行灰度标准差。纯色渐变背景、密集纹理背景上可能找不到纯色行，
  这时会退化成窗口内最均匀处硬切。
- `skip_caption_for_long_image` 是**改写框架函数**实现的（和 AstrNa 一样的做法）。
  多个插件同时改写 `_ensure_img_caption` 时后加载的在外层，长图判断仍然生效。
