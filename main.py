"""长图切分识图插件。

模型主动调用工具时，把超长图片按纯色/边框行切成可读片段，
绕开「启用图片压缩」把长图压成一条细线导致的细节丢失。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import shutil
import time
from pathlib import Path

import mcp.types
from PIL import Image as PILImage

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.provider import Provider, ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

DEFAULT_MAX_SIZE = 1280
MIN_MAX_SIZE = 256
UNIFORM_SPREAD = 2
SCAN_PIXEL_BUDGET = 2_000_000
TEMP_SUBDIR = "long_image"

DEFAULT_SLICE_PROMPT = (
    "这是一张超长图片按从上到下顺序切分后的第 {index}/{total} 片。"
    "请逐字转写你在图中看到的全部文字，保留原有的分段与列表层级。"
    "界面元素、图表、图片内容用一句括号说明即可。"
    "不要总结，不要省略，不要补写没看到的内容。"
)
DEFAULT_SUMMARY_PROMPT = (
    "用户的原问题或关注点：{focus}\n\n"
    "下面是一张超长图片按从上到下顺序切分成的 {total} 片，以及每片各自的识别结果：\n\n"
    "{parts}\n\n"
    "请把这些结果合并成一份连续、去重、不丢信息的识图结果，"
    "保留关键文字、数字、时间、名称与结构；无法辨认处标注「无法辨认」。"
)


def _probe_size(path: str) -> tuple[int, int] | None:
    try:
        with PILImage.open(path) as image:
            return image.size
    except Exception as exc:  # 图片来自用户，文件损坏或格式异常都只跳过
        logger.debug(f"[long_image_reader] 读取图片尺寸失败：{path} / {exc}")
        return None


def _prepare_work_image(path: str, max_side: int, horizontal: bool) -> PILImage.Image:
    with PILImage.open(path) as source:
        source.load()
        image = source.convert("RGB")
    short_side = min(image.size)
    if short_side > max_side:
        scale = max_side / short_side
        size = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        image = image.resize(size, PILImage.Resampling.LANCZOS)
    if horizontal:
        image = image.transpose(PILImage.Transpose.TRANSPOSE)
    return image


def _row_stddev(gray: PILImage.Image, max_width: int) -> list[float]:
    width, height = gray.size
    target = min(max_width, width)
    if height > 0 and target * height > SCAN_PIXEL_BUDGET:
        target = max(16, SCAN_PIXEL_BUDGET // height)
        target = min(target, width)
    if target < width:
        gray = gray.resize((target, height), PILImage.Resampling.BOX)
    width = gray.size[0]
    if width < 2 or height < 1:
        return [0.0] * height
    data = gray.tobytes()
    scores: list[float] = [0.0] * height
    for y in range(height):
        row = data[y * width : (y + 1) * width]
        count = len(row)
        total = sum(row)
        square = sum(value * value for value in row)
        mean = total / count
        variance = square / count - mean * mean
        scores[y] = variance**0.5 if variance > 0.0 else 0.0
    return scores


def _cuttable_rows(scores: list[float], threshold: float) -> list[bool]:
    plain = [score <= threshold for score in scores]
    total = len(plain)
    result: list[bool] = [False] * total
    for y in range(total):
        low = max(0, y - UNIFORM_SPREAD)
        high = min(total, y + UNIFORM_SPREAD + 1)
        result[y] = all(plain[low:high])
    return result


def _plan_ranges(
    total: int,
    scores: list[float],
    cuttable: list[bool],
    window: int,
    overlap: int,
) -> list[tuple[int, int]]:
    if total <= window:
        return [(0, total)]
    ranges: list[tuple[int, int]] = []
    start = 0
    advance = max(1, window // 2)
    while total - start > window:
        hard_end = min(start + window, total)
        scan_high = hard_end - 1
        scan_low = start + advance
        cut = hard_end
        if scan_low <= scan_high:
            picked = -1
            for y in range(scan_high, scan_low - 1, -1):
                if cuttable[y]:
                    picked = y
                    break
            if picked > start:
                cut = picked
            else:
                best = scan_low
                best_score = scores[scan_low]
                for y in range(scan_low + 1, scan_high + 1):
                    if scores[y] < best_score:
                        best_score = scores[y]
                        best = y
                cut = best
        ranges.append((start, cut))
        start = max(cut - overlap, start + 1)
    ranges.append((start, total))
    return ranges


def _write_slices(
    work: PILImage.Image,
    ranges: list[tuple[int, int]],
    out_dir: Path,
    fmt: str,
    horizontal: bool,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(ranges)
    paths: list[Path] = []
    for index, (start, end) in enumerate(ranges, start=1):
        # 切点在转置坐标系里算的，先在转置坐标系里裁，再把每片转回原方向
        piece = work.crop((0, start, work.width, end))
        if horizontal:
            piece = piece.transpose(PILImage.Transpose.TRANSPOSE)
        name = f"{index:02d}_of_{total:02d}"
        if fmt == "png":
            path = out_dir / f"{name}.png"
            piece.save(path, format="PNG", optimize=True)
        else:
            path = out_dir / f"{name}.jpg"
            piece.save(path, format="JPEG", quality=90, optimize=True)
        paths.append(path)
    return paths


@register(
    "astrbot_plugin_long_image_reader",
    "APHu19",
    "把超长图片按纯色/边框行切成清晰片段再交给模型阅读，绕开图片压缩导致的细节丢失。",
    "1.0.0",
)
class LongImageReader(Star):
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config

    # ---------- 配置读取 ----------

    def _cfg_bool(self, key: str, default: bool) -> bool:
        raw = self.config.get(key, default)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return default

    def _cfg_int(self, key: str, default: int) -> int:
        raw = self.config.get(key, default)
        if isinstance(raw, bool):
            return default
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            return int(raw)
        if isinstance(raw, str):
            try:
                return int(raw.strip())
            except ValueError:
                return default
        return default

    def _cfg_float(self, key: str, default: float) -> float:
        raw = self.config.get(key, default)
        if isinstance(raw, bool):
            return default
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw.strip())
            except ValueError:
                return default
        return default

    def _cfg_str(self, key: str, default: str) -> str:
        raw = self.config.get(key, default)
        if isinstance(raw, str):
            return raw.strip()
        return default

    def _resolve_max_side(self) -> int:
        override = self._cfg_int("max_side_override", 0)
        if override > 0:
            return max(MIN_MAX_SIZE, override)
        value = DEFAULT_MAX_SIZE
        settings = self.context.get_config().get("provider_settings")
        if isinstance(settings, dict):
            options = settings.get("image_compress_options")
            if isinstance(options, dict):
                raw = options.get("max_size")
                if isinstance(raw, int) and not isinstance(raw, bool):
                    value = raw
        return max(MIN_MAX_SIZE, value)

    def _should_split(self, width: int, height: int, max_side: int) -> bool:
        long_side = max(width, height)
        short_side = max(1, min(width, height))
        threshold = self._cfg_int("slice_threshold", 0)
        if threshold <= 0:
            threshold = max_side
        if long_side <= threshold:
            return False
        min_ratio = self._cfg_float("min_aspect_ratio", 1.5)
        if min_ratio > 0 and long_side / short_side < min_ratio:
            return False
        return True

    # ---------- 临时目录 ----------

    def _temp_root(self) -> Path:
        return Path(get_astrbot_temp_path()) / TEMP_SUBDIR

    def _sweep_expired(self) -> None:
        root = self._temp_root()
        if not root.is_dir():
            return
        deadline = time.time() - 24 * 3600
        for day_dir in root.iterdir():
            if not day_dir.is_dir():
                continue
            try:
                if day_dir.stat().st_mtime < deadline:
                    shutil.rmtree(day_dir, ignore_errors=True)
            except OSError:
                continue

    def _finalize_temp(self, out_dir: Path) -> None:
        policy = self._cfg_str("temp_file_policy", "keep")
        if policy == "delete":
            shutil.rmtree(out_dir, ignore_errors=True)
        elif policy == "delete_after_24h":
            self._sweep_expired()

    # ---------- 图片与模型 ----------

    @staticmethod
    def _collect_images(event: AstrMessageEvent) -> list[Image]:
        return [comp for comp in event.get_messages() if isinstance(comp, Image)]

    async def _resolve_source(self, component: Image) -> str | None:
        if component.path and os.path.isfile(component.path):
            return component.path
        try:
            return await component.convert_to_file_path()
        except Exception as exc:  # 平台适配器给的引用可能已失效，回退到本地路径
            logger.debug(f"[long_image_reader] 解析原图失败：{exc}")
            return None

    async def _resolve_provider(self, provider_id: str, umo: str) -> Provider | None:
        if provider_id:
            candidate = self.context.get_provider_by_id(provider_id)
            return candidate if isinstance(candidate, Provider) else None
        return await self.context.get_using_provider_async(umo)

    async def _describe_slices(
        self,
        event: AstrMessageEvent,
        paths: list[Path],
        focus: str,
    ) -> str:
        umo = event.unified_msg_origin
        vision = await self._resolve_provider(self._cfg_str("vision_provider_id", ""), umo)
        if vision is None:
            return "没有取到可用的识图模型，两段式识别无法进行。切分文件已写入临时目录，可供人工查看。"

        slice_prompt = self._cfg_str("slice_prompt", "") or DEFAULT_SLICE_PROMPT
        semaphore = asyncio.Semaphore(max(1, self._cfg_int("vision_concurrency", 4)))
        call_timeout = max(10, self._cfg_int("vision_timeout", 90))
        budget = max(20, self._cfg_int("vision_total_budget", 100))
        total = len(paths)

        async def describe(index: int, path: Path) -> str:
            prompt = slice_prompt.replace("{index}", str(index)).replace("{total}", str(total))
            async with semaphore:
                try:
                    response = await asyncio.wait_for(
                        vision.text_chat(prompt=prompt, image_urls=[str(path)]),
                        timeout=call_timeout,
                    )
                except Exception as exc:  # 单张失败不该拖垮整次识图
                    logger.warning(f"[long_image_reader] 第 {index}/{total} 片识别失败：{exc}")
                    return f"[第 {index}/{total} 片识别失败]"
            text = (response.completion_text or "").strip()
            return text or f"[第 {index}/{total} 片没有识别出内容]"

        tasks = [asyncio.create_task(describe(index, path)) for index, path in enumerate(paths, start=1)]
        done, pending = await asyncio.wait(tasks, timeout=budget)
        for task in pending:
            task.cancel()
        parts: list[str] = []
        for index, task in enumerate(tasks, start=1):
            if task in done and not task.cancelled() and task.exception() is None:
                parts.append(task.result())
            else:
                parts.append(f"[第 {index}/{total} 片超时未识别]")

        merged = "\n\n".join(f"### 第 {index}/{total} 片\n{text}" for index, text in enumerate(parts, start=1))
        summary_prompt = self._cfg_str("summary_prompt", "") or DEFAULT_SUMMARY_PROMPT
        prompt = (
            summary_prompt.replace("{focus}", focus or "（用户没有额外说明）")
            .replace("{parts}", merged)
            .replace("{total}", str(total))
        )
        summary_provider = await self._resolve_provider(self._cfg_str("summary_provider_id", ""), umo)
        if summary_provider is None:
            summary_provider = vision
        try:
            response = await asyncio.wait_for(
                summary_provider.text_chat(prompt=prompt),
                timeout=call_timeout,
            )
        except Exception as exc:
            logger.warning(f"[long_image_reader] 汇总失败：{exc}")
            return merged
        summary = (response.completion_text or "").strip()
        return summary or merged

    # ---------- 工具 ----------

    @filter.llm_tool(name="read_long_image")
    async def read_long_image(
        self,
        event: AstrMessageEvent,
        image_index: int = 1,
        focus: str = "",
    ) -> mcp.types.CallToolResult | str:
        """把本条消息里的超长图片切成多张清晰片段，逐片读出其中的文字。

        只在图片很长、直接查看会被压缩到看不清时调用；普通图片不要调用。

        Args:
            image_index(int): 第几张图片，从 1 开始计数，默认 1。
            focus(str): 用户想重点看清的内容或原问题，可为空。
        """
        if not self._cfg_bool("enable", True):
            return "长图切分功能已被管理员关闭。"

        images = self._collect_images(event)
        if not images:
            return "本条消息里没有图片，无法切分。"
        if image_index < 1 or image_index > len(images):
            return f"图片序号超出范围：本条消息共有 {len(images)} 张图片。"

        component = images[image_index - 1]
        source = await self._resolve_source(component)
        if not source:
            return "没能取到这张图片的原文件，无法切分。"

        size = await asyncio.to_thread(_probe_size, source)
        if size is None:
            return "这张图片读不出来（可能已损坏或格式不支持），无法切分。"
        width, height = size
        max_side = self._resolve_max_side()
        if not self._should_split(width, height, max_side):
            threshold = self._cfg_int("slice_threshold", 0) or max_side
            return (
                f"这张图是 {width}×{height}，未超过切分阈值 {threshold}，"
                "不会被压缩到看不清，直接查看即可。"
            )

        horizontal = width > height
        work = await asyncio.to_thread(_prepare_work_image, source, max_side, horizontal)
        gray = work.convert("L")
        scores = await asyncio.to_thread(_row_stddev, gray, self._cfg_int("max_scan_width", 256))
        cuttable = _cuttable_rows(scores, self._cfg_float("uniform_stddev_threshold", 3.0))
        window = max_side
        overlap = int(window * self._cfg_float("overlap_ratio", 0.02))
        ranges = _plan_ranges(work.height, scores, cuttable, window, overlap)

        max_slices = max(1, self._cfg_int("max_slices", 64))
        if len(ranges) > max_slices:
            return (
                f"这张图 {width}×{height} 切分后需要 {len(ranges)} 片，超过上限 {max_slices} 片，"
                "已放弃处理。请提示用户把这部分截图分成几张发送，或提供更小的原图。"
            )

        fmt = "png" if self._cfg_str("output_format", "jpeg") == "png" else "jpeg"
        digest = hashlib.sha1(event.unified_msg_origin.encode("utf-8")).hexdigest()[:8]
        out_dir = self._temp_root() / time.strftime("%Y%m%d") / f"{digest}_{int(time.time())}"
        try:
            paths = await asyncio.to_thread(_write_slices, work, ranges, out_dir, fmt, horizontal)
        except OSError as exc:
            logger.error(f"[long_image_reader] 写切片失败：{exc}")
            return "切分后的图片写入临时目录失败，无法继续。"

        total = len(paths)
        orientation = "横向" if horizontal else "纵向"
        if self._cfg_bool("two_stage_enabled", False):
            result_text = await self._describe_slices(event, paths, focus)
            self._finalize_temp(out_dir)
            return (
                f"已把这张 {width}×{height} 的{orientation}长图切成 {total} 片并逐片识别，"
                f"以下是合并后的识图结果：\n\n{result_text}"
            )

        contents: list[mcp.types.TextContent | mcp.types.ImageContent] = [
            mcp.types.TextContent(
                type="text",
                text=(
                    f"已把这张 {width}×{height} 的{orientation}长图切成 {total} 片"
                    f"（每片最长边不超过 {window}px），按从上到下的顺序附在下方。"
                    "请逐片阅读后再回答，不要臆测没有看到的部分。"
                ),
            )
        ]
        mime = "image/png" if fmt == "png" else "image/jpeg"
        for path in paths:
            payload = base64.b64encode(await asyncio.to_thread(path.read_bytes)).decode("ascii")
            contents.append(mcp.types.ImageContent(type="image", data=payload, mimeType=mime))
        self._finalize_temp(out_dir)
        return mcp.types.CallToolResult(content=list(contents))

    # ---------- 提示注入 ----------

    @filter.on_llm_request()
    async def inject_long_image_hint(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        if not self._cfg_bool("enable", True) or not self._cfg_bool("hint_enabled", True):
            return
        max_side = self._resolve_max_side()
        for component in self._collect_images(event):
            candidate = component.path or component.file
            if not candidate or not os.path.isfile(candidate):
                continue
            size = await asyncio.to_thread(_probe_size, candidate)
            if size is None:
                continue
            width, height = size
            if not self._should_split(width, height, max_side):
                continue
            req.system_prompt += (
                f"\n\n[系统提示] 用户本条消息附带了一张 {width}×{height} 的超长图片，"
                "框架会把它压缩后交给你，细节多半看不清。"
                "如果需要准确读取图中的文字，请调用 read_long_image 工具，"
                "它会把原图切成清晰的片段再给你。"
            )
            return
