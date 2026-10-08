#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiAspect-4K-1M 图片批量下载脚本（串行处理 json + 多线程并行下载 + 断点续传 + 进度条/ETA）

目录约定（默认）:
    data_jsons/0001.json ... data_jsons/1008.json     # 元数据
    datas/0001/0001.jpg ...                           # 下载的图片
    datas/failed_downloads.tsv                        # 失败记录

处理流程:
    1. 串行遍历输入的数字编号列表（处理完一个 json 再处理下一个）；
    2. 对每个编号 N:
         - 读取 data_jsons/NNNN.json；
         - 给文件里**每一条**记录加上 "id" 字段，从 1 开始、按记录总数的位数补零
           （1000 条 -> 0001..1000；230 条 -> 001..230）；
         - 创建 datas/NNNN/ 目录；
         - 取前 max_per_json 条（默认全部）作为任务列表，多线程并行下载，
           文件名即 id（如 0001.jpg），后缀按响应 Content-Type 决定；
         - 每成功下载一张，就在该记录里写入 "image_path"（相对 json 所在目录的路径，
           如 ../datas/0001/0001.jpg）；
         - 处理结束后把新增字段写回原 json（可用 --no-json-update 关闭）。
    3. 断点续传: 目标目录里已存在同名图片（任意图片后缀）就跳过；未完成的
       .<id>.part 临时文件会用 HTTP Range 接着下。

用法示例:
    python download_images.py 1 2 3 -m 100 -j 8
    python download_images.py 1-20 --max-per-json 500 --workers 16
    python download_images.py all -j 16
    python download_images.py 1 5 9 --ids-file more_ids.txt -m 50
    nohup python download_images.py 1-1008 -j 16 > download.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

try:  # requests 更快、连接复用更好；缺失时自动退回 urllib
    import requests
except Exception:  # pragma: no cover
    requests = None

# 服务器上 LANG=C 时 stdio 可能是 ASCII 编码，中文日志会变成 \uXXXX 转义；能改就改成 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # Python 3.7+
    except Exception:
        pass

CHUNK = 1 << 16
ALL_TOKEN = "__ALL__"
DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
# 默认只要原图（jpeg/png）。带上 image/avif 会让 Cloudflare(imgix) 转码成 avif，体积更小但不是原图。
DEFAULT_ACCEPT = "image/*,*/*;q=0.8"

# 认为“已经下载完成”的图片后缀
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".jfif", ".png", ".webp", ".gif", ".bmp",
    ".tif", ".tiff", ".avif", ".heic", ".heif", ".ico", ".svg",
}

# Content-Type -> 文件后缀
CT_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/pjpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/x-ms-bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/avif": ".avif",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/svg+xml": ".svg",
    "image/x-icon": ".ico",
    "image/vnd.microsoft.icon": ".ico",
}
# 这些 Content-Type 也当图片处理
CT_OK_OTHER = {"application/octet-stream", "binary/octet-stream", ""}

_thread_local = threading.local()


class PermanentError(Exception):
    """重试也没用的错误（404 / 403 / 返回的不是图片等）。"""


class RetryableError(Exception):
    """可以重试的错误，可携带服务端要求的等待时间。"""

    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


def fmt_time(seconds: float) -> str:
    if seconds is None or seconds == float("inf") or seconds != seconds or seconds < 0:
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _bar_chars(stream) -> tuple[str, str]:
    """终端编码装不下 █░ 时退回 ASCII，避免在奇怪的环境里报 UnicodeEncodeError。"""
    enc = getattr(stream, "encoding", None) or "utf-8"
    try:
        "█░".encode(enc)
        return "█", "░"
    except Exception:
        return "#", "-"


def relpath_posix(path: Path, start: Path) -> str:
    """生成相对路径并统一用 / 分隔，例如 ../datas/0001/0001.jpg"""
    try:
        rel = os.path.relpath(str(path), str(start))
    except ValueError:  # Windows 跨盘符
        rel = str(path)
    return Path(rel).as_posix()


def get_session():
    """每个线程一个 requests.Session（连接复用，但 Session 不是线程安全的）。"""
    if requests is None:
        return None
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=16, max_retries=0)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _thread_local.session = session
    return session


# --------------------------------------------------------------------------- #
# 进度条
# --------------------------------------------------------------------------- #
class Progress:
    """带速度与 ETA 的进度条；非 TTY 时按 5% 步长打日志。"""

    def __init__(self, total: int, prefix: str, stream=None, min_interval: float = 0.15,
                 bar_width: int = 28):
        self.total = max(int(total), 1)
        self.prefix = prefix
        self.stream = stream if stream is not None else sys.stderr
        self.min_interval = min_interval
        self.bar_width = bar_width
        try:
            self.is_tty = bool(self.stream.isatty())
        except Exception:
            self.is_tty = False
        self.bar_full, self.bar_empty = _bar_chars(self.stream)
        self.done = self.ok = self.skip = self.fail = 0
        self.nbytes = 0
        self.start = time.time()
        self._lock = threading.Lock()
        self._last_render = 0.0
        self._last_pct_line = -100
        self._printed = False

    def update(self, ok: int = 0, skip: int = 0, fail: int = 0, nbytes: int = 0) -> None:
        with self._lock:
            self.ok += ok
            self.skip += skip
            self.fail += fail
            self.done += ok + skip + fail
            self.nbytes += nbytes
        self.render()

    def _build_line(self, now: float) -> tuple[str, int]:
        frac = self.done / self.total
        elapsed = max(now - self.start, 1e-9)
        rate = self.ok / elapsed
        remaining = max(self.total - self.done, 0)
        eta = (remaining / rate) if rate > 0 else None
        bps = self.nbytes / elapsed
        filled = int(frac * self.bar_width)
        bar = self.bar_full * filled + self.bar_empty * (self.bar_width - filled)
        line = (
            f"{self.prefix} {frac * 100:5.1f}% |{bar}| {self.done}/{self.total} "
            f"[{fmt_time(elapsed)}<{fmt_time(eta)}, {rate:5.1f} img/s, {human_bytes(bps)}/s] "
            f"ok={self.ok} skip={self.skip} fail={self.fail}"
        )
        return line, int(frac * 100)

    def render(self, force: bool = False) -> None:
        now = time.time()
        with self._lock:
            if not force and now - self._last_render < self.min_interval:
                return
            self._last_render = now
            line, pct = self._build_line(now)
        if self.is_tty:
            self.stream.write("\r\033[K" + line)
            self.stream.flush()
            self._printed = True
        elif force or (pct - self._last_pct_line >= 5 and self.done < self.total):
            self._last_pct_line = pct
            self.stream.write(line + "\n")
            self.stream.flush()

    def close(self) -> None:
        self.render(force=True)
        if self.is_tty and self._printed:
            self.stream.write("\n")
            self.stream.flush()
            self._printed = False


# --------------------------------------------------------------------------- #
# 失败日志
# --------------------------------------------------------------------------- #
class FailLog:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, num: int, fid: str, url: str, err: str) -> None:
        if self.path is None:
            return
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"{num:04d}\t{fid}\t{url}\t{err}\n")


# --------------------------------------------------------------------------- #
# 下载核心
# --------------------------------------------------------------------------- #
def guess_ext(url: str, content_type: str) -> str:
    """优先按 Content-Type 决定后缀，其次看 URL，最后退回 .jpg。"""
    ext = CT_EXT.get(content_type)
    if ext:
        return ext
    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    if suffix in IMAGE_EXTS:
        return suffix
    return ".jpg"


def _fetch_to_part(url: str, part: Path, timeout: float, headers: dict) -> tuple[int, str]:
    """把 url 下载到 part 临时文件（支持断点续传），返回 (字节数, content_type)。"""
    resume = part.stat().st_size if part.exists() else 0
    hdrs = dict(headers)
    if resume:
        hdrs["Range"] = f"bytes={resume}-"

    if requests is not None:
        session = get_session()
        try:
            r = session.get(url, headers=hdrs, timeout=timeout, stream=True)
        except Exception as e:  # 连接类错误 -> 可重试
            raise RetryableError(f"连接失败: {e}") from e
        try:
            code = r.status_code
            if resume and code == 200:
                resume = 0  # 服务端不支持 Range，从头来
            elif code not in (200, 206):
                if code in (429, 500, 502, 503, 504):
                    ra = r.headers.get("Retry-After")
                    try:
                        ra = float(ra) if ra else None
                    except ValueError:
                        ra = None
                    raise RetryableError(f"HTTP {code}", retry_after=ra)
                raise PermanentError(f"HTTP {code}")
            ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype not in CT_EXT and ctype not in CT_OK_OTHER:
                raise PermanentError(f"返回的不是图片 (Content-Type={ctype or '未知'})")
            mode = "ab" if (resume and code == 206) else "wb"
            n = resume if mode == "ab" else 0
            try:
                with open(part, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            f.write(chunk)
                            n += len(chunk)
            except Exception as e:
                raise RetryableError(f"写入失败: {e}") from e
            if mode == "ab" and not r.headers.get("Content-Encoding"):
                clen = r.headers.get("Content-Length")
                if clen and clen.isdigit() and n != resume + int(clen):
                    raise RetryableError(f"下载不完整 {n}/{resume + int(clen)}")
            return n, ctype
        finally:
            r.close()

    # ---- urllib 兜底 ----
    try:
        with urlopen(Request(url, headers=hdrs), timeout=timeout) as r:
            code = r.getcode()
            if resume and code == 200:
                resume = 0
            elif code not in (200, 206):
                raise PermanentError(f"HTTP {code}")
            ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype not in CT_EXT and ctype not in CT_OK_OTHER:
                raise PermanentError(f"返回的不是图片 (Content-Type={ctype or '未知'})")
            mode = "ab" if (resume and code == 206) else "wb"
            n = resume if mode == "ab" else 0
            with open(part, mode) as f:
                while True:
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    n += len(chunk)
            return n, ctype
    except PermanentError:
        raise
    except OSError as e:
        raise RetryableError(f"连接失败: {e}") from e
    except Exception as e:  # HTTPError 等
        code = getattr(e, "code", None)
        if code in (429, 500, 502, 503, 504):
            raise RetryableError(f"HTTP {code}") from e
        raise PermanentError(f"{type(e).__name__}: {e}") from e


def download_image(url: str, out_dir: Path, stem: str, timeout: float, retries: int,
                   headers: dict, max_backoff: float) -> tuple[str, Path | None, int, str | None]:
    """下载单张图片。返回 (状态, 最终路径, 字节数, 错误信息)，状态 ∈ {ok, skip, fail}。"""
    part = out_dir / f".{stem}.part"
    last_err = "未知错误"
    for attempt in range(1, retries + 1):
        try:
            n, ctype = _fetch_to_part(url, part, timeout, headers)
            if n <= 0:
                raise RetryableError("下载到 0 字节")
            final = out_dir / f"{stem}{guess_ext(url, ctype)}"
            os.replace(part, final)  # 原子改名，避免半成品被当成完成品
            return "ok", final, n, None
        except PermanentError as e:
            return "fail", None, 0, str(e)
        except RetryableError as e:
            last_err = str(e)
            if attempt >= retries:
                break
            wait = min(max_backoff, 2 ** (attempt - 1)) + random.uniform(0, 0.5)
            if e.retry_after:
                wait = max(wait, min(e.retry_after, max_backoff))
            time.sleep(wait)
        except Exception as e:  # 兜底，同样重试
            last_err = f"{type(e).__name__}: {e}"
            if attempt >= retries:
                break
            time.sleep(min(max_backoff, 2 ** (attempt - 1)) + random.uniform(0, 0.5))
    return "fail", None, 0, last_err


# --------------------------------------------------------------------------- #
# json 读写
# --------------------------------------------------------------------------- #
def find_existing(out_dir: Path, fid: str, image_path_field, json_dir: Path) -> Path | None:
    """判断这张图是否已经下载过（优先看 json 里记录的 image_path，其次扫目录）。"""
    if isinstance(image_path_field, str) and image_path_field:
        try:
            p = Path(os.path.normpath(str(json_dir / image_path_field)))
            if p.is_file() and p.stat().st_size > 0:
                return p
        except OSError:
            pass
    try:
        for cand in out_dir.glob(f"{fid}.*"):
            if cand.is_file() and cand.suffix.lower() in IMAGE_EXTS and cand.stat().st_size > 0:
                return cand
    except OSError:
        pass
    return None


def save_json(json_path: Path, records: list, compact: bool = False) -> None:
    tmp = json_path.with_name(json_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        if compact:
            json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, json_path)


# --------------------------------------------------------------------------- #
# 单个 json 的处理
# --------------------------------------------------------------------------- #
def process_one(num: int, args, faillog: FailLog) -> dict | None:
    t0 = time.time()
    json_path = Path(args.json_dir) / f"{num:04d}.json"
    if not json_path.is_file():
        print(f"[跳过] 找不到 {json_path}", file=sys.stderr)
        return None

    try:
        records = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[错误] 解析 {json_path} 失败: {e}", file=sys.stderr)
        return None
    if not isinstance(records, list) or not records:
        print(f"[错误] {json_path} 不是非空列表，已跳过", file=sys.stderr)
        return None

    total = len(records)
    width = len(str(total))  # 1000 -> 4 位, 230 -> 3 位
    out_dir = Path(args.datas_dir) / f"{num:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 给每一条记录补 id（不论是否下载）
    for i, rec in enumerate(records, 1):
        if isinstance(rec, dict):
            rec["id"] = f"{i:0{width}d}"
        else:
            records[i - 1] = {"id": f"{i:0{width}d}", "raw": rec}

    take = total if args.max_per_json <= 0 else min(args.max_per_json, total)

    # 2) 组任务列表：已下载的跳过，缺 url 的记为失败
    tasks: list[tuple[int, str, str]] = []
    skipped = 0
    bad = 0
    for i in range(take):
        rec = records[i]
        fid = rec["id"]
        url = rec.get("image_url")
        if not isinstance(url, str) or not url.strip():
            bad += 1
            faillog.add(num, fid, "", "记录里没有 image_url")
            continue
        url = url.strip()
        found = find_existing(out_dir, fid, rec.get("image_path"), json_path.parent)
        if found is not None:
            rec["image_path"] = relpath_posix(found, json_path.parent)
            skipped += 1
            continue
        tasks.append((i, fid, url))

    headers = {"User-Agent": args.user_agent, "Accept": args.accept}

    prog = Progress(take, json_path.name)
    print(f"[开始] {json_path.name}: 共 {total} 条，本次处理 {take} 条"
          f"（待下载 {len(tasks)}，已存在 {skipped}，无链接 {bad}），线程 {args.workers}",
          file=sys.stderr)
    prog.update(skip=skipped, fail=bad)

    ok = 0
    fail = 0
    nbytes = 0
    pending_since_flush = 0
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {}
    interrupted = False
    try:
        for idx, fid, url in tasks:
            fut = executor.submit(
                download_image, url, out_dir, fid, args.timeout,
                args.retries, headers, args.max_backoff,
            )
            futures[fut] = (idx, fid, url)
        for fut in as_completed(futures):
            idx, fid, url = futures[fut]
            try:
                status, path, n, err = fut.result()
            except Exception as e:  # 理论上不会有
                status, path, n, err = "fail", None, 0, f"{type(e).__name__}: {e}"
            if status == "ok" and path is not None:
                records[idx]["image_path"] = relpath_posix(path, json_path.parent)
                ok += 1
                nbytes += n
                pending_since_flush += 1
                prog.update(ok=1, nbytes=n)
            else:
                fail += 1
                faillog.add(num, fid, url, err or "未知错误")
                prog.update(fail=1)
            # 定期落盘，避免中途崩溃丢掉 image_path
            if not args.no_json_update and args.flush_every > 0 and pending_since_flush >= args.flush_every:
                pending_since_flush = 0
                try:
                    save_json(json_path, records, compact=args.compact_json)
                except Exception as e:
                    print(f"[警告] 中途写回 {json_path} 失败: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        interrupted = True
        for f in futures:
            f.cancel()
        try:
            executor.shutdown(wait=False, cancel_futures=True)  # Python 3.9+
        except TypeError:
            executor.shutdown(wait=False)                        # Python 3.8 没有这个参数
        raise
    finally:
        if not interrupted:
            executor.shutdown(wait=True)
        prog.close()

    # 3) 写回 json
    saved = False
    if not args.no_json_update:
        try:
            save_json(json_path, records, compact=args.compact_json)
            saved = True
        except Exception as e:
            print(f"[错误] 写回 {json_path} 失败: {e}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"[完成] {json_path.name}: 下载 {ok}，跳过 {skipped}，失败 {fail}，"
          f"用时 {fmt_time(elapsed)}，{human_bytes(nbytes)}"
          f"{'' if saved else '（未写回 json）'}", file=sys.stderr)

    return {"num": num, "total": total, "take": take, "ok": ok, "skip": skipped,
            "fail": fail, "bytes": nbytes, "seconds": elapsed}


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_images.py",
        description="MultiAspect-4K-1M 图片下载脚本（串行处理 json + 多线程下载 + 断点续传 + 进度条）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python download_images.py 1 2 3 -m 100 -j 8\n"
            "  python download_images.py 1-20 --max-per-json 500 --workers 16\n"
            "  python download_images.py all -j 16\n"
            "  python download_images.py 1 5 9 --ids-file more_ids.txt -m 50\n"
        ),
    )
    p.add_argument("ids", nargs="*",
                   help="数字编号列表；支持空格/逗号分隔、区间（如 1-20）、以及 all")
    p.add_argument("-m", "--max-per-json", type=int, default=0,
                   help="每个 json 最多下载的图片数量；0 或负数表示全部（默认 0）")
    p.add_argument("-j", "--workers", type=int, default=8,
                   help="单个 json 内的并行下载线程数（默认 8）")
    p.add_argument("--json-dir", default="data_jsons", help="json 所在目录（默认 data_jsons）")
    p.add_argument("--datas-dir", default="datas", help="图片输出目录（默认 datas）")
    p.add_argument("--ids-file", default=None, help="从文件读取编号（每行/逗号分隔，可混用区间）")
    p.add_argument("--timeout", type=float, default=60.0, help="单次请求超时秒数（默认 60）")
    p.add_argument("--retries", type=int, default=3, help="单张图片最大尝试次数（默认 3）")
    p.add_argument("--max-backoff", type=float, default=60.0, help="重试最大等待秒数（默认 60）")
    p.add_argument("--flush-every", type=int, default=200,
                   help="每下载多少张就把 image_path 落盘一次；0 表示只在结束时写（默认 200）")
    p.add_argument("--no-json-update", action="store_true",
                   help="只下载图片，不把 id/image_path 写回 json")
    p.add_argument("--compact-json", action="store_true",
                   help="回写 json 时用紧凑格式（体积更小，默认保持原来的缩进格式）")
    p.add_argument("--failed-log", default=None,
                   help="失败记录日志路径（默认 <datas>/failed_downloads.tsv）")
    p.add_argument("--user-agent", default=DEFAULT_UA, help="请求使用的 User-Agent")
    p.add_argument("--accept", default=DEFAULT_ACCEPT,
                   help="Accept 请求头。默认取原图（jpeg/png）；"
                        "改成 'image/avif,image/webp,image/*,*/*;q=0.8' 可让 Pexels/Cloudflare "
                        "返回更小的 avif/webp（后缀会随之变化）")
    return p


def resolve_ids(args, parser: argparse.ArgumentParser) -> list[int]:
    tokens: list[str] = list(args.ids)
    if args.ids_file:
        try:
            tokens.append(Path(args.ids_file).read_text(encoding="utf-8"))
        except OSError as e:
            parser.error(f"读取 --ids-file 失败: {e}")
    if not tokens:
        parser.error("请提供编号列表，例如：1 2 3 / 1-20 / all，或用 --ids-file 指定文件")

    picked: list[int] = []
    want_all = False
    for token in tokens:
        for part in re.split(r"[,\s]+", token.strip()):
            if not part:
                continue
            low = part.lower()
            if low in ("all", "*"):
                want_all = True
                continue
            m = re.fullmatch(r"(\d+)\s*[-~]\s*(\d+)", part)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                step = 1 if b >= a else -1
                picked.extend(range(a, b + step, step))
            elif part.isdigit():
                picked.append(int(part))
            else:
                parser.error(f"无法识别的编号: {part!r}（支持 1 / 1-20 / all）")

    json_dir = Path(args.json_dir)
    if want_all:
        available = sorted(int(q.stem) for q in json_dir.glob("*.json") if q.stem.isdigit())
        if not available:
            parser.error(f"{json_dir} 下没有找到 NNNN.json")
        return available

    seen: set[int] = set()
    out: list[int] = []
    for n in picked:
        if n < 1:
            print(f"[警告] 忽略非法编号 {n}", file=sys.stderr)
            continue
        if n not in seen:
            seen.add(n)
            out.append(n)
    if not out:
        parser.error("没有有效的编号")
    return out


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers 必须 >= 1")
    if args.retries < 1:
        parser.error("--retries 必须 >= 1")

    # data_jsons 不存在时兜底到当前目录，方便在数据集根目录直接跑
    if not Path(args.json_dir).is_dir() and args.json_dir == "data_jsons" and Path("0001.json").is_file():
        print("[提示] 未找到 data_jsons/，改用当前目录作为 json 目录", file=sys.stderr)
        args.json_dir = "."

    if not Path(args.json_dir).is_dir():
        parser.error(f"json 目录不存在: {args.json_dir}")

    ids = resolve_ids(args, parser)
    Path(args.datas_dir).mkdir(parents=True, exist_ok=True)
    faillog = FailLog(args.failed_log or (Path(args.datas_dir) / "failed_downloads.tsv"))

    max_desc = "全部" if args.max_per_json <= 0 else str(args.max_per_json)
    print(f"[配置] json 目录: {Path(args.json_dir).resolve()}", file=sys.stderr)
    print(f"[配置] 图片目录: {Path(args.datas_dir).resolve()}", file=sys.stderr)
    print(f"[配置] 编号数量: {len(ids)}（{ids[0]}..{ids[-1]}），每个 json 最多下载 {max_desc} 张，"
          f"线程 {args.workers}，下载引擎: {'requests' if requests else 'urllib'}", file=sys.stderr)
    print(f"[配置] 失败日志: {faillog.path}", file=sys.stderr)

    overall = {"ok": 0, "skip": 0, "fail": 0, "bytes": 0, "files": 0, "seconds": 0.0}
    t_start = time.time()
    for k, num in enumerate(ids, 1):
        print(f"\n=== [{k}/{len(ids)}] 处理 {num:04d}.json ===", file=sys.stderr)
        try:
            stat = process_one(num, args, faillog)
        except KeyboardInterrupt:
            print("\n[中断] 用户中止；已下载的图片会保留，重跑时会自动跳过。", file=sys.stderr)
            return 130
        if stat is None:
            continue
        overall["ok"] += stat["ok"]
        overall["skip"] += stat["skip"]
        overall["fail"] += stat["fail"]
        overall["bytes"] += stat["bytes"]
        overall["files"] += 1
        overall["seconds"] += stat["seconds"]

    elapsed = time.time() - t_start
    print("\n" + "=" * 72, file=sys.stderr)
    print(f"[全部结束] 处理 json {overall['files']}/{len(ids)} 个，"
          f"下载 {overall['ok']} 张（{human_bytes(overall['bytes'])}），"
          f"跳过 {overall['skip']} 张，失败 {overall['fail']} 张，总用时 {fmt_time(elapsed)}",
          file=sys.stderr)
    if overall["fail"]:
        print(f"[提示] 失败明细见 {faillog.path}，直接重跑同一条命令即可重试（已成功的会跳过）。",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())