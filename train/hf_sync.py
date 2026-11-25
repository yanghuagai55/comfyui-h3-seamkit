# SPDX-License-Identifier: GPL-3.0-or-later
"""私有 HF Repo 同步：教师包上传 / 训练侧下载 / 训练中异步回传 ckpt+metrics。

设计约束（Colab 单进程、无后台服务）：
  * 上传走后台守护线程 + 队列，训练主循环零阻塞；退出前 `close(wait=True)`。
  * 每个文件独立重试（默认 5 次，指数退避），失败只写日志不抛，训练不中断。
  * token 只从文件读（默认 `/content/work/hftoken.txt`），不打印内容。

命令行（教师号打包后一次推完）：
  python train/hf_sync.py upload --repo <user>/<repo> --src /content/work/teacher_cache \
      --dst teacher_cache_v1 --pattern "*.pt" --pattern "_manifest.json"
  python train/hf_sync.py upload --repo <user>/<repo> --src /content/work/teacher_cache \
      --dst teacher_cache_v1/teacher_cache_v1.tar.zst --single

命令行（训练号拉包）：
  python train/hf_sync.py download --repo <user>/<repo> --dst /content/work/teacher_cache \
      --prefix teacher_cache_v1 --pattern "*.pt" --pattern "_manifest.json"
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import queue
import sys
import threading
import time
from pathlib import Path

DEFAULT_TOKEN_FILE = "/content/work/hftoken.txt"


def read_token(path: str = DEFAULT_TOKEN_FILE) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"HF token 文件不存在：{p}（先写入 tokens，注意不要提交进 git）")
    token = p.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"HF token 文件为空：{p}")
    return token


def _api(repo: str, repo_type: str, token: str):
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type=repo_type, private=True, exist_ok=True)
    return api


class AsyncUploader:
    """后台线程串行上传（HF 单文件接口本身并发受限，串行最稳）。"""

    def __init__(self, repo: str, repo_type: str = "model", token_file: str = DEFAULT_TOKEN_FILE,
                 retries: int = 5, log_path: str | None = None):
        self.repo = repo
        self.repo_type = repo_type
        self.token_file = token_file
        self.retries = int(retries)
        self.log_path = Path(log_path) if log_path else None
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._api = None
        self._thread = threading.Thread(target=self._run, name="hf-sync", daemon=True)
        self._thread.start()

    # -- public ---------------------------------------------------------
    def submit(self, local: str | Path, repo_path: str) -> None:
        self._q.put((str(local), str(repo_path)))

    def close(self, wait: bool = True, timeout: float = 900.0) -> None:
        self._q.put(None)
        if wait:
            self._thread.join(timeout=timeout)

    # -- internals ------------------------------------------------------
    def _log(self, msg: str) -> None:
        line = f"[hf-sync {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except Exception:
                pass

    def _ensure_api(self):
        if self._api is None:
            self._api = _api(self.repo, self.repo_type, read_token(self.token_file))
            self._log(f"repo ready: {self.repo} ({self.repo_type})")
        return self._api

    def _upload_one(self, local: str, repo_path: str) -> bool:
        path = Path(local)
        if not path.is_file():
            self._log(f"skip missing {path}")
            return False
        for attempt in range(1, self.retries + 1):
            try:
                api = self._ensure_api()
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=repo_path,
                    repo_id=self.repo,
                    repo_type=self.repo_type,
                    commit_message=f"sync {path.name} @ {time.strftime('%Y-%m-%d %H:%M:%S')}",
                )
                self._log(f"ok {path.name} -> {repo_path} ({path.stat().st_size / 1e6:.1f} MB)")
                return True
            except Exception as exc:
                wait = min(60.0, 2.0 ** attempt)
                self._log(f"retry {attempt}/{self.retries} {path.name}: {exc}")
                if attempt < self.retries:
                    time.sleep(wait)
        self._log(f"FAIL {path.name} -> {repo_path}")
        return False

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._q.get()
            if item is None:
                break
            try:
                self._upload_one(*item)
            except Exception as exc:  # 后台线程绝不能把主进程带走
                self._log(f"worker error: {exc}")


def cmd_upload(args) -> int:
    src = Path(args.src)
    token = read_token(args.token_file)
    api = _api(args.repo, args.repo_type, token)
    if args.single:
        api.upload_file(path_or_fileobj=str(src), path_in_repo=args.dst, repo_id=args.repo,
                        repo_type=args.repo_type, commit_message=f"upload {src.name}")
        print(f"[hf-sync] uploaded {src} -> {args.dst}")
        return 0

    patterns = args.pattern or ["*"]
    files = sorted(p for p in src.rglob("*") if p.is_file() and any(fnmatch.fnmatch(p.name, pat) for pat in patterns))
    print(f"[hf-sync] {len(files)} files -> {args.repo}:{args.dst}", flush=True)
    failed = 0
    for i, p in enumerate(files, 1):
        rel = p.relative_to(src).as_posix()
        dest = f"{args.dst.strip('/')}/{rel}" if args.dst.strip("/") else rel
        ok = False
        for attempt in range(1, args.retries + 1):
            try:
                api.upload_file(path_or_fileobj=str(p), path_in_repo=dest, repo_id=args.repo,
                                repo_type=args.repo_type, commit_message=f"upload {rel}")
                ok = True
                break
            except Exception as exc:
                print(f"  retry {attempt}/{args.retries} {rel}: {exc}", flush=True)
                time.sleep(min(60.0, 2.0 ** attempt))
        print(f"[{i}/{len(files)}] {'ok' if ok else 'FAIL'} {rel} ({p.stat().st_size / 1e6:.1f} MB)", flush=True)
        failed += 0 if ok else 1
    print(f"[hf-sync] done failed={failed}")
    return 0 if failed == 0 else 1


def cmd_download(args) -> int:
    from huggingface_hub import snapshot_download

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    allow = args.pattern or None
    path = snapshot_download(
        repo_id=args.repo,
        repo_type=args.repo_type,
        local_dir=str(dst),
        token=read_token(args.token_file),
        allow_patterns=[f"{args.prefix.strip('/')}/{p}" for p in allow] if allow else None,
        resume_download=True,
    )
    print(f"[hf-sync] downloaded -> {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload")
    up.add_argument("--repo", required=True)
    up.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    up.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    up.add_argument("--src", required=True)
    up.add_argument("--dst", default="")
    up.add_argument("--pattern", action="append", default=[])
    up.add_argument("--single", action="store_true", help="--src 是单个文件，直接传到 --dst")
    up.add_argument("--retries", type=int, default=5)

    dn = sub.add_parser("download")
    dn.add_argument("--repo", required=True)
    dn.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    dn.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    dn.add_argument("--dst", required=True)
    dn.add_argument("--prefix", default="")
    dn.add_argument("--pattern", action="append", default=[])

    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    if args.cmd == "upload":
        return cmd_upload(args)
    return cmd_download(args)


if __name__ == "__main__":
    sys.exit(main())
