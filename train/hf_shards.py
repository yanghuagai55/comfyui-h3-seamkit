# SPDX-License-Identifier: GPL-3.0-or-later
"""私有 HF Repo 上的教师分片仓库：原子增量续传。

布局（全部在一个私有 repo 里，前缀默认 `teacher_cache_v1`）：
  <prefix>/manifest.json                 # 唯一权威索引：shard 列表 + 每片 ids
  <prefix>/shards/shard_000.tar.zst      # 10 条已打包样本（<id>.pt）

语义
----
* 上传顺序 = 先 shard、后 manifest。manifest 是提交点：中途断电最多留下一个
  孤儿 shard（下次 push 会覆盖同名文件），不会出现「索引里有、文件没有」。
* 下载/续跑只信 manifest；本地已存在的 `<id>.pt` 不重复解压。
* manifest 每次更新前重新从云端拉一遍，避免多号并行时互相覆盖。
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import time
from pathlib import Path

DEFAULT_TOKEN_FILE = "/content/work/hftoken.txt"
DEFAULT_PREFIX = "teacher_cache_v1"


def read_token(path: str = DEFAULT_TOKEN_FILE) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"HF token 文件不存在：{p}")
    token = p.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"HF token 文件为空：{p}")
    return token


def _zstd():
    try:
        import zstandard  # type: ignore

        return zstandard
    except Exception:
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ShardStore:
    def __init__(self, repo: str, prefix: str = DEFAULT_PREFIX, token_file: str = DEFAULT_TOKEN_FILE,
                 repo_type: str = "model", verbose: bool = True):
        self.repo = repo
        self.prefix = prefix.strip("/")
        self.token_file = token_file
        self.repo_type = repo_type
        self.verbose = verbose
        self._api_obj = None

    # -- logging ---------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[shard {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    # -- hf --------------------------------------------------------------
    def _api(self):
        if self._api_obj is None:
            from huggingface_hub import HfApi

            api = HfApi(token=read_token(self.token_file))
            api.create_repo(repo_id=self.repo, repo_type=self.repo_type, private=True, exist_ok=True)
            self._api_obj = api
        return self._api_obj

    def _remote(self, name: str) -> str:
        return f"{self.prefix}/{name}"

    def fetch_manifest(self) -> dict:
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(self.repo, self._remote("manifest.json"), repo_type=self.repo_type,
                                   token=read_token(self.token_file))
            man = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            man = {}
        man.setdefault("version", 1)
        man.setdefault("shards", [])
        man["count"] = sum(len(s.get("ids") or []) for s in man["shards"])
        return man

    def done_ids(self) -> set[str]:
        return {i for s in self.fetch_manifest()["shards"] for i in (s.get("ids") or [])}

    def push_shard(self, tar_path: Path, ids: list[str]) -> dict:
        api = self._api()
        name = tar_path.name
        size = tar_path.stat().st_size
        digest = _sha256(tar_path)
        api.upload_file(path_or_fileobj=str(tar_path), path_in_repo=self._remote(f"shards/{name}"),
                        repo_id=self.repo, repo_type=self.repo_type,
                        commit_message=f"shard {name} ({len(ids)} samples)")
        man = self.fetch_manifest()
        man["shards"] = [s for s in man["shards"] if s.get("name") != name]
        man["shards"].sort(key=lambda s: s.get("name", ""))
        man["shards"].append({"name": name, "ids": ids, "bytes": size, "sha256": digest})
        man["shards"].sort(key=lambda s: s.get("name", ""))
        man["count"] = sum(len(s.get("ids") or []) for s in man["shards"])
        man["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = tar_path.with_suffix(tar_path.suffix + ".manifest.tmp")
        tmp.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
        api.upload_file(path_or_fileobj=str(tmp), path_in_repo=self._remote("manifest.json"),
                        repo_id=self.repo, repo_type=self.repo_type,
                        commit_message=f"manifest: +{name} total={man['count']}")
        tmp.unlink(missing_ok=True)
        self._log(f"pushed {name} ({size / 1e6:.1f} MB, {len(ids)} samples) total={man['count']}")
        return man

    def pull_shard(self, shard: dict, work_dir: Path, cache_dir: Path, clean: bool = True) -> int:
        from huggingface_hub import hf_hub_download

        name = shard["name"]
        local = Path(work_dir) / self.prefix / "shards" / name
        if not local.is_file():
            got = hf_hub_download(self.repo, self._remote(f"shards/{name}"), repo_type=self.repo_type,
                                  local_dir=str(work_dir), token=read_token(self.token_file))
            local = Path(got)
        cache_dir.mkdir(parents=True, exist_ok=True)
        added = 0
        zstd = _zstd()
        if name.endswith(".zst") and zstd is not None:
            raw = local.open("rb")
            reader = zstd.ZstdDecompressor().stream_reader(raw)
            tf = tarfile.open(fileobj=reader, mode="r|")
            closer = (tf, reader, raw)
        else:
            tf = tarfile.open(local, "r:*")
            closer = (tf,)
        try:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".pt"):
                    continue
                target = cache_dir / Path(member.name).name
                if target.exists():
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                tmp = target.with_suffix(".pt.tmp")
                with tmp.open("wb") as out:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        out.write(chunk)
                tmp.replace(target)
                added += 1
        finally:
            for c in closer:
                try:
                    c.close()
                except Exception:
                    pass
        if clean:
            local.unlink(missing_ok=True)
        self._log(f"pulled {name}: +{added} samples")
        return added

    def pull_all(self, cache_dir, work_dir) -> int:
        man = self.fetch_manifest()
        cache_dir = Path(cache_dir)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for shard in man["shards"]:
            missing = [i for i in shard.get("ids") or [] if not (cache_dir / f"{i}.pt").is_file()]
            if not missing:
                continue
            total += self.pull_shard(shard, work_dir, cache_dir)
        self._log(f"pull_all: manifest={len(man['shards'])} shards, {man['count']} samples, +{total} local")
        return total


def make_shard(cache_dir: Path, ids: list[str], name: str) -> Path | None:
    """把 cache 里这 10 条打成 tar.zst（先写 .tmp 再原子改名）。"""
    files = [Path(cache_dir) / f"{i}.pt" for i in ids]
    files = [p for p in files if p.is_file()]
    if not files:
        return None
    out = Path(cache_dir) / name
    zstd = _zstd()
    if zstd is None and out.name.endswith(".zst"):
        out = out.with_name(out.name[:-4] + ".gz")
    tmp = out.with_suffix(out.suffix + ".tmp")
    if zstd is not None:
        cctx = zstd.ZstdCompressor(level=10, threads=4)
        with tmp.open("wb") as raw_fh, cctx.stream_writer(raw_fh, closefd=False) as zfh:
            with tarfile.open(fileobj=zfh, mode="w|") as tf:
                for p in files:
                    info = tarfile.TarInfo(p.name)
                    info.size = p.stat().st_size
                    info.mtime = int(p.stat().st_mtime)
                    with p.open("rb") as fh:
                        tf.addfile(info, fh)
    else:
        with tarfile.open(tmp, "w:gz") as tf:
            for p in files:
                tf.add(p, arcname=p.name)
    tmp.replace(out)
    return out
