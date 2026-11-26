# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""Zero-RAM-OOM streaming RelayVSR Flash decoder + H.264/AAC MP4 writer.

Stock ComfyUI `VSRDecode` (`comfy/sd.py:1393`) pre-allocates the entire decoded
3K/2.7K video as one contiguous CPU FP32 tensor (`[1, 3, F, H*2, W*2]`):
  - 10s @ 3136x1728 (243f): 14.72 GiB contiguous RAM (on top of 3.68 GiB `lq_images`)
  - 15s @ 3136x1728 (362f): 21.93 GiB contiguous RAM (on top of 5.49 GiB `lq_images`)

On 32 GB RAM machines right after MiniMax H3 + Qwen3VL + Sol-VAE + RelayVSR UNet,
this single allocation exhausts Windows commit space and hard-crashes the process.

`MiniMaxH3StreamingVSRSave` runs the exact same `RelayFlash` per-frame math
(`model_vsr.py:345-372`), converts each decoded frame immediately to RGB24 `uint8`,
and streams it directly to `ffmpeg` (H.264 `yuv420p`) while muxing the original
ComfyUI `AUDIO` track — peak extra CPU RAM is < 0.2 GiB regardless of clip length.
"""

from __future__ import annotations

import gc
import os
import re
import subprocess
import tempfile
import time
import wave

import numpy as np
import torch
import folder_paths
import comfy.model_management as mm
import comfy.utils
from comfy.ldm.wan.model_vsr import relay_keyframes, relay_latent_size
from comfy_api.latest import io

CATEGORY = "MiniMax H3 Hard Cut"


def _resolve_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError("ffmpeg not found (neither imageio_ffmpeg nor system PATH)")


def _next_output_paths(output_dir: str, filename_prefix: str, has_audio: bool) -> tuple[str, str]:
    """Allocate next sequential `<prefix>_00001(-audio).mp4` and `<prefix>_00001.png` without overwriting."""
    full_dir, base_name, _, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, output_dir, 0, 0
    )
    os.makedirs(full_dir, exist_ok=True)
    pat = re.compile(rf"^{re.escape(base_name)}_(\d+)(?:-audio)?\.(?:mp4|png)$", re.IGNORECASE)
    max_n = 0
    for fn in os.listdir(full_dir):
        m = pat.match(fn)
        if m:
            max_n = max(max_n, int(m.group(1)))
    idx = max_n + 1
    suffix = "-audio" if has_audio else ""
    mp4_path = os.path.join(full_dir, f"{base_name}_{idx:05d}{suffix}.mp4")
    png_path = os.path.join(full_dir, f"{base_name}_{idx:05d}.png")
    return mp4_path, png_path


def _write_audio_wav(audio: dict | None, wav_path: str) -> bool:
    """Serialize a ComfyUI AUDIO dict `{'waveform': [B,C,T], 'sample_rate': int}` to 16-bit PCM WAV."""
    if not isinstance(audio, dict):
        return False
    wf = audio.get("waveform")
    sr = int(audio.get("sample_rate", 44100))
    if wf is None or not torch.is_tensor(wf) or wf.numel() == 0:
        return False
    w = wf[0] if wf.ndim == 3 else wf
    if w.ndim == 1:
        w = w.unsqueeze(0)
    # [C, T] -> [T, C] int16
    arr = w.detach().float().cpu().clamp(-1.0, 1.0).transpose(0, 1).numpy()
    pcm = (arr * 32767.0).round().astype(np.int16)
    with wave.open(wav_path, "wb") as wf_out:
        wf_out.setnchannels(int(pcm.shape[1]))
        wf_out.setsampwidth(2)
        wf_out.setframerate(sr)
        wf_out.writeframes(pcm.tobytes())
    return True


def stream_relay_vsr_to_mp4(
    vae,
    samples: dict | torch.Tensor,
    images: torch.Tensor,
    *,
    audio: dict | None = None,
    detail: float = 0.5,
    frame_rate: float = 24.0,
    crf: int = 15,
    preset: str = "medium",
    filename_prefix: str = "MiniMaxH3/H3_13MP_VSR3K",
    save_first_frame_png: bool = True,
) -> tuple[str, str]:
    """Stream-decode `RelayFlash` latents directly to H.264 MP4 (< 0.2 GiB extra CPU RAM)."""
    lat = samples["samples"] if isinstance(samples, dict) else samples
    fsm = getattr(vae, "first_stage_model", None)
    if fsm is None or not hasattr(fsm, "decode_frame") or not hasattr(fsm, "prepare_condition"):
        raise ValueError("MiniMaxH3StreamingVSRSave requires a RelayVSR Flash VAE (e.g. relayvsr_flash_vae_x2_fp32).")

    f_cnt, h_in, w_in = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    up = int(fsm.upscale_factor)
    out_h, out_w = h_in * up, w_in * up
    kf = relay_keyframes(f_cnt)
    expected = (len(kf), *relay_latent_size(h_in, w_in, fsm.latent_stride))
    if tuple(lat.shape[2:]) != expected:
        raise ValueError(
            f"RelayVSR latent shape {tuple(lat.shape[2:])} != expected {expected} for "
            f"{f_cnt}f {w_in}x{h_in} x{up} decoder."
        )

    # Free any lingering diffusion models (MiniMaxH3 / RelayVSR UNet) before loading Flash VAE
    mm.unload_all_models()
    mm.soft_empty_cache()
    gc.collect()

    mem_req = vae.memory_used_decode(lat.shape, vae.vae_dtype)
    mm.load_models_gpu([vae.patcher], memory_required=mem_req, force_full_load=vae.disable_offload)

    ffmpeg_exe = _resolve_ffmpeg_exe()
    out_dir = folder_paths.get_output_directory()
    tmp_wav = None
    has_audio = False
    if audio is not None:
        fd, tmp_wav = tempfile.mkstemp(prefix="seamkit_vsr_", suffix=".wav")
        os.close(fd)
        has_audio = _write_audio_wav(audio, tmp_wav)
        if not has_audio:
            try:
                os.remove(tmp_wav)
            except OSError:
                pass
            tmp_wav = None

    mp4_path, png_path = _next_output_paths(out_dir, filename_prefix, has_audio)
    fd_log, log_path = tempfile.mkstemp(prefix="seamkit_ffmpeg_", suffix=".log")
    os.close(fd_log)

    cmd = [
        ffmpeg_exe, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}",
        "-r", f"{float(frame_rate):.6g}", "-i", "pipe:0",
    ]
    if has_audio and tmp_wav:
        cmd += ["-i", tmp_wav, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += [
        "-c:v", "libx264", "-crf", str(int(crf)), "-pix_fmt", "yuv420p",
        "-preset", str(preset), "-movflags", "+faststart",
        mp4_path,
    ]

    t0 = time.perf_counter()
    stock_gib = (f_cnt * out_h * out_w * 3 * 4) / (1024 ** 3)
    print(
        f"[SeamKit] Streaming VSR Decode: {f_cnt}f @ {w_in}×{h_in} → {out_w}×{out_h} (x{up}, detail={detail:.2f}) | "
        f"省去 {stock_gib:.2f} GiB 连续 FP32 内存分配 -> {os.path.basename(mp4_path)}",
        flush=True,
    )

    pbar = comfy.utils.ProgressBar(f_cnt)
    written = 0
    first_frame_saved = False
    dev = vae.device
    dtype = vae.vae_dtype

    with open(log_path, "wb") as log_f:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_f)
        try:
            with torch.no_grad():
                x_gpu = lat.to(device=dev, dtype=dtype)
                b, _, _, h_lat, w_lat = x_gpu.shape
                depth = len(fsm.backbone.blocks)
                freqs = fsm.grid_rope(0, h_lat, w_lat, dev)
                freqs_past = fsm.grid_rope(-1, h_lat, w_lat, dev)

                start_cond = fsm.prepare_condition(x_gpu[:, :, 0], freqs)
                for win_idx, (first, last) in enumerate(zip(kf, kf[1:] or kf)):
                    mm.throw_exception_if_processing_interrupted()
                    end_cond = fsm.prepare_condition(x_gpu[:, :, min(win_idx + 1, len(kf) - 1)], freqs)
                    condition = list(zip(start_cond, end_cond))
                    history = [None] * depth
                    lr_win = images[first : last + 1].to(device=dev)[..., :3].movedim(-1, 1) * 2.0 - 1.0
                    lr_win = torch.nn.functional.pad(
                        lr_win,
                        (0, w_lat * fsm.latent_stride - w_in, 0, h_lat * fsm.latent_stride - h_in),
                        mode="replicate",
                    )
                    for i, fidx in enumerate(range(first, last + 1)):
                        decoded, history = fsm.decode_frame(
                            lr_win[i : i + 1].expand(b, -1, -1, -1),
                            x_gpu.dtype, freqs, freqs_past, condition, history, float(detail),
                        )
                        if win_idx == 0 or fidx > first:
                            rgb = (
                                decoded[0, :, :out_h, :out_w]
                                .add_(1.0).div_(2.0).clamp_(0.0, 1.0)
                                .mul_(255.0).round_().byte()
                            )
                            frame_np = rgb.permute(1, 2, 0).contiguous().cpu().numpy()
                            if save_first_frame_png and not first_frame_saved:
                                try:
                                    from PIL import Image
                                    Image.fromarray(frame_np).save(png_path)
                                    first_frame_saved = True
                                except Exception:
                                    pass
                            assert proc.stdin is not None
                            proc.stdin.write(frame_np.tobytes())
                            written += 1
                            pbar.update(1)
                    start_cond = end_cond
                    del lr_win, condition, history
        except BaseException:
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.terminate()
            except Exception:
                pass
            raise
        finally:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            rc = proc.wait()

    stderr_txt = ""
    try:
        stderr_txt = open(log_path, "rb").read().decode("utf-8", errors="replace")
        os.remove(log_path)
    except OSError:
        pass
    if tmp_wav:
        try:
            os.remove(tmp_wav)
        except OSError:
            pass

    if rc != 0:
        raise RuntimeError(f"ffmpeg VSR encode failed (rc={rc}):\n{stderr_txt[-1500:]}")

    dt = time.perf_counter() - t0
    sz_mb = os.path.getsize(mp4_path) / (1024 ** 2)
    report = f"{mp4_path} ({written}f @ {out_w}×{out_h}, {sz_mb:.2f} MiB, {dt:.1f}s)"
    print(f"[SeamKit] Streaming VSR 完成: {report}", flush=True)
    return mp4_path, report


class MiniMaxH3StreamingVSRSave(io.ComfyNode):
    """Streaming RelayVSR Flash decoder + MP4 saver avoiding 15–22 GiB CPU RAM OOM on 32GB hosts."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3StreamingVSRSave",
            display_name="MiniMax H3 Streaming VSR Decode & Save (32GB Safe)",
            category=CATEGORY,
            is_output_node=True,
            description=(
                "流式 RelayVSR Flash 解码并直接封装 H.264 MP4（32GB 内存救星）：\n"
                "• 逐帧将 3K/2.7K 超分帧直通 ffmpeg 编码，彻底消除原版 VSRDecode 一次性分配 15~22 GiB FP32 内存导致的闪退/卡死；\n"
                "• 与原版 RelayFlash 逐位等价，支持直接复用 H3 原生立体声音轨。"
            ),
            inputs=[
                io.Vae.Input("vae"),
                io.Latent.Input("samples"),
                io.Image.Input("images"),
                io.Float.Input("detail", default=0.5, min=0.0, max=1.0, step=0.05),
                io.Float.Input("frame_rate", default=24.0, min=1.0, max=120.0, step=1.0),
                io.Int.Input("crf", default=15, min=0, max=51, step=1),
                io.String.Input("filename_prefix", default="MiniMaxH3/H3_13MP_VSR3K"),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="video_path"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        vae,
        samples,
        images,
        detail: float = 0.5,
        frame_rate: float = 24.0,
        crf: int = 15,
        filename_prefix: str = "MiniMaxH3/H3_13MP_VSR3K",
        audio=None,
    ) -> io.NodeOutput:
        mp4_path, report = stream_relay_vsr_to_mp4(
            vae,
            samples,
            images,
            audio=audio,
            detail=float(detail),
            frame_rate=float(frame_rate),
            crf=int(crf),
            filename_prefix=str(filename_prefix),
        )
        return io.NodeOutput(mp4_path, report)
