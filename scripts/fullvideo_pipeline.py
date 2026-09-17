#!/usr/bin/env python3
"""Phase C orchestrator: preprocess → train → carousel → site push.

Logs: /tmp/neurascii-fullvideo-pipeline.log
PID:  /tmp/neurascii-fullvideo-pipeline.pid
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/home/decentricity/neurascii")
PAGES = Path("/home/decentricity/neurascii.github.io")
VENV_PY = ROOT / ".venv" / "bin" / "python"
RAW = ROOT / "data/raw/UCF101Fullvideo"
OUT = ROOT / "data/processed/fullvideo"
LOG = Path("/tmp/neurascii-fullvideo-pipeline.log")
PRE_LOG = Path("/tmp/neurascii-fullvideo-preprocess.log")
TRAIN_LOG = Path("/tmp/neurascii-fullvideo-train.log")
CAR_LOG = Path("/tmp/neurascii-fullvideo-carousel.log")
N_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 2))


def log(msg: str, path: Path = LOG) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def announce(msg: str) -> None:
    try:
        sys.path.insert(0, str(Path.home() / ".local/share/voice-announce"))
        from voice_announce import gpu_is_idle, speak_with_piper

        if gpu_is_idle():
            speak_with_piper(msg)
        else:
            subprocess.run(["/home/decentricity/bin/say-alert", msg], check=False)
    except Exception as e:
        log(f"announce_failed: {e}")


def _convert_one(args: tuple[str, str, str]) -> tuple[str, str | None, str | None]:
    """Worker: (src, out_path, config_yaml) -> (src, out_or_none, err_or_none)."""
    src_s, out_s, cfg_s = args
    try:
        sys.path.insert(0, str(ROOT / "src"))
        import yaml
        from neurascii.caca_convert import convert_and_save
        from neurascii.format import RendererConfig

        raw = yaml.safe_load(Path(cfg_s).read_text()) or {}
        cfg = RendererConfig.from_dict(raw.get("renderer", raw))
        out = Path(out_s)
        if out.is_file() and out.stat().st_size > 0:
            return src_s, out_s, None
        convert_and_save(Path(src_s), out, cfg=cfg)
        return src_s, out_s, None
    except Exception as e:
        return src_s, None, f"{type(e).__name__}: {e}"


def preprocess() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PRE_LOG.write_text("", encoding="utf-8")
    files: list[Path] = []
    for ext in ("*.avi", "*.mp4", "*.mkv", "*.webm", "*.mov"):
        files.extend(sorted(RAW.rglob(ext)))
    # de-dupe
    seen: set[Path] = set()
    uniq: list[Path] = []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(f)
    files = uniq
    log(f"preprocess start n={len(files)} workers={N_WORKERS} out={OUT}", PRE_LOG)
    log(f"preprocess start n={len(files)} workers={N_WORKERS}")

    cfg_path = str(ROOT / "configs/preprocess.yaml")
    jobs: list[tuple[str, str, str]] = []
    for f in files:
        parent = f.parent.name
        out_name = f"{parent}__{f.stem}.avm.npz" if parent not in (".", "") else f"{f.stem}.avm.npz"
        jobs.append((str(f), str(OUT / out_name), cfg_path))

    manifest: list[dict] = []
    errors = 0
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_convert_one, j): j for j in jobs}
        for fut in as_completed(futs):
            src, out, err = fut.result()
            done += 1
            if err:
                errors += 1
                log(f"ERROR {src}: {err}", PRE_LOG)
            else:
                p = Path(out)  # type: ignore[arg-type]
                parent = Path(src).parent.name
                manifest.append(
                    {
                        "source": src,
                        "avm": out,
                        "class": parent,
                        "stem": Path(src).stem,
                    }
                )
            if done % 100 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(jobs) - done) / max(rate, 1e-6)
                msg = f"preprocess {done}/{len(jobs)} ok={len(manifest)} err={errors} rate={rate:.2f}/s eta={eta/60:.1f}m"
                log(msg, PRE_LOG)
                log(msg)

    man_path = OUT / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    log(f"preprocess done wrote={len(manifest)} errors={errors} -> {man_path}", PRE_LOG)
    log(f"preprocess done wrote={len(manifest)} errors={errors}")
    if len(manifest) < 1000:
        raise SystemExit(f"preprocess produced too few clips: {len(manifest)}")
    return len(manifest)


def train() -> None:
    TRAIN_LOG.write_text("", encoding="utf-8")
    log("train start")
    announce("Fullvideo preprocess finished. Starting GPU training.")
    cmd = [
        str(VENV_PY),
        "-m",
        "neurascii.train",
        "--config",
        "configs/train_fullvideo.yaml",
        "--device",
        "cuda",
    ]
    # optional warm-start from videos_all if present
    warm = ROOT / "runs/videos_all/ckpt_best.pt"
    if warm.is_file():
        cmd.extend(["--init-checkpoint", str(warm)])
        log(f"warm-start from {warm}")
    with TRAIN_LOG.open("a", encoding="utf-8") as f:
        f.write("+ " + " ".join(cmd) + "\n")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with TRAIN_LOG.open("a", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise SystemExit(f"train failed rc={proc.returncode}; see {TRAIN_LOG}")
    best = ROOT / "runs/fullvideo/ckpt_best.pt"
    if not best.is_file():
        raise SystemExit("train finished but ckpt_best.pt missing")
    log(f"train done best={best}")


def carousel() -> list[tuple[str, str]]:
    CAR_LOG.write_text("", encoding="utf-8")
    log("carousel start")
    announce("Fullvideo training finished. Generating carousel GIFs.")
    with CAR_LOG.open("a", encoding="utf-8") as f:
        proc = subprocess.run(
            [str(VENV_PY), str(ROOT / "scripts/carousel_fullvideo.py")],
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise SystemExit(f"carousel failed rc={proc.returncode}; see {CAR_LOG}")
    caps_path = PAGES / "assets/gifs/fullvideo_captions.txt"
    if not caps_path.is_file():
        raise SystemExit(f"missing captions {caps_path}")
    pairs: list[tuple[str, str]] = []
    for line in caps_path.read_text().splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        idx, cap = line.split("|", 1)
        pairs.append((idx, cap))
    log(f"carousel done n={len(pairs)}")
    return pairs


def update_site(pairs: list[tuple[str, str]]) -> None:
    log("updating site index.html")
    index = PAGES / "index.html"
    html = index.read_text(encoding="utf-8")

    # Build carousel section from captions; gif names from carousel_fullvideo.py
    slides = []
    for idx, cap in pairs:
        # caption like "Class / name"; class is first token before " / "
        cls = cap.split(" / ", 1)[0].strip()
        gif = f"assets/gifs/fullvideo_{idx}_{cls}.gif"
        slides.append(
            f'        <div class="carousel-slide" data-caption="{cap}">'
            f'<img src="{gif}" alt="{cls} fullvideo rollout" width="800"></div>'
        )
    first_cap = pairs[0][1] if pairs else ""
    section = f"""
  <h2>UCF101 Fullvideo (Phase C)</h2>
  <div class="carousel" id="fullvideo-carousel" aria-roledescription="carousel" aria-label="UCF101 Fullvideo rollouts">
    <div class="carousel-viewport">
      <div class="carousel-track">
{chr(10).join(slides)}
      </div>
    </div>
    <p class="carousel-caption">{first_cap}</p>
    <div class="carousel-nav">
      <button type="button" data-dir="-1" aria-label="Previous fullvideo rollout">Prev</button>
      <button type="button" data-dir="1" aria-label="Next fullvideo rollout">Next</button>
      <div class="carousel-dots" role="tablist" aria-label="Select fullvideo rollout"></div>
    </div>
  </div>
  <p>Next-frame train on bitmind/UCF101Fullvideo (~13k clips / 101 classes), then autoregressive rollout. Carousel: 10 diverse class seeds · sampler greedy (temp=0); multi-horizon stability diagnostics under samples/rollouts/.</p>
"""

    marker = "  <h2>Hypothesis</h2>"
    if "id=\"fullvideo-carousel\"" in html:
        # replace existing fullvideo section (from h2 through next h2)
        import re

        html = re.sub(
            r"\n  <h2>UCF101 Fullvideo \(Phase C\)</h2>.*?(?=\n  <h2>)",
            "\n" + section.lstrip("\n"),
            html,
            count=1,
            flags=re.S,
        )
    else:
        if marker not in html:
            raise SystemExit("Hypothesis marker missing in index.html")
        html = html.replace(marker, section + "\n" + marker, 1)

    # Update data ladder line for Fullvideo
    html = html.replace(
        "<li><a href=\"https://huggingface.co/datasets/bitmind/UCF101Fullvideo\">UCF101Fullvideo</a> (~7GB) — next</li>",
        "<li><a href=\"https://huggingface.co/datasets/bitmind/UCF101Fullvideo\">UCF101Fullvideo</a> (~7GB) — current Phase C</li>",
    )
    index.write_text(html, encoding="utf-8")
    log("site html updated")


def git_push_site() -> None:
    log("git commit+push site")
    subprocess.run(["git", "add", "index.html", "assets/gifs"], cwd=str(PAGES), check=True)
    # only commit if changes
    st = subprocess.run(["git", "status", "--porcelain"], cwd=str(PAGES), capture_output=True, text=True)
    if not st.stdout.strip():
        log("site: nothing to commit")
        return
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Add UCF101 Fullvideo Phase C carousel (10 class GIFs).",
        ],
        cwd=str(PAGES),
        check=True,
    )
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=str(PAGES), check=True)
    log("site pushed")


def git_push_neurascii() -> None:
    log("git commit+push neurascii code/config")
    files = [
        "configs/train_fullvideo.yaml",
        "scripts/carousel_fullvideo.py",
        "scripts/fullvideo_stability_diag.py",
        "scripts/phase_c_regression.py",
        "scripts/download_ucf101_fullvideo.py",
        "src/neurascii/train.py",
        "src/neurascii/generate.py",
        "AGENTS.md",
    ]
    # optional carousel helpers from earlier phases
    optional = [
        "scripts/carousel_eye_lipstick.py",
        "scripts/carousel_eye_lipstick.sh",
        "scripts/carousel_phase_b.sh",
        "scripts/carousel_smoke_apply.py",
    ]
    existing = [f for f in files + optional if (ROOT / f).exists()]
    subprocess.run(["git", "add", *existing], cwd=str(ROOT), check=True)
    st = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT), capture_output=True, text=True)
    # only commit staged/relevant
    if not any(line and not line.endswith("/") for line in st.stdout.splitlines()):
        # check if staged
        pass
    st2 = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=str(ROOT), capture_output=True, text=True)
    if not st2.stdout.strip():
        log("neurascii: nothing to commit")
        return
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Phase C: fullvideo train config, carousel, download script, init-checkpoint.",
        ],
        cwd=str(ROOT),
        check=True,
    )
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=str(ROOT), check=True)
    log("neurascii pushed")


def main() -> int:
    Path("/tmp/neurascii-fullvideo-pipeline.pid").write_text(str(os.getpid()))
    LOG.write_text("")
    log("=== Phase C pipeline start ===")
    announce("Starting Fullvideo preprocess for Phase C.")
    try:
        # Push code early so configs are on GitHub even if later stages fail
        try:
            git_push_neurascii()
        except Exception as e:
            log(f"early code push failed (will retry later): {e}")

        n = preprocess()
        log(f"preprocess ok n={n}")
        train()
        pairs = carousel()
        if len(pairs) < 10:
            raise SystemExit(f"expected 10 captions, got {len(pairs)}")
        update_site(pairs)
        git_push_site()
        try:
            git_push_neurascii()
        except Exception as e:
            log(f"final code push note: {e}")
        announce("Phase C complete. Fullvideo carousel is live on the site.")
        log("=== Phase C pipeline DONE ===")
        log("site=https://neurascii.github.io/")
        return 0
    except Exception:
        log("PIPELINE FAILED:\n" + traceback.format_exc())
        announce("Phase C pipeline failed. Check the full video logs.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
