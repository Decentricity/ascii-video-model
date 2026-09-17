#!/usr/bin/env python3
"""Orchestrate bug-blob attractor experiment phases.

prepare overfit → train overfit → diag → train sched → diag → motion → diag
→ delta → diag → patch2 → diag → decode_sweep → Phase 7 gate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PY = str(_ROOT / ".venv" / "bin" / "python")
STATUS = Path("/tmp/bugblob_status.txt")
LOG_DIR = Path("/tmp")


def status(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def run(
    cmd: list[str],
    *,
    log_name: str,
    cwd: Path = _ROOT,
    env: dict | None = None,
) -> int:
    log_path = LOG_DIR / log_name
    status(f"RUN {' '.join(cmd)} -> {log_path}")
    e = os.environ.copy()
    if env:
        e.update(env)
    e.setdefault("PYTHONPATH", str(_ROOT / "src"))
    with log_path.open("w") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=e,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    status(f"EXIT {proc.returncode} log={log_path}")
    return int(proc.returncode)


def train(config: str, log_name: str, max_steps: int | None = None, init: Path | None = None) -> int:
    cmd = [PY, "-m", "neurascii.train", "--config", config, "--device", "cuda"]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    if init is not None:
        cmd += ["--init-checkpoint", str(init)]
    return run(cmd, log_name=log_name)


def diag(ckpt: Path, processed: Path, tag: str, n_seeds: int = 2) -> int:
    cmd = [
        PY,
        str(_ROOT / "scripts" / "stability_diag.py"),
        "--checkpoint",
        str(ckpt),
        "--processed-dir",
        str(processed),
        "--tag",
        tag,
        "--n-seeds",
        str(n_seeds),
        "--device",
        "cuda",
    ]
    return run(cmd, log_name=f"bugblob_diag_{tag}.log")


def read_summary(tag: str) -> dict:
    path = _ROOT / "samples" / "rollouts" / tag / f"{tag}_stability_summary.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def best_val(run_dir: Path) -> float | None:
    ckpt = run_dir / "ckpt_best.pt"
    if not ckpt.is_file():
        ckpt = run_dir / "ckpt_last.pt"
    if not ckpt.is_file():
        return None
    import torch

    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    return float(obj.get("best_val", float("inf")))


def aggregate_h96(summary: dict) -> dict:
    agg = summary.get("aggregate") or {}
    return {
        "mean_any_change": agg.get("mean_any_change"),
        "mean_blob_growth_slope": agg.get("mean_blob_growth_slope"),
        "mean_blob_final_frac": agg.get("mean_blob_final_frac"),
        "frac_bug_blob_signature": agg.get("frac_bug_blob_signature"),
    }


def gate_decision(results: dict) -> dict:
    """PASS if blob growth reduced AND motion persists vs overfit/baseline failure."""
    overfit = results.get("overfit") or results.get("overfit_eye") or {}
    candidates = ["eye_sched", "eye_motion", "eye_delta", "eye_patch2"]
    overfit_slope = (overfit.get("diag") or {}).get("mean_blob_growth_slope")
    overfit_change = (overfit.get("diag") or {}).get("mean_any_change")

    winners = []
    for name in candidates:
        d = (results.get(name) or {}).get("diag") or {}
        slope = d.get("mean_blob_growth_slope")
        change = d.get("mean_any_change")
        if slope is None or change is None:
            continue
        # fail signature: growth and frozen motion
        still_blob = slope > 0 and change is not None and change < 0.01
        improved = False
        if overfit_slope is not None:
            improved = slope < overfit_slope * 0.5 and change >= 0.01
        else:
            improved = slope <= 0 and change >= 0.01
        if improved and not still_blob:
            winners.append(name)

    # Gate FAIL / SKIP fullvideo if overfit OR all improved runs still blob
    overfit_fails = (
        overfit_slope is not None
        and overfit_slope > 0
        and overfit_change is not None
        and overfit_change < 0.01
    )
    any_still = False
    for name in candidates:
        d = (results.get(name) or {}).get("diag") or {}
        slope = d.get("mean_blob_growth_slope")
        change = d.get("mean_any_change")
        if slope is not None and change is not None and slope > 0 and change < 0.01:
            any_still = True

    if winners:
        decision = "PASS"
        reason = f"improved runs without bug-blob signature: {winners}"
        best = winners[0]
        # prefer motion/delta
        for pref in ("eye_motion", "eye_delta", "eye_sched", "eye_patch2"):
            if pref in winners:
                best = pref
                break
    elif overfit_fails or any_still or not winners:
        decision = "SKIP_FULLVIDEO"
        reason = (
            "overfit or narrow runs still show blob_growth "
            "(largest region increases and mean_any_change at h96 < 0.01)"
        )
        best = None
    else:
        decision = "SKIP_FULLVIDEO"
        reason = "no clear winner"
        best = None

    return {"decision": decision, "reason": reason, "best_config": best, "winners": winners}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-train", action="store_true", help="Only diag existing ckpts")
    p.add_argument("--max-steps-override", type=int, default=None)
    p.add_argument("--phases", nargs="+", default=None, help="Subset of phases to run")
    args = p.parse_args(argv)

    STATUS.write_text(f"started {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    results: dict = {}

    # Preload any existing diag summaries (e.g. overfit from prior run)
    for tag in ("overfit", "eye_sched", "eye_motion", "eye_delta", "eye_patch2"):
        summary = read_summary(tag)
        if summary:
            run_guess = {
                "overfit": "runs/overfit_eye",
                "eye_sched": "runs/eye_sched",
                "eye_motion": "runs/eye_motion",
                "eye_delta": "runs/eye_delta",
                "eye_patch2": "runs/eye_patch2",
            }[tag]
            results[tag] = {
                "train_rc": 0,
                "diag_rc": 0,
                "best_val": best_val(_ROOT / run_guess),
                "ckpt": summary.get("checkpoint"),
                "diag": aggregate_h96(summary),
                "preloaded": True,
            }
            status(f"preloaded {tag} diag={results[tag]['diag']}")

    phases = args.phases or [
        "prepare",
        "overfit",
        "sched",
        "motion",
        "delta",
        "patch2",
        "decode",
        "gate",
    ]

    if "prepare" in phases:
        status("phase=prepare_overfit")
        rc = run(
            [PY, str(_ROOT / "scripts" / "prepare_overfit_eye.py")],
            log_name="bugblob_prepare.log",
        )
        if rc != 0:
            status("FAIL prepare")
            return rc

    plan = [
        ("overfit", "overfit", "configs/train_overfit_eye.yaml", "data/processed/overfit_eye", "runs/overfit_eye", None),
        ("sched", "eye_sched", "configs/train_eye_sched.yaml", "data/processed/poc_eye", "runs/eye_sched", "runs/overfit_eye/ckpt_best.pt"),
        ("motion", "eye_motion", "configs/train_eye_motion.yaml", "data/processed/poc_eye", "runs/eye_motion", "runs/overfit_eye/ckpt_best.pt"),
        ("delta", "eye_delta", "configs/train_eye_delta.yaml", "data/processed/poc_eye", "runs/eye_delta", "runs/overfit_eye/ckpt_best.pt"),
        # patch2 has different n_patches — no warm-start
        ("patch2", "eye_patch2", "configs/train_eye_patch2.yaml", "data/processed/poc_eye", "runs/eye_patch2", None),
    ]

    for phase_key, tag, config, processed, run_dir, init_rel in plan:
        if phase_key not in phases and tag not in phases:
            continue
        status(f"phase={tag}")
        run_path = _ROOT / run_dir
        if not args.skip_train:
            init = (_ROOT / init_rel) if init_rel and (_ROOT / init_rel).is_file() else None
            if init is not None:
                status(f"warm_start {tag} from {init}")
            rc = train(
                str(_ROOT / config),
                f"bugblob_train_{tag}.log",
                max_steps=args.max_steps_override,
                init=init,
            )
            if rc != 0:
                status(f"FAIL train {tag} rc={rc}")
                results[tag] = {"train_rc": rc}
                continue
        ckpt = run_path / "ckpt_best.pt"
        if not ckpt.is_file():
            ckpt = run_path / "ckpt_last.pt"
        if not ckpt.is_file():
            status(f"FAIL no ckpt for {tag}")
            results[tag] = {"error": "no_ckpt"}
            continue
        rc = diag(ckpt, _ROOT / processed, tag=tag, n_seeds=2)
        summary = read_summary(tag)
        results[tag] = {
            "train_rc": 0,
            "diag_rc": rc,
            "best_val": best_val(run_path),
            "ckpt": str(ckpt),
            "diag": aggregate_h96(summary),
            "init": init_rel,
        }
        status(f"done {tag} best_val={results[tag]['best_val']} diag={results[tag]['diag']}")

    if "decode" in phases:
        status("phase=decode_sweep")
        # Use best available of motion/delta/sched/overfit
        ckpt = None
        for cand in (
            _ROOT / "runs/eye_motion/ckpt_best.pt",
            _ROOT / "runs/eye_delta/ckpt_best.pt",
            _ROOT / "runs/eye_sched/ckpt_best.pt",
            _ROOT / "runs/overfit_eye/ckpt_best.pt",
            _ROOT / "runs/poc_eye/ckpt_best.pt",
        ):
            if cand.is_file():
                ckpt = cand
                break
        if ckpt:
            run(
                [
                    PY,
                    str(_ROOT / "scripts" / "decode_sweep.py"),
                    "--checkpoint",
                    str(ckpt),
                    "--processed-dir",
                    str(_ROOT / "data/processed/poc_eye"),
                    "--device",
                    "cuda",
                ],
                log_name="bugblob_decode_sweep.log",
            )
        else:
            status("SKIP decode_sweep (no ckpt)")

    gate = gate_decision(results)
    status(f"phase=gate decision={gate['decision']}")

    gate_path = _ROOT / "samples" / "rollouts" / "bugblob_gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": results,
        "gate": gate,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    gate_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if gate["decision"] == "PASS" and gate.get("best_config"):
        status(f"Phase7 PASS — starting fullvideo warm-start from {gate['best_config']}")
        best_cfg = gate["best_config"]
        init = _ROOT / "runs/fullvideo/ckpt_best.pt"
        # Train fullvideo with best eye config knobs but fullvideo data — use a temp approach:
        # warm-start fullvideo config from best eye run weights is wrong shape if patch2.
        # Prefer eye_motion/eye_delta (same 4x4 as fullvideo).
        if best_cfg == "eye_patch2":
            status("best is patch2 — skip fullvideo warm-start (patch mismatch)")
            gate["decision"] = "SKIP_FULLVIDEO"
            gate["reason"] = "best config is patch2 (incompatible with fullvideo 4x4)"
            payload["gate"] = gate
            gate_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        else:
            # Create a short fullvideo finetune config in runs/
            fv_cfg_path = _ROOT / "configs" / "train_fullvideo_bugblob.yaml"
            fv_cfg_path.write_text(
                (_ROOT / "configs" / "train_fullvideo.yaml").read_text()
                + "\n# bugblob warm-start overrides injected by run_bugblob_phases\n",
                encoding="utf-8",
            )
            # Prefer copying train knobs from best eye yaml into a dedicated config
            import yaml

            base = yaml.safe_load((_ROOT / "configs" / "train_fullvideo.yaml").read_text())
            eye = yaml.safe_load((_ROOT / "configs" / f"train_{best_cfg}.yaml").read_text())
            for k in (
                "change_weight",
                "use_delta",
                "scheduled_sampling_p_max",
                "scheduled_sampling_warmup_steps",
                "rollout_train_steps",
                "rollout_train_steps_max",
            ):
                if k in eye.get("train", {}):
                    base["train"][k] = eye["train"][k]
            base["data"]["future_frames"] = eye.get("data", {}).get("future_frames", 1)
            base["paths"]["run_dir"] = "runs/fullvideo_bugblob"
            base["train"]["max_steps"] = 3000
            fv_cfg_path.write_text(yaml.safe_dump(base), encoding="utf-8")
            rc = train(
                str(fv_cfg_path),
                "bugblob_train_fullvideo.log",
                max_steps=args.max_steps_override or 3000,
                init=init if init.is_file() else None,
            )
            results["fullvideo_bugblob"] = {"train_rc": rc}
            if rc == 0:
                diag(
                    _ROOT / "runs/fullvideo_bugblob/ckpt_best.pt",
                    _ROOT / "data/processed/fullvideo",
                    tag="fullvideo_bugblob",
                    n_seeds=4,
                )
                # carousel + site only if phase 7 ran successfully
                run(
                    [PY, str(_ROOT / "scripts" / "carousel_fullvideo.py")],
                    log_name="bugblob_carousel.log",
                )
    else:
        status(f"Phase7 SKIP fullvideo: {gate['reason']}")

    payload["results"] = results
    gate_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    status("ALL_DONE")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
