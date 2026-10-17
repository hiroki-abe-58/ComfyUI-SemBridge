"""CPU "real queue" check for CI and for a clean-install test: start a real ComfyUI server, install this node pack into it,
submit the shipped API workflows through POST /prompt and check the results.

    python scripts/ci_queue_check.py --comfyui <ComfyUI checkout> --tokenizer-dir <official tokenizer files> [--keep-going]

The model is the tiny random-initialized stand-in from tests/tiny_model.py (no trained weights, so the audio is noise): this
proves that the nodes load in a stock ComfyUI, appear in /object_info, accept the shipped workflows and run end to end on CPU.
The real-weights runs are documented in docs/VERIFICATION.md. The server is started and stopped by this script only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import queue_client  # noqa: E402

TINY = {"max_new_patches": 6, "min_new_patches": 2, "flow_steps": 2}


def wait_up(server: str, proc: subprocess.Popen, timeout: float = 300) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"ComfyUI exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"{server}/system_stats", timeout=3).read()
            return
        except Exception:
            time.sleep(2)
    raise SystemExit("ComfyUI did not start in time")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8190)
    parser.add_argument("--workdir", type=Path, default=None, help="where to put the ComfyUI copy's logs (default: a temp folder)")
    args = parser.parse_args(argv)

    import tiny_model

    comfy = args.comfyui.resolve()
    pack = comfy / "custom_nodes" / "ComfyUI-SemBridge"
    if pack.exists():
        shutil.rmtree(pack)
    shutil.copytree(ROOT, pack, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs"))
    model_dir = comfy / "models" / "sembridge" / "SemBridge"
    tiny_model.build(model_dir, args.tokenizer_dir)

    log_dir = args.workdir or Path(tempfile.mkdtemp(prefix="sembridge-ci-"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log = (log_dir / "comfyui.log").open("w", encoding="utf-8")
    server = f"http://127.0.0.1:{args.port}"
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(args.port), "--disable-auto-launch", "--disable-api-nodes"],
        cwd=comfy, stdout=log, stderr=subprocess.STDOUT,
    )
    failures = []
    try:
        wait_up(server, proc)
        info = json.loads(urllib.request.urlopen(f"{server}/object_info", timeout=30).read())
        expected = {"SemBridgeLoader", "SemBridgeTTS", "SemBridgeSinging", "SemBridgeUnified", "SemBridgeScorePreview", "SemBridgeUnload"}
        missing = expected - set(info)
        print("registered:", sorted(expected & set(info)))
        if missing:
            failures.append(f"nodes missing from /object_info: {sorted(missing)}")
        spec = info["SemBridgeLoader"]["input"]["required"]["model_dir"]
        listed = spec[1]["options"] if spec[0] == "COMBO" else spec[0]
        if "SemBridge" not in listed:
            failures.append(f"loader does not list the model folder: {listed}")

        for name in ("score_preview", "tts_basic", "singing_basic", "unified_basic"):
            workflow = json.loads((ROOT / "workflows" / "api" / f"{name}.json").read_text(encoding="utf-8"))
            for node in workflow.values():
                if node["class_type"] == "SemBridgeLoader":
                    node["inputs"]["device"] = "cpu"
                elif node["class_type"] in ("SemBridgeTTS", "SemBridgeSinging", "SemBridgeUnified"):
                    node["inputs"].update(TINY)
                    node["inputs"]["length_control"] = "manual" if node["class_type"] == "SemBridgeSinging" else None
                    if node["inputs"]["length_control"] is None:
                        del node["inputs"]["length_control"]
            result = queue_client.run(server, workflow, log_dir / "out", 600)
            ok = result["status"] == "success"
            print(json.dumps({"workflow": name, "status": result["status"], "seconds": result["wall_seconds"], "files": len(result["files"]), "error": result.get("error")}))
            if not ok:
                failures.append(f"{name}: {result.get('error') or result['status']}")
                continue
            if name != "score_preview":
                if not result["files"]:
                    failures.append(f"{name}: no audio saved")
                report = json.loads(next(iter(result["text"].values()))[0])
                if report["sample_rate"] != 44100 or report["audio_seconds"] <= 0:
                    failures.append(f"{name}: bad report {report}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        print((log_dir / "comfyui.log").read_text(encoding="utf-8", errors="replace")[-4000:])
        return 1
    print("queue check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
