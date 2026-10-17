"""Read-only production overlay discovery for an isolated capture container.

Only source/binary overlays and explicitly named numerical-kernel settings are
inherited. No production caches, credentials, service lifecycle, or speculative
worker is used. The generated model overlay lives inside the research project.
"""

import argparse
import hashlib
import json
import shlex
import shutil
import sys
from pathlib import Path

from .prepare_overlay import patch

MODEL_DEST = "/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py"


def prepare(project, runtime, unit):
    project = Path(project)
    extra = []
    source = project / "ref/qwen4_exp-original.py"
    if runtime == "production":
        snapshot = project / "ref/production-capture-runtime"
        manifest = snapshot / "manifest.json"
        if manifest.exists():
            frozen = json.loads(manifest.read_text())
            source, extra = Path(frozen["source"]), frozen["arguments"]
            for name, expected in frozen["hashes"].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                    raise ValueError("pinned capture runtime has changed")
            command = []
        else:
            snapshot.mkdir(parents=True, exist_ok=True)
            command = None
        if command is None:
            line = next(
                x
                for x in Path(unit).read_text().splitlines()
                if x.startswith("ExecStart=")
            )
            command = shlex.split(line.removeprefix("ExecStart="))
        for index, item in enumerate(command):
            if item == "-v":
                mount = command[index + 1].split(":")
                origin, destination = mount[:2]
                if destination == MODEL_DEST:
                    source = Path(origin)
                elif (
                    destination.endswith((".py", ".so"))
                    or destination == "/opt/qwen-mtp-hotmap.json"
                ):
                    original = Path(origin)
                    digest = hashlib.sha256(original.read_bytes()).hexdigest()
                    pinned = snapshot / f"{digest}-{original.name}"
                    if not pinned.exists():
                        shutil.copyfile(original, pinned)
                    extra.extend(("-v", f"{pinned}:{destination}:ro"))
            elif item == "-e":
                setting = command[index + 1]
                name = setting.split("=", 1)[0]
                if name.startswith("QWENFAST_") or name in (
                    "QWENFAST",
                    "QWEN_NVFP4_MARLIN",
                ):
                    extra.extend(("-e", setting))
        if not manifest.exists():
            original = source
            source = snapshot / "qwen4_exp.py"
            shutil.copyfile(original, source)
            files = [source] + [
                Path(extra[i + 1].split(":")[0])
                for i, x in enumerate(extra)
                if x == "-v"
            ]
            frozen = {
                "source": str(source),
                "arguments": extra,
                "hashes": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
                },
            }
            temp = manifest.with_suffix(".tmp")
            temp.write_text(json.dumps(frozen, indent=2) + "\n")
            temp.replace(manifest)
    output = project / f"ref/qwen4_exp-capture-{runtime}.py"
    output.write_text(patch(source.read_text()))
    extra.extend(("-v", f"{output}:{MODEL_DEST}:ro"))
    hashes = {}
    for index, item in enumerate(extra):
        if item == "-v":
            origin = Path(extra[index + 1].split(":")[0])
            hashes[str(origin)] = hashlib.sha256(origin.read_bytes()).hexdigest()
    receipt = {"runtime": runtime, "source": str(source), "overlay_hashes": hashes}
    (project / f"ref/capture-runtime-{runtime}.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--runtime", choices=("base", "production"), default="base"
    )
    parser.add_argument(
        "--unit",
        default="./config/target.service",
    )
    args = parser.parse_args()
    sys.stdout.buffer.write(
        b"\0".join(x.encode() for x in prepare(args.project, args.runtime, args.unit))
        + b"\0"
    )


if __name__ == "__main__":
    main()
