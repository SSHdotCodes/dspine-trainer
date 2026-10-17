"""Absolute-path launcher that does not depend on ai-job's working directory."""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
module = sys.argv.pop(1)
runpy.run_module(module, run_name="__main__")
