"""Golden Set 评测入口：python scripts/run_evals.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.eval_runner import run_all

if __name__ == "__main__":
    run_all()
