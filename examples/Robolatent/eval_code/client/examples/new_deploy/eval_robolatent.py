#!/usr/bin/env python3
"""Robolatent 真机评估入口。

核心实现放在仓库根目录的 pi_infer.py，保留这个路径以匹配 Robolatent
评估脚本约定：

python examples/robolatent/eval_robolatent.py --host=127.0.0.1 --port=8000
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pi_infer import main


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
