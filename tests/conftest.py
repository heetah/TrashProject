# -*- coding: utf-8 -*-
"""讓根目錄 tests/ 可直接匯入 scripts/ 內的 production modules。"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
