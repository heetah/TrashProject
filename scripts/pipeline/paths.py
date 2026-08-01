# -*- coding: utf-8 -*-
"""集中管理專案路徑與 mmaction2 bootstrap。

搬進套件前,action.py 與 litterTracker.py 各自用 __file__ 相對路徑推算 repo root
並把 mmaction2 塞進 sys.path;搬進 pipeline/ 後檔案深度改變,若各自 hard-code parent
層數容易漂移出錯。集中在此一處計算,單一事實來源。
"""
import os
import sys

# 本檔位於 <repo>/scripts/pipeline/paths.py。
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))          # scripts/pipeline
SCRIPTS_DIR = os.path.dirname(PACKAGE_DIR)                         # scripts
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)                          # <repo>
MMACTION_REPO = os.path.join(REPO_ROOT, "mmaction2")


def ensure_mmaction_on_path():
    """把專案內的 mmaction2 加到 sys.path 最前面(若尚未加入),回傳其路徑。

    使用專案內 mmaction2,避免吃到系統其他版本。
    """
    if MMACTION_REPO not in sys.path:
        sys.path.insert(0, MMACTION_REPO)
    return MMACTION_REPO
