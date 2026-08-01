# -*- coding: utf-8 -*-
"""相容 shim:實作已搬至 pipeline.action。

保留此路徑讓既有 `from action import STGCNActionModule`(main.py)在套件翻新期間
持續可用。新程式碼請改用 pipeline.action。
"""
from pipeline import action as _module

globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
del _module
