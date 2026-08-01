# -*- coding: utf-8 -*-
"""相容 shim:實作已搬至 pipeline.litter_tracker。

保留此路徑讓既有 `from litterTracker import GlobalLitterTracker, ...`(main.py、
tests/)在套件翻新期間持續可用。新程式碼請改用 pipeline.litter_tracker。
"""
from pipeline import litter_tracker as _module

globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
del _module
