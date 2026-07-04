# -*- coding: utf-8 -*-
"""相容 shim:實作已搬至 pipeline.detect。

保留此路徑讓既有 `from detect import detect_batch`(main.py)與
`from detect import compute_pixel_change_map`(tests/)在套件翻新期間持續可用。
新程式碼請改用 pipeline.detect。
"""
from pipeline import detect as _module

globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
del _module
