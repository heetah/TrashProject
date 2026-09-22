# -*- coding: utf-8 -*-
"""相容 shim:實作已搬至 pipeline.infra。

保留此路徑讓既有 `from examine import ...`(舊呼叫點/測試)在套件翻新期間持續可用。
新程式碼與 main.py 已改用 pipeline.infra。
"""
from pipeline import infra as _module

globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
del _module
