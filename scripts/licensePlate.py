# -*- coding: utf-8 -*-
"""相容 shim:實作已搬至 pipeline.plate。

保留此路徑讓既有 `from licensePlate import ...`(main.py、detect.py)在套件翻新
期間持續可用。新程式碼請改用 pipeline.plate。
"""
from pipeline import plate as _module

globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
del _module
