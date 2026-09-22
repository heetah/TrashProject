# -*- coding: utf-8 -*-
"""相容 shim:實作已搬至 pipeline.profiling。

保留此路徑讓既有 `from timeUtils import ...`(含 main.py、detect.py、action.py、
licensePlate.py)在套件翻新期間持續可用。新程式碼請改用 pipeline.profiling。
底下把 pipeline.profiling 的所有頂層名稱(含底線前綴的內部工具)複製進本命名空間,
確保 re-export 行為與搬移前完全一致。
"""
from pipeline import profiling as _module

globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
del _module
