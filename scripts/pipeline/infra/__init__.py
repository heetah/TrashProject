# -*- coding: utf-8 -*-
"""基礎設施子套件。原 pipeline/infra.py 拆成 constants/video_io/motion/models;
此處 re-export 全部名稱(含底線前綴),讓 `from pipeline.infra import ...`(main.py)與
examine.py 相容 shim 維持不變。
"""
from . import constants as _constants
from . import contracts as _contracts
from . import video_io as _video_io
from . import motion as _motion
from . import models as _models

for _m in (_constants, _contracts, _video_io, _motion, _models):
    globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
del _m, _constants, _contracts, _video_io, _motion, _models
