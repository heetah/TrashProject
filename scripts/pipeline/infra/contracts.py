# -*- coding: utf-8 -*-
"""影片前處理階段在有界 queue 中傳遞的資料契約。"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PreparedFrame:
    """一幀未標註來源影像，以及可在背景執行緒先完成的 CPU 前處理。"""

    index: int
    source_bgr: np.ndarray
    foreground_mask: np.ndarray
    litter_model_input: Optional[np.ndarray] = None
